from collate import (
    _tokenization_tr, 
    _tokenization_ev
)
from transformers import (
    Trainer, 
    TrainerCallback
)
from torch.utils.data import DataLoader
import evaluate, torch, pandas as pd, os, gc
from bert_score import score as bert_score_compute

def safe_divide(numerator, denominator, default=0.0):
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

def calculate_bertscore(predictions, references):
    try:
        P, R, F1 = bert_score_compute(
            predictions, 
            references, 
            device="cpu",
            nthreads=8,
            batch_size=256,
            lang="en", 
            model_type="neuml/pubmedbert-base-embeddings",
            verbose=False,
            use_fast_tokenizer=True
        )
        precision_avg = safe_divide(P.sum().item(), len(P))
        recall_avg = safe_divide(R.sum().item(), len(R))
        f1_avg = safe_divide(F1.sum().item(), len(F1))
        
        return {
            "precision": precision_avg,
            "recall": recall_avg,
            "f1": f1_avg
        }
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

def log_metrics(pred, ref, metrics_dict, state, append=True, save_path=None):
    predictions, references = [], []
    predictions.extend(pred)
    references.extend(ref['suffixes'])
    results = {}

    try:
        for name, metric in metrics_dict.items():
            if name == "bertscore":
                results[name] = calculate_bertscore(predictions, references)
            else:
                results[name] = metric.compute(
                    predictions=predictions,
                    references=references
                )
        print("Metrics:", results)
        df = pd.DataFrame({
            "epoch": [state.epoch] * len(predictions),
            "step": [state.global_step] * len(predictions),
            "reference": references,
            "generated": predictions
        })
        for name, score_dict in results.items():
            if isinstance(score_dict, dict):
                for k, v in score_dict.items():
                    df[f"{name}_{k}"] = [v] * len(predictions)
            else:
                df[name] = [score_dict] * len(predictions)
        if append:
            df.to_csv(
                save_path, 
                mode='a', 
                index=False,
                header=not os.path.exists(save_path)
            )
        else:
            df.to_csv(save_path, index=False)

    except Exception as e:
        print(f"Error during evaluation: {e}")

    finally:
        del predictions, references, ref, pred, results, df
        gc.collect()


class GenerationCallback(TrainerCallback):
    def __init__(self, processor, _metrics_path):
        self.processor = processor
        self._metrics_path = _metrics_path
        self.metrics = {
            "rouge": evaluate.load("rouge"),
            "bleu": evaluate.load("bleu"),
            "meteor": evaluate.load("meteor"),
        }

    @torch.no_grad()
    @torch.amp.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=True
    )
    def on_evaluate(self, args, state, control, **kwargs):
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = 'left'
        
        try:
            model = kwargs['model']
            eval_dataloader = kwargs['eval_dataloader']
            model.eval()

            all_predictions = []
            all_references = []

            for i, batch in enumerate(eval_dataloader):
                if i >= 20:  
                    break
                print(f"Evaluating batch {i}...")
                
                device = model.device
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }

                if 'pixel_values' in batch and batch['pixel_values'] is not None:
                    inputs['pixel_values'] = batch['pixel_values'].to(device)

                if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
                    inputs['image_grid_thw'] = batch['image_grid_thw'].to(device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    output_scores=False,
                    return_dict_in_generate=False,
                )

                input_lengths = inputs['attention_mask'].sum(dim=1)
                generated_ids_trimmed = []
                
                for input_len, out_ids in zip(input_lengths, generated_ids):
                    if len(out_ids) > input_len:
                        generated_ids_trimmed.append(out_ids[input_len:])
                    else:
                        generated_ids_trimmed.append(torch.tensor([], device=out_ids.device, dtype=out_ids.dtype))

                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True   
                )
                cleaned_outputs = []
                for text in output_text:
                    cleaned = text.strip()
                    cleaned = cleaned.replace('<|endoftext|>', '').replace('<|im_end|>', '').replace('Describe the following image.', '').strip()
                    cleaned_outputs.append(cleaned)
                
                all_predictions.extend(cleaned_outputs)
                all_references.extend(batch['suffixes'])
                
                del generated_ids, generated_ids_trimmed, inputs
                torch.cuda.empty_cache()

            if all_predictions and all_references:
                log_metrics(
                    pred=all_predictions,
                    ref=all_references,
                    metrics_dict=self.metrics,
                    state=state,
                    save_path=self._metrics_path
                )
            else:
                print("No predictions generated for evaluation")

        except Exception as e:
            print(f"Error in evaluation (non-fatal): {e}")
            import traceback
            traceback.print_exc()

        finally:
            del cleaned_outputs, all_predictions, all_references
            self.processor.tokenizer.padding_side = original_padding_side
            torch.cuda.empty_cache()


class CustomTrainer(Trainer):
    def __init__(
        self, 
        processor, 
        num_workers_tr,
        num_workers_ev,
        pin_memory_tr,
        pin_memory_ev,
        persistent_tr,
        persistent_ev,
        *args, 
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.processor = processor

        self.num_workers_tr=num_workers_tr
        self.num_workers_ev=num_workers_ev

        self.is_pin_memory_tr=pin_memory_tr
        self.is_pin_memory_ev=pin_memory_ev

        self.is_persistent_tr=persistent_tr
        self.is_persistent_ev=persistent_ev

    def get_train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=lambda b: _tokenization_tr(b, self.processor),
            num_workers=self.num_workers_tr,
            pin_memory=self.is_pin_memory_tr,
            persistent_workers=self.is_persistent_tr
        )
    
    def get_eval_dataloader(self, eval_dataset=None):

        return DataLoader(
            eval_dataset or self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=True,
            collate_fn=lambda b: _tokenization_ev(b, self.processor),
            num_workers=self.num_workers_ev,
            pin_memory=self.is_pin_memory_tr,
            persistent_workers=self.is_persistent_ev
        )
