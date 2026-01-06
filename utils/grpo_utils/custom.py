from trl import (
    GRPOTrainer
)
from transformers import TrainerCallback

from .. qwen_utils.dataloader import (
    _tokenization_tr, 
    _tokenization_ev
)
from .. qwen_utils.bert_embedding import compute_prf1
from torch.utils.data import DataLoader
import torch, os, gc, pprint, traceback, pandas as pd, time
from loguru import logger


def safe_divide(numerator, denominator, default=0.0):
    try:
        if denominator != 0:
            logger.info(f"Denominator is {denominator}")
        return numerator / denominator if denominator != 0 else default
    
    except (TypeError, ZeroDivisionError):
        logger.error(f"Zero division for metrics, returning {default}.")
        traceback.print_exc()
        return default

def calculate_bertscore(predictions, references):
    logger.info("Calculating P, R, F1 for this interation...")
    try:
        P, R, F1 = compute_prf1(
            cands=predictions,
            refs=references,
            model_path="./models/pubmedbert-base-embeddings",
            max_length=512,
            batch_size=25,
            device="cpu",
            fp16=False
        )

        precision_avg           = safe_divide(P.sum().item(), len(P))
        recall_avg              = safe_divide(R.sum().item(), len(R))
        f1_avg                  = safe_divide(F1.sum().item(), len(F1))
        
        return {
            "precision": precision_avg,
            "recall": recall_avg,
            "f1": f1_avg
        }
    
    except Exception as e:
        logger.error(f"Error calculating BERTScore: {e}")
        traceback.print_exc()
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def compute(results, predictions, references, metrics_dict):
    try:
        for name, metric in metrics_dict.items():
            print(f"Computing: {name}")
            if name == "bertscore":
                results[name] = calculate_bertscore(predictions, references)
            else:
                results[name] = metric.compute(
                    predictions=predictions,
                    references=references
                )

    except Exception as e:
        logger.error(f"Error during computing metrics: {e}")
        traceback.print_exc()
    finally:
        gc.collect()
    return results if results is not None else {}


def log_metrics(pred, ref, metrics_dict, state, append=True, save_path=None):

    predictions, references = [], []
    predictions.extend(pred)
    references.extend(ref)
    results = {}

    try:
        results = compute(results, predictions, references, metrics_dict)
        logger.info(f"Metrics: {results}")

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
        logger.error(f"Error during evaluation: {e}")
        traceback.print_exc()
        
    finally:
        del predictions, references, ref, pred, results
        if df is not None:
            del df
        gc.collect()


def input_tensors(batch, device):

    # language modeling tensors
    inputs = {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device)
    }

    # vision tensors
    if 'pixel_values' in batch and batch['pixel_values'] is not None:
        inputs['pixel_values'] = batch['pixel_values'].to(device)
    else:
        logger.warning("pixel_values doesn't exist")

    if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
        inputs['image_grid_thw'] = batch['image_grid_thw'].to(device)
    else:
        logger.warning("image_grid_thw doesn't exist")

    return inputs


@torch.inference_mode(mode=True)
def causal_generate(model, inputs, processor):
    generated_ids = model.generate(
        **inputs,
        max_new_tokens          =256,
        do_sample               =True,
        temperature             =1.2,      
        top_k                   =20,            
        top_p                   =0.95,          
        pad_token_id            =processor.tokenizer.pad_token_id,
        eos_token_id            =processor.tokenizer.eos_token_id,
        output_scores           =False,
        return_dict_in_generate =False,
    )

    input_lengths = inputs['attention_mask'].sum(dim=1)
    generated_ids_trimmed = []
    
    for input_len, out_ids in zip(input_lengths, generated_ids):
        if len(out_ids) > input_len:
            generated_ids_trimmed.append(out_ids[input_len:])
        else:
            generated_ids_trimmed.append(torch.tensor([], device=out_ids.device, dtype=out_ids.dtype))

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens             =True,
        clean_up_tokenization_spaces    =True   
    )
    torch.cuda.empty_cache()
    del generated_ids, generated_ids_trimmed
    return output_text


def clean_text(output_text):
    cleaned_outputs = []
    for text in output_text:
        cleaned = text.strip()
        cleaned = cleaned.replace('<|endoftext|>', '').replace('<|im_end|>', '').replace('Describe the following image.', '').replace('assistant', '').strip()
        cleaned_outputs.append(cleaned)
    return cleaned_outputs


class GenerationCallback(TrainerCallback):
    def __init__(self, processor, metrics_path, metrics):
        self.processor = processor
        self.metrics_path = metrics_path
        self.metrics = metrics

    def on_evaluate(self, args, state, control, **kwargs):
        start_time = time.time()  
        pprint.pprint(kwargs)
        logger.info("Starting evaluation...")
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = 'left'
        
        try:
            model = kwargs['model']
            device = model.device
            eval_dataloader = kwargs['eval_dataloader']
            model.eval()

            all_predictions = []
            all_references = []
            
            iteration_start = time.time()   
            for i, batch in enumerate(eval_dataloader):
                logger.info(f"Evaluation: {i}")
                if i >= 100:  
                    logger.info(f"Iteration ended at {i}")
                    break
                 
                inputs = input_tensors(batch, device)
                 
                output_text = causal_generate(model, inputs, self.processor)
                
                cleaned_outputs = clean_text(output_text)
                
                all_predictions.extend(cleaned_outputs)
                all_references.extend(batch['suffixes'])
                
                torch.cuda.empty_cache()

            iteration_time = time.time() - iteration_start
            logger.info(f"Evaluation took {iteration_time} seconds")
 
            if all_predictions and all_references:
                logger.info("Computing NLP metrics...")
                metrics_start = time.time()
                log_metrics(
                    pred=all_predictions,
                    ref=all_references,
                    metrics_dict=self.metrics,
                    state=state,
                    save_path=self.metrics_path
                )
                metrics_time = time.time() - metrics_start
                logger.info(f"Metrics computation took {metrics_time:.2f} seconds")
            else:
                logger.warning("No predictions generated for evaluation")

        except Exception as e:
            logger.error(f"Error in evaluation (non-fatal): {e}")
            traceback.print_exc()

        finally:
            total_time = time.time() - start_time
            logger.info(f"Total evaluation completed in {total_time:.2f} seconds")
            del cleaned_outputs, all_predictions, all_references
            self.processor.tokenizer.padding_side = original_padding_side
            torch.cuda.empty_cache()

class CustomTrainer(GRPOTrainer):
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
        self.processor          =processor

        self.num_workers_tr     =num_workers_tr
        self.num_workers_ev     =num_workers_ev

        self.is_pin_memory_tr   =pin_memory_tr
        self.is_pin_memory_ev   =pin_memory_ev

        self.is_persistent_tr   =persistent_tr
        self.is_persistent_ev   =persistent_ev

    def get_train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size          =self.args.per_device_train_batch_size,
            shuffle             =True,
            collate_fn          =lambda b: _tokenization_tr(b, self.processor),
            num_workers         =self.num_workers_tr,
            pin_memory          =self.is_pin_memory_tr,
            persistent_workers  =self.is_persistent_tr
        )
    
    def get_eval_dataloader(self, eval_dataset=None):

        return DataLoader(
            eval_dataset or self.eval_dataset,
            batch_size          =self.args.per_device_eval_batch_size,
            shuffle             =False,
            collate_fn          =lambda b: _tokenization_ev(b, self.processor),
            num_workers         =self.num_workers_ev,
            pin_memory          =self.is_pin_memory_tr,
            persistent_workers  =self.is_persistent_ev
        )