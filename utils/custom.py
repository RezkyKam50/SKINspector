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


def log_metrics(pred, ref, metrics_dict, state, append=True, save_path=None):
    predictions, references = [], []
    predictions.extend(pred)
    references.extend(ref['suffixes'])
    results = {}
    for name, metric in metrics_dict.items():
        if name == "bertscore":
            results[name] = metric.compute(
                predictions=predictions,
                references=references,
                lang="en",
                model_type="distilbert-base-uncased" 
            )
            results[name] = {
                "precision": sum(results[name]["precision"]) / len(results[name]["precision"]),
                "recall": sum(results[name]["recall"]) / len(results[name]["recall"]),
                "f1": sum(results[name]["f1"]) / len(results[name]["f1"]),
            }
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
        df.to_csv(save_path, mode='a', index=False,
                  header=not os.path.exists(save_path))
    else:
        df.to_csv(save_path, index=False)

    torch.cuda.empty_cache()



class GenerationCallback(TrainerCallback):
    def __init__(self, processor, _metrics_path):
        self.processor = processor
        self._metrics_path = _metrics_path
        self.metrics = {
            "rouge": evaluate.load("rouge"),
            "bleu": evaluate.load("bleu"),
            "meteor": evaluate.load("meteor"),
            "bertscore": evaluate.load("bertscore")
        }

    @torch.no_grad()
    @torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True)
    def on_evaluate(self, args, state, control, **kwargs):
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = 'left'
        
        try:
            model = kwargs['model']
            eval_dataloader = kwargs['eval_dataloader']
            model.eval()

            for i, batch in enumerate(eval_dataloader):
                if i >= 10:   
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
                    output_scores=False,
                    return_dict_in_generate=False,
                )
                input_lengths = batch['attention_mask'].sum(dim=1)

                generated_ids_trimmed = [
                    out_ids[input_len:] 
                    for input_len, out_ids in zip(input_lengths, generated_ids)
                ]
                del generated_ids, inputs
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                log_metrics(
                    pred=output_text,
                    ref=batch,
                    metrics_dict=self.metrics,
                    state=state,
                    save_path=self._metrics_path
                )
                torch.cuda.empty_cache()

        finally:
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
            shuffle=False,
            collate_fn=lambda b: _tokenization_ev(b, self.processor),
            num_workers=self.num_workers_ev,
            pin_memory=self.is_persistent_ev,
            persistent_workers=self.is_persistent_ev
        )