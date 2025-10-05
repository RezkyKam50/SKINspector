from transformers import (
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model
)
from custom import (
    CustomTrainer,
    GenerationCallback
)
from model_load import LLM_LOAD_HF
from preprocess import _train_splits
import warnings, torch, numpy as np, evaluate, pandas as pd, gc


_revision="revision_1"

_model_path=f"./models/Qwen2.5-VL-3B-Instruct"
_model_patch_liger=True
_model_dynamic_res=True


_train_ft_output=f"./models/lora_qwen_vl_{_revision}/"
_train_log_output=f"./utils/logs_{_revision}/"
_train_metrics_log=f"./docs/metrics_{_revision}.csv"


# this should be the folder containing both image and its annotation.
_dataset_path=f"./datasets"
# images should be under 'images' folder.
_dataset_file_images=str('images')
# annotation should be under 'annotations' folder.
_dataset_images_annotations=str('SkinCAP/skincap_v240623.csv')
# features exclusion
_dataset_features_exclusion=str("Do not consider this image")
# id of the features exclusion
_dataset_exclusion_id = [1, True]


_dataset_features_image_features=str('images')
_dataset_features_generation_caption=str('caption_zh_polish_en')
_dataset_features_match_to_id=str('id')

_dataset_ev_tr_images_max_pixel= (1024 * (28 * 28))
_dataset_ev_tr_images_min_pixel= (512 * (28 * 28))
_dataset_ev_tr_split=0.2
_dataset_ev_tr_seed=42

_dataset_tr_workers_count=4             
_dataset_ev_workers_count=4             
_dataset_tr_pin_memory=False        
_dataset_ev_pin_memory=False    
_dataset_tr_persistent=False        
_dataset_ev_persistent=False    

_train_lora_modules=[
    "q_proj", "k_proj", "o_proj", 
    "v_proj", "down_proj", "up_proj"
    ]
_train_lora_alpha=2
_train_lora_ranks=4
_train_lora_dropout=0.4
_train_lora_dora=True
_train_lora_bias="none"

_train_epochs=3
_train_batch_size=4
_eval_batch_size=4
_train_grad_accum=2
_train_learning_rate=1e-4
_train_optimizer="lion_8bit"  
_train_scheduler="cosine"      
_train_max_grad_norm=5.5
_train_weight_decay=0.07
_train_warmup_ratio=0.03
_train_seed=42

_train_save_strategy="steps"  
_train_save_steps=100
_train_save_total_limit=None
_train_logging_steps=10
_train_eval_steps=100
_train_eval_strategy="steps"    
_train_load_best_model=False
_train_metric_for_best=None
_train_greater_is_better=False

_train_report_to="tensorboard"  
_train_dataloader_drop_last=False
_train_logging_first_step=True
_train_eval_on_start=False

_train_fp16=False    
_train_bf16=True
_train_gradient_checkpointing=True
_train_tf32=False      
_train_ddp_find_unused_params=False
_train_group_by_length=False
_train_drop_unused_column=False

def Prepare(use_cuda=None, require_grad=None):
    warnings.filterwarnings(
        "ignore", 
        message="None of the inputs have requires_grad=True.*"
        )

    model, processor = LLM_LOAD_HF(
        path_to_model=_model_path,
        min_pixels=_dataset_ev_tr_images_min_pixel,
        max_pixels=_dataset_ev_tr_images_max_pixel,
        apply_liger_kernel=_model_patch_liger,
        apply_dynamic_resolution=_model_dynamic_res
        )

    train_dataset, eval_dataset = _train_splits(
        path_to_dataset_parent=_dataset_path, 
        images_dataset_file=_dataset_file_images,
        annotation_dataset_file=_dataset_images_annotations,
        image=_dataset_features_image_features,
        caption=_dataset_features_generation_caption,
        caption_id=_dataset_features_match_to_id,
        exclusion=_dataset_features_exclusion,
        exclusion_id=_dataset_exclusion_id,
        train_split=_dataset_ev_tr_split,
        split_seed=_dataset_ev_tr_seed
    )

    if use_cuda and require_grad:
        model.to('cuda')
        model.enable_input_require_grads()
        print(torch.cuda.memory_summary())
    else:
        model.to('cpu')
        model.enable_input_require_grads()
        print("Using CPU; Training will take significantly longer.")

    return model, processor, train_dataset, eval_dataset


def Qwen2_5_VL_Train(
    enable_cuda=None, 
    enable_grad=None, 
    clear_cache_after=None
    ):
    
    try:
        model, processor, train_dataset, eval_dataset = Prepare(
            use_cuda=enable_cuda,
            require_grad=enable_grad
        )
        print("Train dataset size:", len(train_dataset))
        print("Eval dataset size:", len(eval_dataset))
        peft_config = LoraConfig(
                                    lora_alpha=_train_lora_alpha,
                                    r=_train_lora_ranks,
                                    lora_dropout=_train_lora_dropout,
                                    use_dora=_train_lora_dora,
                                    bias=_train_lora_bias,
                                    target_modules=_train_lora_modules,
                                    task_type="CAUSAL_LM"
                                )

        peft_model = get_peft_model(
            model, peft_config
        )

        training_args = TrainingArguments(
            output_dir=_train_ft_output,
            logging_dir=_train_log_output,
            num_train_epochs=_train_epochs,
            per_device_train_batch_size=_train_batch_size,
            per_device_eval_batch_size=_eval_batch_size,
            gradient_accumulation_steps=_train_grad_accum,
            learning_rate=_train_learning_rate,
            optim=_train_optimizer,
            lr_scheduler_type=_train_scheduler,
            max_grad_norm=_train_max_grad_norm,
            weight_decay=_train_weight_decay,
            warmup_ratio=_train_warmup_ratio,
            seed=_train_seed,

            save_strategy=_train_save_strategy,
            save_steps=_train_save_steps,
            save_total_limit=_train_save_total_limit,
            logging_steps=_train_logging_steps,
            eval_steps=_train_eval_steps,
            eval_strategy=_train_eval_strategy,
            load_best_model_at_end=_train_load_best_model,
            metric_for_best_model=_train_metric_for_best,
            greater_is_better=_train_greater_is_better,

            report_to=_train_report_to,
            dataloader_drop_last=_train_dataloader_drop_last,
            logging_first_step=_train_logging_first_step,
            eval_on_start=_train_eval_on_start,

            fp16=_train_fp16,
            bf16=_train_bf16,
            gradient_checkpointing=_train_gradient_checkpointing,
            tf32=_train_tf32,
            ddp_find_unused_parameters=_train_ddp_find_unused_params,
            group_by_length=_train_group_by_length,

            remove_unused_columns=_train_drop_unused_column
        )
        trainer = CustomTrainer(
            processor=processor,
            num_workers_tr=_dataset_tr_workers_count,
            num_workers_ev=_dataset_ev_workers_count,
            pin_memory_tr=_dataset_tr_pin_memory,
            pin_memory_ev=_dataset_ev_pin_memory,
            persistent_tr=_dataset_tr_persistent,
            persistent_ev=_dataset_ev_persistent,
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[
                    GenerationCallback(
                        processor, 
                        _train_metrics_log
                        )
                ]
        )

        peft_model.print_trainable_parameters()
        trainer.train()
        trainer.save_model(training_args.output_dir)

    finally:
        if clear_cache_after:
            gc.collect()
            
if __name__ == "__main__":

    Qwen2_5_VL_Train(True, True, True)
