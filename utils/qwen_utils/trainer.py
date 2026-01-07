from transformers import (
    TrainingArguments
    # DataCollatorForSeq2Seq
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
from preprocess import dataset
import warnings, torch, gc, evaluate
from loguru import logger


lora_targets=[
    [
    # qwen3 language layer
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # qwen3 vision layer
    "qkv", "proj", "linear_fc1", "linear_fc2"
    ],
    [
    # qwen2.5 v + l
    "q_proj", "k_proj", "o_proj", 
    "v_proj", "down_proj", "up_proj"
    ]
]

_revision="revision_2-DERM1M"
_model_path=f"./models/Qwen3-VL-4B-Instruct"
_checkpoint_resume="./models/lora_qwen_vl_revision_2-DERM1M/checkpoint-100"
_model_applybnb=True
_model_dmap="auto"
_model_dtype=torch.bfloat16
_model_isqwen3=True
_model_patch_liger=True

_train_ft_output=f"./models/lora_qwen_vl_{_revision}/"
_train_metrics_log=f"./docs/metrics_{_revision}.csv"

_dataset_ev_tr_images_max_pixel= (512 * 28 * 28) if _model_isqwen3 is False else (512 * 32 * 32) 
_dataset_ev_tr_images_min_pixel= (256 * 28 * 28) if _model_isqwen3 is False else (256 * 32 * 32)

_dataset_tr_workers_count=2             
_dataset_ev_workers_count=2             
_dataset_tr_pin_memory=False 
_dataset_ev_pin_memory=False    
_dataset_tr_persistent=True if _dataset_tr_pin_memory else False
_dataset_ev_persistent=True if _dataset_tr_pin_memory else False

_train_lora_modules = lora_targets[0] if _model_isqwen3 else lora_targets[1]
_train_lora_alpha=16
_train_lora_ranks=_train_lora_alpha*2
_train_lora_dropout=0.05
_train_lora_dora=True
_train_lora_bias="none"

_train_epochs=15
_train_batch_size=1
_eval_batch_size=1
_train_grad_accum=_train_batch_size*2

_train_learning_rate=5e-5
_train_optimizer="lion_8bit"  
_train_scheduler="cosine"      
_train_max_grad_norm=15.5
_train_weight_decay=0.001
_train_warmup_steps=50
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
_train_drop_unused_column=True
 
_eval_rouge = evaluate.load("rouge", experiment_id=f"{_revision}_rouge")
_eval_bleu = evaluate.load("bleu", experiment_id=f"{_revision}_bleu")
_eval_meteor = evaluate.load("meteor", experiment_id=f"{_revision}_meteor")

def Prepare(use_cuda=None, require_grad=None):
    warnings.filterwarnings(
        "ignore", 
        message="None of the inputs have requires_grad=True.*"
        )

    model, processor = LLM_LOAD_HF(
            path_to_model=_model_path,
            dtype=_model_dtype,
            dmap=_model_dmap,
            apply_quant=_model_applybnb,
            min_pixels=_dataset_ev_tr_images_min_pixel,
            max_pixels=_dataset_ev_tr_images_max_pixel,
            apply_liger_kernel=_model_patch_liger,
            qwen3=_model_isqwen3
        )

    train_dataset, eval_dataset = dataset()

    if use_cuda and require_grad:
        model.to('cuda')
        model.enable_input_require_grads()
        # print(torch.cuda.memory_summary())
    else:
        model.to('cpu')
        model.enable_input_require_grads()
        logger.info("Using CPU; Training will take significantly longer.")

    return model, processor, train_dataset, eval_dataset


def VL_Train(
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
                                    inference_mode=False,
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
        # logger.info(f"{peft_model}")

        training_args = TrainingArguments(
            output_dir=_train_ft_output,
            num_train_epochs=_train_epochs,
            per_device_train_batch_size=_train_batch_size,
            per_device_eval_batch_size=_eval_batch_size,
            gradient_accumulation_steps=_train_grad_accum,
            learning_rate=_train_learning_rate,
            optim=_train_optimizer,
            lr_scheduler_type=_train_scheduler,
            max_grad_norm=_train_max_grad_norm,
            weight_decay=_train_weight_decay,
            warmup_steps=_train_warmup_steps,
            seed=_train_seed,
            logging_dir=f"./utils/logs_{_revision}",

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
            remove_unused_columns=_train_drop_unused_column,

            use_liger_kernel=_model_patch_liger,
            liger_kernel_config={
                "rope": True,
                "swiglu": True,
                "fused_linear_cross_entropy": True,  
                "rms_norm": True
            }
        )
        trainer = CustomTrainer(
            model=peft_model,
            processor=processor,
            num_workers_tr=_dataset_tr_workers_count,
            num_workers_ev=_dataset_ev_workers_count,
            pin_memory_tr=_dataset_tr_pin_memory,
            pin_memory_ev=_dataset_ev_pin_memory,
            persistent_tr=_dataset_tr_persistent,
            persistent_ev=_dataset_ev_persistent,
            # (in a case where our custom dataloader returns raw tensor, we'll use Seq2Seq collator)
            # data_collator=DataCollatorForSeq2Seq(
            #     tokenizer=processor.tokenizer, 
            #     padding=True
            # ),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[
                    GenerationCallback(
                        processor    = processor, 
                        metrics_path =_train_metrics_log,
                        metrics      = {
                            "rouge": _eval_rouge,
                            "bleu": _eval_bleu,
                            "meteor": _eval_meteor,
                            "bertscore": "bertscore"
                        }
                    )
                ]
        )

        peft_model.print_trainable_parameters()
        trainer.train(resume_from_checkpoint=_checkpoint_resume)
        trainer.save_model(training_args.output_dir)

    finally:
        if clear_cache_after:
            gc.collect()
            
if __name__ == "__main__":

    VL_Train(True, True, True)