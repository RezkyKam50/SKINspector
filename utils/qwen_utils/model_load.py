import os, torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2_5_VLProcessor, 
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor
)
from quantize_bnb import _quantizecfg
from liger_kernel.transformers import (
    apply_liger_kernel_to_qwen2_5_vl, 
    apply_liger_kernel_to_qwen3_vl
)


def LLM_LOAD_HF(
    path_to_model, 
    apply_liger_kernel=None,
    min_pixels=None, 
    max_pixels=None, 
    apply_dynamic_resolution=None,
    qwen3=None
    ):
    model_path = path_to_model

    if qwen3:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, 
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            local_files_only=True,
            quantization_config=_quantizecfg()
        )

        if apply_liger_kernel:
            apply_liger_kernel_to_qwen3_vl(
                rope=True,
                fused_linear_cross_entropy=True,
                rms_norm=True,
                swiglu=True,
                model=model
            )
         
        processor = Qwen3VLProcessor.from_pretrained(
            model_path, 
            use_fast=False,
            trust_remote_code=True
            )

        return model, processor
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            local_files_only=True,
            quantization_config=_quantizecfg()
        )

        if apply_liger_kernel:
            apply_liger_kernel_to_qwen2_5_vl(
                rope=True,
                fused_linear_cross_entropy=True,
                rms_norm=True,
                swiglu=True,
                model=model
            )
        
        if apply_dynamic_resolution:
            print(f"Processor loaded with dynamic resolution scaling.\nMax Pixels = {max_pixels}\nMin Pixels = {min_pixels}")
            processor = Qwen2_5_VLProcessor.from_pretrained(
                model_path, 
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                use_fast=False,
                trust_remote_code=True
                )
        else:
            processor = Qwen2_5_VLProcessor.from_pretrained(
                model_path, 
                use_fast=False,
                trust_remote_code=True
                )

    return model, processor