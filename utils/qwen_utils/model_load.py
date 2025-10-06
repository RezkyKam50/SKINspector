from ultralytics import YOLO
import os, torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from quant_hf import _quantizecfg
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl

def LLM_LOAD_HF(
    path_to_model, 
    apply_liger_kernel=None,
    min_pixels=None, 
    max_pixels=None, 
    apply_dynamic_resolution=None
    ):
    model_path = path_to_model

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
            trust_remote_code=True
            )
    else:
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
            )

    return model, processor