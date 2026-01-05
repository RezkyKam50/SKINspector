from transformers import (
    Qwen2TokenizerFast,
    Qwen2VLImageProcessor,
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2_5_VLProcessor, 
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor
)
from liger_kernel.transformers import (
    apply_liger_kernel_to_qwen2_5_vl, 
    apply_liger_kernel_to_qwen3_vl,
    apply_liger_kernel_to_qwen3_vl_moe
)
from transformers import BitsAndBytesConfig
import torch
from PIL import Image
from loguru import logger

def quantizecfg():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    return bnb_config

def LLM_LOAD_HF(
    path_to_model, 
    dtype=None,
    dmap=None,
    apply_quant=None,
    apply_liger_kernel=None,
    min_pixels=None, 
    max_pixels=None, 
    qwen3=None
    ):
    model_path = path_to_model

    if qwen3:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, 
            device_map=dmap,
            dtype=dtype,
            attn_implementation="flash_attention_3",
            trust_remote_code=True,
            local_files_only=True,
            quantization_config=quantizecfg() if apply_quant else None
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

        if min_pixels and max_pixels:
            # qwen3vl inherits qwen2vl processor
            logger.info(f"Processor loaded with dynamic resolution scaling.\nMax Pixels = {max_pixels}\nMin Pixels = {min_pixels}")
            processor.image_processor.min_pixels = min_pixels
            processor.image_processor.max_pixels = max_pixels
            processor.image_processor.resample = Image.Resampling.LANCZOS

        return model, processor
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            device_map=dmap,
            dtype=dtype,
            attn_implementation="flash_attention_3",
            trust_remote_code=True,
            local_files_only=True,
            quantization_config=quantizecfg() if apply_quant else None
        )

        if apply_liger_kernel:
            apply_liger_kernel_to_qwen2_5_vl(
                rope=True,
                fused_linear_cross_entropy=True,
                rms_norm=True,
                swiglu=True,
                model=model
            )
        
        if min_pixels and max_pixels:
            logger.info(f"Processor loaded with dynamic resolution scaling.\nMax Pixels = {max_pixels}\nMin Pixels = {min_pixels}")
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