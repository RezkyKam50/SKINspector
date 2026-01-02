#!/bin/bash
set -e

# Model configuration
BASE_MODEL_NAME="Qwen2.5-VL-3B-Instruct"   
MERGED_MODEL_NAME="Qwen2.5-VL-3B-SkinCAP-DoRA"

# LoRA configuration  
LORA_REVISION="revision_1-DERM1M"
LORA_CHECKPOINT=5000

# Paths
BASE_MODEL_PATH="./models/Qwen2.5-VL-3B-Instruct"  
LORA_MERGE_OUTPUT="./models/${MERGED_MODEL_NAME}"

# Conversion settings
CONV_PRECISION="f16"  
Q_TYPE="q4_k_m"       
 


echo "Base model: ${BASE_MODEL_PATH}"
echo "LoRA revision: ${LORA_REVISION}"
echo "Checkpoint step: ${LORA_CHECKPOINT}"
echo "Output directory: ${LORA_MERGE_OUTPUT}"

# Merge LoRA/DoRA with base model
python .devops/merge_hf_lora.py \
    --base-model "${BASE_MODEL_PATH}" \
    --lora-revision "${LORA_REVISION}" \
    --checkpoint-step ${LORA_CHECKPOINT} \
    --output-dir "${LORA_MERGE_OUTPUT}"
 
echo "Converting to ${CONV_PRECISION}..."
 
python ./3rdparty/llama.cpp/convert_hf_to_gguf.py \
    "${LORA_MERGE_OUTPUT}" \
    --outtype ${CONV_PRECISION} \
    --outfile "${LORA_MERGE_OUTPUT}/${MERGED_MODEL_NAME}-${CONV_PRECISION}.gguf"

# Conversion from HF to GGUF - MMPROJ (for vision)
python ./3rdparty/llama.cpp/convert_hf_to_gguf.py \
    "${LORA_MERGE_OUTPUT}" \
    --outtype ${CONV_PRECISION} \
    --mmproj \
    --outfile "${LORA_MERGE_OUTPUT}/${MERGED_MODEL_NAME}-${CONV_PRECISION}-mmproj.gguf"
 
echo "Quantizing to ${Q_TYPE}..."

# Quantize GGUF Model
./3rdparty/llama.cpp/build/bin/llama-quantize \
    "${LORA_MERGE_OUTPUT}/${MERGED_MODEL_NAME}-${CONV_PRECISION}.gguf" \
    "${LORA_MERGE_OUTPUT}/${MERGED_MODEL_NAME}-${Q_TYPE}.gguf" \
    ${Q_TYPE} \
    $(nproc)
 
echo "Models saved in: ${LORA_MERGE_OUTPUT}"
echo "Files created:"
ls -lh "${LORA_MERGE_OUTPUT}"/*.gguf