#!/bin/bash
# run this script from parent directory
# convert from HF to GGUF
set -e

model_name="Qwen2.5-VL-3B-SkinCAP-DoRA"

python ./3rdparty/llama.cpp/convert_hf_to_gguf.py ./models/${model_name} --outtype f16 
python ./3rdparty/llama.cpp/convert_hf_to_gguf.py ./models/${model_name} --outtype f16 --mmproj
