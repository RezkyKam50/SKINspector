#!/bin/bash
# run this script from parent directory
# convert from HF to GGUF
set -e

model_name="JSL-MedLlama-3-8B-v2.0"

python ./llama.cpp/convert_hf_to_gguf.py ./models/${model_name} --outtype f16 
python ./llama.cpp/convert_hf_to_gguf.py ./models/${model_name} --outtype f16 --mmproj
