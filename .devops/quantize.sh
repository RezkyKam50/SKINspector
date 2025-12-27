# you must compile llama.cpp from source first to use the quantization tool
# refer to SKINspector/scripts/build_llama.sh (Make sure you have all the essential build tools on your Linux machine)

model_name_gguf="Qwen2.5-VL-3B-SkinCAP-DoRA-F16"
model_dir="./models/Qwen2.5-VL-3B-SkinCAP-DoRA"

./3rdparty/llama.cpp/build/bin/llama-quantize ${model_dir}/${model_name_gguf}.gguf ${model_dir}/${model_name_gguf}-Q4_K_M.gguf Q4_K_M 8