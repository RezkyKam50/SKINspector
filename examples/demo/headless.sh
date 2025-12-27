model_name_gguf="Qwen2.5-3B-SkinCAP-DoRA-F16-Q4_K_M.gguf"
model_dir="./models/Qwen2.5-3B-SkinCAP-DoRA"

./3rdparty/llama.cpp/build/bin/llama-mtmd-cli -m ${model_dir}/${model_name_gguf} \
    --mmproj ${model_dir}/mmproj* \
    --image ./examples/testing/basal.jpg \
    -p "Describe this image, list also the possible scenarios of the medical condition and its medical term."