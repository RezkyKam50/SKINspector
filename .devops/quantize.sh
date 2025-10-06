# you must compile llama.cpp from source first to use the quantization tool
# refer to SKINspector/scripts/build_llama.sh (Make sure you have all the essential build tools on your Linux machine)

./llama.cpp/build/bin/llama-quantize ./models/llama-medical.gguf ./models/llama-medical-Q4_K_M.gguf Q4_K_M 8