export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="0"
# export TENSORBOARD_LOGGING_DIR="./utils/logs/"

source .venv/bin/activate
python3 utils/qwen_utils/trainer.py
