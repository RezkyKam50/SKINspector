export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# export TENSORBOARD_LOGGING_DIR="./utils/logs/"
python utils/qwen_utils/trainer.py

#accelerate launch --mixed_precision=fp8 utils/qwen_utils/trainer.py