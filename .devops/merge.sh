python .devops/merge_hf_lora.py \
    --base-model "./models/my-custom-model" \
    --lora-revision "experiment_1" \
    --checkpoint-step 500 \
    --output-dir "./models/merged-model-final"