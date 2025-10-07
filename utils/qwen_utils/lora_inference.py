from model_load import LLM_LOAD_HF
from qwen_vl_utils import process_vision_info
from peft import PeftModel

model, processor = LLM_LOAD_HF(
    f"./models/Qwen2.5-VL-3B-Instruct",
    True,
    (512 * (28 * 28)),
    (1024 * (28 * 28)),
    True
)

model = PeftModel.from_pretrained(model, f"./models/lora_qwen_vl_revision_1/checkpoint-300")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./examples/testing/basal_1.jpg"},
            {"type": "text", "text": "Describe and examine to the smallest detail of the skin condition on this image, list also the possible scenarios of the medical condition and its medical term."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
