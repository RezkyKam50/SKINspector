from model_load import LLM_LOAD_HF
from qwen_vl_utils import process_vision_info
from peft import PeftModel 

model, processor = LLM_LOAD_HF(
    f"./models/Qwen2.5-3B-SkinCAP-DoRA",
    True,
    (512 * (28 * 28)),
    (1024 * (28 * 28)),
    True
)

messages = [
    {
        "role": "system",
        "content": "You are a medical dermatology expert assistant. Analyze skin conditions from images with thorough clinical precision. Provide detailed observations, differential diagnoses with medical terminology, and maintain professional medical standards in all responses."
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./examples/testing/basal.jpg"},
            {"type": "text", "text": "Describe this image."},
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
