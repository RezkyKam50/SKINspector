from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    f"./models/Qwen2.5-VL-3B-Instruct"
)
processor = AutoProcessor.from_pretrained(
    f"./models/Qwen2.5-VL-3B-Instruct"
)

peft_model = PeftModel.from_pretrained(model, f"./models/lora_qwen_vl/checkpoint-2300")

merged_model = peft_model.merge_and_unload()
 
merged_model.save_pretrained(f"./models/qwen2.5-vl-3b-merged")
processor.save_pretrained(f"./models/qwen2.5-vl-3b-merged")