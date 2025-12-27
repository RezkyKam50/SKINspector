from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

revision = "revision_5"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    f"./models/Qwen2.5-VL-3B-Instruct"
)
processor = AutoProcessor.from_pretrained(
    f"./models/Qwen2.5-VL-3B-Instruct"
)

peft_model = PeftModel.from_pretrained(model, f"./models/lora_qwen_vl_{revision}/checkpoint-300")

merged_model = peft_model.merge_and_unload()
 
merged_model.save_pretrained(f"./models/Qwen2.5-VL-3B-SkinCAP-DoRA")
processor.save_pretrained(f"./models/Qwen2.5-VL-3B-SkinCAP-DoRA")