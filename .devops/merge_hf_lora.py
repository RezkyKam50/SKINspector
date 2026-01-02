import argparse
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with Qwen2.5-VL model and save the merged model"
    )
     
    parser.add_argument(
        "--base-model",
        type=str,
        default="./models/Qwen2.5-VL-3B-Instruct",
        help="Path to the base model directory"
    )
    
    parser.add_argument(
        "--lora-revision",
        type=str,
        default="revision_5",
        help="Revision name for LoRA model"
    )
    
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=300,
        help="Checkpoint step to load from LoRA model"
    )
    
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA model directory (if not using default structure)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/Qwen2.5-VL-3B-SkinCAP-DoRA",
        help="Directory to save the merged model"
    )
     
    args = parser.parse_args()
     
    if args.lora_path:
        lora_path = args.lora_path
    else:
        lora_path = f"./models/lora_qwen_vl_{args.lora_revision}/checkpoint-{args.checkpoint_step}"
     
    print(f"Loading base model from: {args.base_model}")
    model = AutoModelForImageTextToText.from_pretrained(args.base_model)
    processor = AutoProcessor.from_pretrained(args.base_model)
     
    print(f"Loading LoRA adapter from: {lora_path}")
    peft_model = PeftModel.from_pretrained(model, lora_path)
     
    print("Merging LoRA adapters with base model...")
    merged_model = peft_model.merge_and_unload()
    
    print(f"Saving merged model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()