from datasets import Dataset, load_dataset, DatasetDict
from prompting import system_message, query_message
from path_resolver import PARENT
import pandas as pd


def _train_splits():

    path = f"{PARENT()}/dataset"
     
    ann = pd.read_csv(f"{path}/anno.csv")
     
    def fetch(image_id):
        return f"{path}/images/img_{image_id}.jpg"

    ann = ann[ann['id'].between(1, 3700)].copy()
    df = pd.DataFrame({
        'image': [fetch(img_id) for img_id in ann['id']],
        'caption_zh_polish_en': ann['caption_zh_polish_en']
    })

    def format_data(sample, index):
        _image = sample['image']   
        conversation = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": _image,   
                    },
                    {
                        "type": "text",
                        "text": query_message,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": sample['caption_zh_polish_en'],
            },
        ]
        
        return {
            "image": _image,  
            "conversations": conversation,
            "text": sample['caption_zh_polish_en'] # <- this is for suffix or golden reference 
        }

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    train_dataset = [format_data(sample, i) for i, sample in enumerate(dataset["train"])]
    eval_dataset = [format_data(sample, i) for i, sample in enumerate(dataset["test"])]


    return train_dataset, eval_dataset






# def _tokenization(messages, processor):
#     processed_samples = []
    
#     for message in messages:
#         text = processor.apply_chat_template(
#             message["conversations"], 
#             tokenize=False, 
#             add_generation_prompt=False,  # FALSE for training
#         )
        
#         # Process vision information - this should handle base64 images
#         image_inputs, video_inputs = process_vision_info(
#             message["conversations"], 
#             return_video_kwargs=False
#         )
        
#         # Tokenize THIS SINGLE SAMPLE
#         sample_inputs = processor(
#             text=text,
#             images=image_inputs[0] if image_inputs else None,  # First image if exists
#             padding=True,  # Will pad later during collation
#             return_tensors="pt",
#         )
        
#         # Convert to individual sample format
#         sample_dict = {
#             "input_ids": sample_inputs["input_ids"].squeeze(0),        # Remove batch dim
#             "attention_mask": sample_inputs["attention_mask"].squeeze(0),
#             "labels": sample_inputs["input_ids"].squeeze(0).clone(),   # Labels for LM
#         }
         
#         if "pixel_values" in sample_inputs:
#             sample_dict["pixel_values"] = sample_inputs["pixel_values"].squeeze(0)

#         if "image_grid_thw" in sample_inputs:
#             sample_dict["image_grid_thw"] = sample_inputs["image_grid_thw"]
        
#         processed_samples.append(sample_dict)
#         print(processed_samples)

#     return processed_samples
















