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

















