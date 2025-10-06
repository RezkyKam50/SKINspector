from datasets import Dataset
from prompting import system_message, query_message
import cudf #, pandas as pd

def format_data(sample, image, caption):
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
                    "image": sample[image],   
                },
                {
                    "type": "text",
                    "text": query_message,
                },
            ],
        },
        {
            "role": "assistant",
            "content": sample[caption],
        },
    ]
    
    return {
        "image": sample[image],  
        "conversations": conversation,
        "text": sample[caption] # <- this is for suffix or golden reference 
    }

def fetch(idx, parent, img_folder):
    return f"{parent}/{img_folder}/img_{idx}.jpg"

def _train_splits(
    path_to_dataset_parent=None, 
    images_dataset_file=None,
    annotation_dataset_file=None,
    image=None, 
    caption=None, 
    caption_id=None, 
    exclusion=None,
    exclusion_id=None,
    train_split=None, 
    split_seed=None,
    ):
    ann = cudf.read_csv(f"{path_to_dataset_parent}/{annotation_dataset_file}")

    if exclusion is not None:
        ann = ann[~ann[exclusion].isin(exclusion_id)].reset_index(drop=True)

    print(image, caption, caption_id)

    ann = ann[ann[caption_id].between(1, 4000)].copy().reset_index(drop=True)

    df = cudf.DataFrame({
        image: [fetch(
            idx, 
            path_to_dataset_parent, 
            images_dataset_file
            ) for idx in ann[caption_id].values_host],
        caption: ann[caption]
    })

    dataset = Dataset.from_pandas(df.to_pandas())
    dataset = dataset.train_test_split(test_size=train_split, seed=split_seed)
    
    train_data = [sample for sample in dataset["train"]]
    eval_data = [sample for sample in dataset["test"]]
    
    train_dataset = [format_data(
        sample, 
        image, 
        caption
        ) for sample in train_data]

    eval_dataset = [format_data(
        sample, 
        image, 
        caption
        ) for sample in eval_data]
    
    return train_dataset, eval_dataset


