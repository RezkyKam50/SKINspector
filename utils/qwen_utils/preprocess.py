from datasets import Dataset
from prompting import system_message, query_message
import pandas as pd

def format_data(sample, image_col, caption_col):
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
                    "image": sample[image_col],
                },
                {
                    "type": "text",
                    "text": query_message,
                },
            ],
        },
        {
            "role": "assistant",
            "content": sample[caption_col],
        },
    ]
    return {
        "image": sample[image_col],
        "conversations": conversation,
        "text": sample[caption_col]
    }

def fetch(filename, img_folder):
    img_path = f"{img_folder}/{filename}"
    return img_path

path_to_dataset_parent = "./datasets/Derm1M"
train_annotation_dataset_file = "Derm1M_v2_pretrain.parquet"
eval_annotation_dataset_file = "Derm1M_v2_validation.parquet"
train_images_folder = f"{path_to_dataset_parent}"
eval_images_folder = f"{path_to_dataset_parent}"

def _train_anno():
    ann = pd.read_parquet(f"{path_to_dataset_parent}/{train_annotation_dataset_file}")
    df = pd.DataFrame({
        "image": [fetch(filename, train_images_folder) for filename in ann["filename"]],
        "caption": ann["caption"]
    })
    dataset = Dataset.from_pandas(df)
    
    train_data = [format_data(sample, "image", "caption") for sample in dataset]
    
    return train_data

def _val_anno():
    ann = pd.read_parquet(f"{path_to_dataset_parent}/{eval_annotation_dataset_file}")
    df = pd.DataFrame({
        "image": [fetch(filename, eval_images_folder) for filename in ann["filename"]],
        "caption": ann["caption"]
    })
    dataset = Dataset.from_pandas(df)
    val_data = [format_data(sample, "image", "caption") for sample in dataset]
    
    return val_data

def dataset():
    train_data = _train_anno()
    val_data = _val_anno()

    # train_data = Dataset.from_list(train_data)
    # val_data = Dataset.from_list(val_data)

    return train_data, val_data

if __name__ == "__main__":
    train_data, val_data = dataset()
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    print("Sample training data:", train_data[0])
    print("Sample validation data:", val_data[0])
