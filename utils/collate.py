from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor
import torch

def _tokenization_tr(batch, processor):

    try:
        conversations = [sample["conversations"] for sample in batch]
        texts = [
            processor.apply_chat_template(conversation, tokenize=False)
            for conversation in conversations
        ]
        image_inputs = [
            process_vision_info(conversation)[0]
            for conversation in conversations
        ]
        
        model_inputs = processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True
        )
        
        labels = model_inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        if isinstance(processor, Qwen2_5_VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        
        tensors = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "pixel_values": model_inputs["pixel_values"],
            "image_grid_thw": model_inputs["image_grid_thw"],
            "labels": labels
        }

        return tensors

    finally:
        del conversations, texts, image_inputs, model_inputs
        torch.cuda.empty_cache()

"""
'input_ids': tensor([151644,   8948,    198,   2610,    525,    264,  30441,  11434,   
4903,  27076,    304,  65644,   9124,    821,    504,   9487,   5335,    624,   7771,   
3383,    374,    311,  23643,    279,   3897,   9487,   2168,    323,   5889,    311,  
19556,    448,  63594,  11253,     11,   5990,    264,   3175,   3409,

attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

'labels': tensor([151644,   8948,    198,   2610,    525,    264,  30441,  11434,   4903, 
 27076,    304,  65644,   9124,    821,    504,   9487,   5335,    624,   7771,   3383,    
 374,    311,  23643,    279,   3897,   9487,   2168,    323,   5889,    311,  19556,    
 448,  63594,  11253,     11,   5990,    264,   3175,   3409,
11,   1372,     11,    476,   2805,  17133,    624,    785,  26131,   2924,    
264,   8045,    315,   4494,    320,     68,   1302,   2572,   1555,  26131,     
11,   3619,  26131,      8,    323,   6644,   7987,     11,   9201,     11,    
323,   1467,    624,  13819,    389,  23988,  13382,     11,  98632,

'pixel_values': tensor([[ 0.3829,  0.4997,  0.3245,  ..., -0.2289, -0.1009, -0.1151],
        [ 1.0544,  1.0398,  1.1858,  ...,  0.2262,  0.2688,  0.3684],
        [ 0.4121,  0.6165,  0.6457,  ...,  0.0413,  0.1835,  0.2404],
        ...,
        [-1.6171, -1.6025, -1.6025,  ..., -1.3096, -1.2954, -1.2669],
        [-1.1645, -1.2959, -1.4857,  ..., -1.1532, -1.1816, -1.2243],
        [-1.5587, -1.5733, -1.5879,  ..., -1.2385, -1.1816, -1.2243]])}]


The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. 
The model config and generation config were aligned accordingly, being updated with the tokenizer's 
values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.

"""

def _tokenization_ev(batch, processor):

    try:
        conversations = [sample["conversations"][:2] for sample in batch]   
        suffixes = [sample["text"] for sample in batch]
        
        texts = [
            processor.apply_chat_template(conversation, tokenize=False)
            for conversation in conversations
        ]
        image_inputs = [
            process_vision_info(conversation)[0]
            for conversation in conversations
        ]
        
        model_inputs = processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True
        )
        
        tensors = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "pixel_values": model_inputs["pixel_values"],
            "image_grid_thw": model_inputs["image_grid_thw"],
            "suffixes": suffixes
        }

        return tensors
        
    finally:
        del conversations, texts, image_inputs, model_inputs
        torch.cuda.empty_cache()