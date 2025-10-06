import torch
import torch.nn as nn

from awq import AutoAWQForCausalLM
from awq.utils.qwen_vl_utils import process_vision_info
from awq.quantize.quantizer import AwqQuantizer, clear_memory, get_best_device

from preprocess import _train_splits

# Specify paths and hyperparameters for quantization
model_path = "./models/qwen2.5-vl-3b-merged"
quant_path = "./models/Qwen2.5-VL-3b-Dermatology-AWQ"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    device_map='cpu',
    torch_dtype=torch.bfloat16
)
print(model.model)


class Qwen2VLAwqQuantizer(AwqQuantizer):
    def get_model_layers(self, model):
        """Override to get the correct layers for Qwen2.5-VL architecture"""
        return model.model.language_model.layers
    
    def move_embed(self, model, device: torch.device):
        """Move embedding layer to device"""
        model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(device)
    
    def init_quant(self, n_samples=None, max_seq_len=None):
        modules = self.get_model_layers(self.model)
        samples = self.calib_data

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.move_embed(self.model, best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        def move_to_device(obj: torch.Tensor | nn.Module, device: torch.device):
            def get_device(obj: torch.Tensor | nn.Module):
                if isinstance(obj, torch.Tensor):
                    return obj.device
                return next(obj.parameters()).device

            if get_device(obj) != device:
                obj = obj.to(device)
            return obj

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        for k, v in samples.items():
            if isinstance(v, (torch.Tensor, nn.Module)):
                samples[k] = move_to_device(v, best_device)
        try:
            self.model(**samples)
        except ValueError:  # work with early exit
            pass
        finally:
            for k, v in samples.items():
                if isinstance(v, (torch.Tensor, nn.Module)):
                    samples[k] = move_to_device(v, "cpu")
        modules[0] = modules[0].module  # restore

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.move_embed(self.model, "cpu")

        clear_memory()

        return modules, layer_kwargs, inps

_dataset_path=f"./datasets"
_dataset_file_images=str('images')
_dataset_images_annotations=str('SkinCAP/skincap_v240623.csv')
_dataset_features_exclusion=str("Do not consider this image")
_dataset_exclusion_id = [0, False]
_dataset_features_image_features=str('images')
_dataset_features_generation_caption=str('caption_zh_polish_en')
_dataset_features_match_to_id=str('id')
_dataset_ev_tr_split=0.2
_dataset_ev_tr_seed=42

train_dataset, eval_dataset = _train_splits(
    path_to_dataset_parent=_dataset_path, 
    images_dataset_file=_dataset_file_images,
    annotation_dataset_file=_dataset_images_annotations,
    image=_dataset_features_image_features,
    caption=_dataset_features_generation_caption,
    caption_id=_dataset_features_match_to_id,
    exclusion=_dataset_features_exclusion,
    exclusion_id=_dataset_exclusion_id,
    train_split=_dataset_ev_tr_split,
    split_seed=_dataset_ev_tr_seed
)
print("Train dataset size:", len(train_dataset))
print("Eval dataset size:", len(eval_dataset))

# Extract the conversation lists for AWQ
chat_dataset = [item["conversations"] for item in train_dataset + eval_dataset]

text = model.processor.apply_chat_template(chat_dataset, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(chat_dataset)
inputs = model.processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

model.quantize(calib_data=inputs, quant_config=quant_config, quantizer_cls=Qwen2VLAwqQuantizer)

# Save the model
model.model.config.use_cache = model.model.generation_config.use_cache = True
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")