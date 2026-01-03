import torch
import numpy as np
from loguru import logger
import time

from PIL import Image
import matplotlib.pyplot as plt

from qwen_vl_utils import process_vision_info
from model_load import LLM_LOAD_HF

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

 
class VisionModelWrapper(torch.nn.Module):
    def __init__(self, model, inputs):
        super().__init__()
        self.model = model
        self.inputs = inputs
        self.grid_hw = None  # (H, W)

    def forward(self, x):
        outputs = self.model(**self.inputs)
        logits = outputs.logits[:, -1, :]
        return logits.float()
 
def reshape_transform_vit(tensor, grid_hw):
    while tensor.dim() > 4:
        tensor = tensor.squeeze(-1)
    logger.info(f"Found tensor shape: {tensor.shape} with dim: {tensor.dim()}")
 
    if tensor.dim() == 2:
        N, C = tensor.shape
        H, W = grid_hw
        spatial = H * W
        tensor = tensor[:spatial]
        logger.info("Case: [N, C]")
        return tensor.T.reshape(1, C, H, W)
 
    if tensor.dim() == 3:
        B, N, C = tensor.shape
        H, W = grid_hw
        spatial = H * W
        tensor = tensor[:, :spatial, :]
        logger.info("Case [B, N, C]")
        return tensor.permute(0, 2, 1).reshape(B, C, H, W)
 
    if tensor.dim() == 4:
        logger.info("Case [B, C, H, W]")
        return tensor

    raise RuntimeError(f"CAM reshape failed for tensor shape {tensor.shape}")
 
def visualize(model, processor, image_path, question, target_layer):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
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
    ).to("cuda")
    
    # qwen have built-in image_grid_thw
    grid_thw = inputs.image_grid_thw[0].tolist()
    _, H, W = grid_thw

    pil_image = Image.open(image_path).convert("RGB")
    image_np = np.array(pil_image).astype(np.float32) / 255.0

    wrapped_model = VisionModelWrapper(model, inputs)
    wrapped_model.grid_hw = (H, W)

    cam = GradCAM(
        model=wrapped_model,
        target_layers=[target_layer],
        reshape_transform=lambda t: reshape_transform_vit(t, wrapped_model.grid_hw),
    )

    dummy = torch.zeros(1, 3, 224, 224, device="cuda")
    cam_map = cam(input_tensor=dummy)[0]

    cam_map = np.maximum(cam_map, 0)
    cam_map = cam_map / (cam_map.max() + 1e-8)

    cam_resized = np.array(
        Image.fromarray((cam_map * 255).astype(np.uint8)).resize(
            (image_np.shape[1], image_np.shape[0]), Image.BILINEAR
        )
    ) / 255.0

    visualization = show_cam_on_image(
        image_np,
        cam_resized,
        use_rgb=True,
    )

    return visualization, cam_resized
  
def generate_response(model, processor, image_path, question):
    messages = [
        {
            "role": "system",
            "content": "You are a medical dermatology expert assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
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
    ).to("cuda")

    with torch.inference_mode():
        generated_ids = model.generate(
        **inputs,
        max_new_tokens          =256,
        do_sample               =True,
        temperature             =1.4,      
        top_k                   =50,            
        top_p                   =0.90,          
        pad_token_id            =processor.tokenizer.pad_token_id,
        eos_token_id            =processor.tokenizer.eos_token_id,
        output_scores           =False,
        return_dict_in_generate =False,
        )
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

    return output_text[0]

 
def main():
    model, processor = LLM_LOAD_HF(
        "./models/Qwen3-VL-2B-Instruct",
        apply_liger_kernel=True,
        qwen3=True
    )
    model = model.to("cuda").eval()
    model.model.visual = model.model.visual.float()

    image_path = "./examples/testing/basal_1.jpg"
    question = "Describe this skin lesion."

    target_layer = model.model.visual.blocks[-1].mlp.linear_fc2

    print("Generating EigenCAM...")
    visualization, cam_heatmap = visualize(
        model,
        processor,
        image_path,
        question,
        target_layer,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(Image.open(image_path))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam_heatmap, cmap="jet")
    axes[1].set_title("EigenCAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(visualization)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nModel response:")

    start = time.time()
    print(generate_response(model, processor, image_path, question))
    end = time.time() - start
    logger.info(f"Done in {end} seconds")
 

if __name__ == "__main__":
    main()
