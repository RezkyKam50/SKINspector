from .. utils.qwen_utils.model_load import LLM_LOAD_HF
from .. utils.qwen_utils.gradcamviz import visualize, generate_response
import matplotlib.pyplot as plt
from PIL import Image
from loguru import logger
import time

if __name__ == "__main__":
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
 
 