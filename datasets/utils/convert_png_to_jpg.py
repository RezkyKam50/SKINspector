from PIL import Image
import os

from path_resolver import PARENT

path_to_images = f"{PARENT(levels=1)}/SkinCAP/skincap/"
save_path = f"{PARENT(levels=1)}/images/"
os.makedirs(save_path, exist_ok=True)
 
for filename in os.listdir(path_to_images):
    if filename.endswith(".png"):
        idx = os.path.splitext(filename)[0] 
        input_file = os.path.join(path_to_images, filename)
        output_file = os.path.join(save_path, f"{idx}.jpg")

        with Image.open(input_file) as img:
            rgb_img = img.convert("RGB")  
            rgb_img.save(output_file, "JPEG")

        print(f"Converted {input_file} -> {output_file}")
