from PIL import Image
import os


def _skincap():
    path_to_images = f"./datasets/SkinCAP/skincap/"
    save_path = f"./datasets/images_skincap/"
    os.makedirs(save_path, exist_ok=True)
    
    for filename in os.listdir(path_to_images):
        if filename.endswith(".png"):
            idx = os.path.splitext(filename)[0] 
            input_file = os.path.join(path_to_images, filename)
            output_file = os.path.join(save_path, f"img_{idx}.jpg")

            with Image.open(input_file) as img:
                rgb_img = img.convert("RGB")  
                rgb_img.save(output_file, "JPEG")

            print(f"Converted {input_file} -> {output_file}")

if __name__ == "__main__":
    _skincap()
