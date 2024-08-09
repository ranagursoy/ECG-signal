import os
from PIL import Image

def convert_images_to_grayscale(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.jpeg')):
            img_path = os.path.join(input_folder, file_name)
            img = Image.open(img_path)
            gray_img = img.convert("L")
            gray_img_path = os.path.join(output_folder, file_name)
            gray_img.save(gray_img_path)
            print(f"Converted {file_name} to grayscale and saved to {output_folder}")

def convert_images_to_binary(input_folder, output_folder, threshold=128):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.jpeg')):
            img_path = os.path.join(input_folder, file_name)
            img = Image.open(img_path)
            gray_img = img.convert("L")
            binary_img = gray_img.point(lambda x: 255 if x > threshold else 0, '1')
            binary_img_path = os.path.join(output_folder, file_name)
            binary_img.save(binary_img_path)
            print(f"Converted {file_name} to binary and saved to {output_folder}")


input_folder = 'C:/Users/ranag/Downloads/all/all/train/KAH'
grayscale_output_folder = 'C:/Users/ranag/Downloads/all/all/train/KAHgray'
binary_output_folder = 'C:/Users/ranag/Downloads/all/all/train/KAHfinal'

convert_images_to_grayscale(input_folder, grayscale_output_folder)
convert_images_to_binary(input_folder, binary_output_folder)
