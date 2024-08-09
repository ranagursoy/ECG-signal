import os
from PIL import Image

def convert_images_to_grayscale(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.lower().startswith('field-') and file_name.lower().endswith(('.jpeg')):
                img_path = os.path.join(root, file_name)
                img = Image.open(img_path)
                gray_img = img.convert("L")
                relative_path = os.path.relpath(root, input_folder)
                gray_img_folder = os.path.join(output_folder, relative_path)
                
                if not os.path.exists(gray_img_folder):
                    os.makedirs(gray_img_folder)
                
                gray_img_path = os.path.join(gray_img_folder, file_name)
                gray_img.save(gray_img_path)
                print(f"Converted {file_name} to grayscale and saved to {gray_img_path}")

def convert_images_to_binary(input_folder, output_folder, threshold=128):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.lower().startswith('field-') and file_name.lower().endswith(('.jpeg')):
                img_path = os.path.join(root, file_name)
                img = Image.open(img_path)
                gray_img = img.convert("L")
                binary_img = gray_img.point(lambda x: 255 if x > threshold else 0, '1')
                relative_path = os.path.relpath(root, input_folder)
                binary_img_folder = os.path.join(output_folder, relative_path)
                
                if not os.path.exists(binary_img_folder):
                    os.makedirs(binary_img_folder)
                
                binary_img_path = os.path.join(binary_img_folder, file_name)
                binary_img.save(binary_img_path)
                print(f"Converted {file_name} to binary and saved to {binary_img_path}")

# Ana klasör yolu ve çıkış klasörleri
input_folder = 'C:/Users/ranag/Desktop/temizlenmis-tum-resim-warmup-recovery-ayrilmis-tum-datase/warmup-recovery-ayrilmis-tum-dataset/'
#grayscale_output_folder = 'C:/Users/ranag/Desktop/temizlenmis-tum-resim-warmup-recovery-ayrilmis-tum-datase/gary'
binary_output_folder = 'C:/Users/ranag/Desktop/temizlenmis-tum-resim-warmup-recovery-ayrilmis-tum-datase/binary'

# Fonksiyonları çağır
#convert_images_to_grayscale(input_folder, grayscale_output_folder)
convert_images_to_binary(input_folder, binary_output_folder)
