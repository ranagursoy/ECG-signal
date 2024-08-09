import os
from PIL import Image

def crop_images_in_folder(source_folder, target_folder, crop_bottom=10):
    # Hedef klasör yoksa oluştur
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Belirtilen klasördeki tüm dosyaları listele
    for file_name in os.listdir(source_folder):
        if file_name.lower().endswith(('.jpeg', '.jpg')):  # JPEG ve JPG dosyalarını kontrol et
            image_path = os.path.join(source_folder, file_name)
            
            try:
                # Resmi yükleyin
                image = Image.open(image_path)

                # Resmi kesin
                cropped_image = image.crop((38, 0, image.width, image.height - crop_bottom))

                # Kesilmiş resmi kaydedin
                new_file_name = file_name
                cropped_image.save(os.path.join(target_folder, new_file_name))
                print(f"Cropped image saved as {new_file_name}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

# Kaynak klasör yolunu belirtin
source_folder = 'C:/Users/ranag/Downloads/new_signal1_v5_v6'
# Hedef klasör yolunu belirtin
target_folder = 'C:/Users/ranag/Downloads/crop'
# Alt kısmından 10 piksel kadar kesmek için
crop_images_in_folder(source_folder, target_folder, crop_bottom=10)


