import os
from PIL import Image

def crop_images_in_folder(source_folder, target_folder):
    # Hedef klasör yoksa oluştur
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Belirtilen klasördeki tüm dosyaları listele
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.jpeg'):  # JPEG dosyalarını kontrol et
            # Resmi yükleyin
            image_path = os.path.join(source_folder, file_name)
            image = Image.open(image_path)

            # Resmi kesin (Buradaki değerleri ihtiyacınıza göre ayarlayın)
            cropped_image = image.crop((42, 0, image.width, image.height))

            # Kesilmiş resmi kaydedin
            new_file_name = file_name
            cropped_image.save(os.path.join(target_folder, new_file_name))
            print(f"Cropped image saved as {new_file_name}")

# Kaynak klasör yolunu belirtin
source_folder = 'D:/signal_v5_çizgili_Edited/Normal'
# Hedef klasör yolunu belirtin
target_folder = 'D:/signal_v5_çizgili_Edited/edited-v5/Normal'
crop_images_in_folder(source_folder, target_folder)
