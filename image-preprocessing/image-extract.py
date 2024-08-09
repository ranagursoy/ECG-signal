import os
from PIL import Image

# İşlem yapılacak ana klasörün yolu
input_root_folder_path = 'C:/Users/ranag/Desktop/ECG-Article/temizlenmis-tum-resim-warmup-recovery-ayrilmis-tum-datase/warmup-recovery-ayrilmis-tum-dataset'

# Çıkış klasörünün yolunu belirle
output_folder_path = 'C:/Users/ranag/Desktop/ECG-Article/temizlenmis-tum-resim-warmup-recovery-ayrilmis-tum-datase/rawdata-çizgili'

# Eğer çıkış klasörü yoksa oluştur
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Ana klasördeki her alt klasörü kontrol et
for parent_folder in os.listdir(input_root_folder_path):
    parent_folder_path = os.path.join(input_root_folder_path, parent_folder)

    # Eğer bu bir klasörse
    if os.path.isdir(parent_folder_path):
        # Alt klasördeki "field" ile başlayan JPEG dosyalarını bul
        field_jpeg_files = [f for f in os.listdir(parent_folder_path) if f.lower().startswith('field') and f.lower().endswith(('.jpg', '.jpeg'))]

        if field_jpeg_files:
            # "Field" ile başlayan ilk JPEG dosyasını seç
            filename = field_jpeg_files[0]
            input_image_path = os.path.join(parent_folder_path, filename)

            try:
                # Resmi aç
                image = Image.open(input_image_path)

                # Yeni dosya ismini klasör ismine göre oluştur
                base_name, ext = os.path.splitext(filename)
                new_filename = f"{parent_folder}{ext}"
                tile_path = os.path.join(output_folder_path, new_filename)
                image.save(tile_path)

                # Kaydedilen resmin yolunu yazdır
                print(f"Image saved to: {tile_path}")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

print("Her alt klasörden 'field' ile başlayan bir JPEG dosyası başarıyla işlendi ve kaydedildi.")
