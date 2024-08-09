import os
import shutil
import pandas as pd
import unicodedata

# Excel dosyasını oku
file_path = 'C:/Users/ranag/Documents/GitHub/EKG-signal/data/Labellar_efor_testi_bileske_anonim.xlsx'
data = pd.read_excel(file_path)

# Klasör adlarını belirleyin
class_folders = ['KAH', 'KKAH', 'Normal']

# Her bir sınıf için klasör oluştur
for folder in class_folders:
    os.makedirs(folder, exist_ok=True)

# Resimlerin bulunduğu ana klasör yolu
image_folder_path = 'C:/Users/ranag/Desktop/ECG-Article/temizlenmis-tum-resim-warmup-recovery-ayrilmis-tum-datase/warmup-recovery-ayrilmis-tum-dataset'

# Dosyaları listele
all_directories = [d for d in os.listdir(image_folder_path) if os.path.isdir(os.path.join(image_folder_path, d))]

# Türkçe karakter sorunlarını çözmek için bir yardımcı fonksiyon
def normalize_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    replacements = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U',
        ' ': '_', '-': '_', '/': '_'
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text.lower()

# Normalize edilmiş klasör adlarını hesapla
data['NormalizedKlasorAdi'] = data['KlasorAdi'].apply(normalize_text)

# Debugging: Normalize edilmiş klasör adlarını ve dizin adlarını yazdır
print("Unique normalized folder names in Excel:", data['NormalizedKlasorAdi'].nunique())
normalized_directories = [normalize_text(d) for d in all_directories]
print("Unique normalized directories in path:", len(set(normalized_directories)))

# Her satırı kontrol et ve uygun klasöre dosyaları taşı
for index, row in data.iterrows():
    folder_name = normalize_text(row['KlasorAdi'])
    if folder_name in normalized_directories:
        # Define the source directory path
        source_directory = os.path.join(image_folder_path, all_directories[normalized_directories.index(folder_name)])
        
        # Determine the destination folder based on classification
        if row['KAH'] == 1:
            destination_folder = 'KAH'
        elif row['KKAH'] == 1:
            destination_folder = 'KKAH'
        else:
            destination_folder = 'Normal'
        
        # Move all files from the source directory to the destination folder
        for file in os.listdir(source_directory):
            source_path = os.path.join(source_directory, file)
            destination_path = os.path.join(destination_folder, file)
            
            # Move the file to the appropriate class directory
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
                print(f'Moved {file} from {folder_name} to {destination_folder}')
            else:
                print(f"File not found: {source_path}")

# Eşleşmeyen klasör ve dosya adlarını kontrol et
unmatched_folders = set(data['NormalizedKlasorAdi']) - set(normalized_directories)
unmatched_directories = set(normalized_directories) - set(data['NormalizedKlasorAdi'])

print(f"Eşleşmeyen klasör adları (Excel'de olup dizin olarak bulunmayan): {len(unmatched_folders)}")
print(f"Eşleşmeyen dizinler (klasörde olup Excel'de bulunmayan): {len(unmatched_directories)}")

if unmatched_folders:
    print("Eşleşmeyen klasör adları:", list(unmatched_folders)[:10])  # Sadece ilk 10 tanesini yazdır
if unmatched_directories:
    print("Eşleşmeyen dizinler:", list(unmatched_directories)[:10])  # Sadece ilk 10 tanesini yazdır

print("Dosyalar başarıyla taşındı.")
