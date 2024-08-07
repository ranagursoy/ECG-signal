import os
import shutil
import pandas as pd
import unicodedata

# Excel dosyasını oku
file_path = 'C:/Users/ranag/Documents/GitHub/EKG-signal/Labellar_efor_testi_bileske_anonim.xlsx'
data = pd.read_excel(file_path)

# Klasör adlarını belirleyin
class_folders = ['KAH_v5', 'KKAH_v5', 'Normal_v5']

# Her bir sınıf için klasör oluştur
for folder in class_folders:
    os.makedirs(folder, exist_ok=True)

# Resimlerin bulunduğu ana klasör yolu
image_folder_path = 'C:/Users/ranag/Downloads/crop'

# Dosyaları listele
all_files = os.listdir(image_folder_path)

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

# Debugging: Normalize edilmiş KlasorAdi değerlerini yazdır
data['NormalizedKlasorAdi'] = data['KlasorAdi'].apply(normalize_text)
print("Unique normalized folder names in Excel:", data['NormalizedKlasorAdi'].nunique())

# Debugging: Tüm dosyaların normalize edilmiş isimlerini yazdır
normalized_files = [normalize_text(f) for f in all_files]
print("Unique normalized file names in directory:", len(set(normalized_files)))

# Her satırı kontrol et ve uygun klasöre dosyaları taşı
for index, row in data.iterrows():
    folder_name = normalize_text(row['KlasorAdi'])
    patient_files = [f for f in all_files if folder_name in normalize_text(f)]

    if row['KAH'] == 1:
        destination_folder = 'KAH_v5'
    elif row['KKAH'] == 1:
        destination_folder = 'KKAH_v5'
    else:
        destination_folder = 'Normal_v5'

    for file in patient_files:
        source_path = os.path.join(image_folder_path, file)
        destination_path = os.path.join(destination_folder, file)
        
        # Check if the source file exists
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f'Moved {file} to {destination_folder}')
        else:
            print(f"File not found: {source_path}")

# Eşleşmeyen dosya ve klasör adlarını kontrol et
unmatched_folders = set(data['NormalizedKlasorAdi']) - set(normalized_files)
unmatched_files = set(normalized_files) - set(data['NormalizedKlasorAdi'])

print(f"Eşleşmeyen klasör adları (Excel'de olup dosya olarak bulunmayan): {len(unmatched_folders)}")
print(f"Eşleşmeyen dosyalar (klasörde olup Excel'de bulunmayan): {len(unmatched_files)}")

if unmatched_folders:
    print("Eşleşmeyen klasör adları:", list(unmatched_folders)[:10])  # Sadece ilk 10 tanesini yazdır
if unmatched_files:
    print("Eşleşmeyen dosyalar:", list(unmatched_files)[:10])  # Sadece ilk 10 tanesini yazdır

print("Dosyalar başarıyla taşındı.")
