import os
import shutil
import pandas as pd

# .xlsx dosyasını oku
file_path = 'C:/Users/ranag/Documents/GitHub/EKG-signal/Labellar_efor_testi_bileske_anonim.xlsx'
data = pd.read_excel(file_path)

# Klasör adlarını belirleyin
class_folders = ['KAH', 'KKAH', 'Normal']

# Her bir sınıf için klasör oluştur
for folder in class_folders:
    os.makedirs(folder, exist_ok=True)

# Resimlerin bulunduğu ana klasör yolu
image_folder_path = 'D:/signal_1/signal_1'

# Dosyaları listele ve göster
all_files = os.listdir(image_folder_path)
print("Available files in the directory:", all_files)

# Türkçe karakter sorunlarını çözmek için bir yardımcı fonksiyon
def normalize_text(text):
    replacements = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U',
        ' ': '_', '-': '_', '/': '_'
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text.lower()

# Her satırı kontrol et ve uygun klasöre dosyaları taşı
for index, row in data.iterrows():
    folder_name = normalize_text(row['KlasorAdi'])
    print(f"Looking for files matching: {folder_name}")

    patient_files = [f for f in all_files if folder_name in normalize_text(f)]
    print(f"Matching files: {patient_files}")

    if row['KAH'] == 1:
        destination_folder = 'KAH'
    elif row['KKAH'] == 1:
        destination_folder = 'KKAH'
    else:
        destination_folder = 'Normal'

    for file in patient_files:
        source_path = os.path.join(image_folder_path, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.move(source_path, destination_path)
        print(f'Moved {file} to {destination_folder}')

print("Dosyalar başarıyla taşındı.")
