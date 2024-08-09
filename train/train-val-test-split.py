import os
import shutil
from random import shuffle

# Define paths
base_path = 'C:/Users/ranag/Desktop/ECG-Article/temizlenmis-tum-resim-warmup-recovery-ayrilmis-tum-datase/rawdata-çizgili'
train_path = os.path.join(base_path, 'train')
val_path = os.path.join(base_path, 'val')
test_path = os.path.join(base_path, 'test')

# Proportions
train_pct = 0.7
val_pct = 0.15

# Create directories
for folder in ['train', 'val', 'test']:
    for subfolder in ['KAH', 'Normal']:
        os.makedirs(os.path.join(base_path, folder, subfolder), exist_ok=True)

# Function to split data
def split_data(class_folder):
    files = os.listdir(os.path.join(base_path, class_folder))
    shuffle(files)
    
    train_end = int(len(files) * train_pct)
    val_end = train_end + int(len(files) * val_pct)

    for i, file in enumerate(files):
        if i < train_end:
            dest = os.path.join(train_path, class_folder)
        elif i < val_end:
            dest = os.path.join(val_path, class_folder)
        else:
            dest = os.path.join(test_path, class_folder)

        shutil.move(os.path.join(base_path, class_folder, file), os.path.join(dest, file))

# Apply the function to each class folder
for class_folder in ['KAH', 'Normal']:
    split_data(class_folder)

print("Files have been successfully split into train, val, and test sets.")
