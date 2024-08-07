import os
import shutil
from random import shuffle

# Define paths
base_path = 'C:/Users/ranag/Desktop/full-data'

# Ensure base directories exist and create them if not
class_folders = ['KAH', 'KKAH', 'Normal']
versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']

# Check and create source class directories if they do not exist
for folder in class_folders:
    for version in versions:
        os.makedirs(os.path.join(base_path, folder + '_' + version), exist_ok=True)

# Define subpaths for train, val, and test datasets
train_path = os.path.join(base_path, 'train')
val_path = os.path.join(base_path, 'val')
test_path = os.path.join(base_path, 'test')

# Proportions for splitting
train_pct = 0.7
val_pct = 0.15

# Create directories for each set and version
for set_folder in [train_path, val_path, test_path]:
    for folder in class_folders:
        for version in versions:
            os.makedirs(os.path.join(set_folder, folder + '_' + version), exist_ok=True)

# Function to split data into train, val, and test sets
def split_data(class_folder):
    files = os.listdir(os.path.join(base_path, class_folder))
    shuffle(files)
    
    # Calculate the indices for splitting
    train_end = int(len(files) * train_pct)
    val_end = train_end + int(len(files) * val_pct)

    # Split files into train, val, and test sets
    for i, file in enumerate(files):
        if i < train_end:
            dest = os.path.join(train_path, class_folder)
        elif i < val_end:
            dest = os.path.join(val_path, class_folder)
        else:
            dest = os.path.join(test_path, class_folder)

        # Move files to their new locations
        shutil.move(os.path.join(base_path, class_folder, file), os.path.join(dest, file))

# Apply the function to each class folder
for folder in class_folders:
    for version in versions:
        class_version_folder = folder + '_' + version
        split_data(class_version_folder)

print("Files have been successfully split into train, val, and test sets.")
