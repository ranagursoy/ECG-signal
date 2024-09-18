import os
import shutil
import random
from collections import Counter

# Define directories
base_dir = 'C:/Users/asil_/Downloads/archive/ECG_Image_data'  # Adjust this if necessary
combined_dir = os.path.join(base_dir, 'combined')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Step 1: Combine the data
os.makedirs(combined_dir, exist_ok=True)

for class_name in os.listdir(train_dir):
    class_train_path = os.path.join(train_dir, class_name)
    class_test_path = os.path.join(test_dir, class_name)
    class_combined_path = os.path.join(combined_dir, class_name)
    
    os.makedirs(class_combined_path, exist_ok=True)
    
    # Copy train images
    for file_name in os.listdir(class_train_path):
        source = os.path.join(class_train_path, file_name)
        destination = os.path.join(class_combined_path, file_name)
        shutil.copy(source, destination)
    
    # Copy test images
    for file_name in os.listdir(class_test_path):
        source = os.path.join(class_test_path, file_name)
        destination = os.path.join(class_combined_path, file_name)
        shutil.copy(source, destination)

# Step 2: Analyze the distribution of images
combined_class_counts = {}
for class_name in os.listdir(combined_dir):
    class_combined_path = os.path.join(combined_dir, class_name)
    image_count = len(os.listdir(class_combined_path))
    combined_class_counts[class_name] = image_count

print("Image distribution after combining:")
for class_name, count in combined_class_counts.items():
    print(f"{class_name}: {count} images")
# Step 3: Balance the classes (if needed)
# Optionally, you can balance the classes by limiting the number of images in each class.
# For this example, we'll just ensure each class has the same number of images as the smallest class.
min_images = min(combined_class_counts.values())
for class_name in os.listdir(combined_dir):
    class_combined_path = os.path.join(combined_dir, class_name)
    images = os.listdir(class_combined_path)
    if len(images) > min_images:
        images_to_remove = random.sample(images, len(images) - min_images)
        for img in images_to_remove:
            os.remove(os.path.join(class_combined_path, img))

# Step 4: Split the data into train, test, and val sets
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
output_dirs = {k: os.path.join(base_dir, k) for k in split_ratios.keys()}

for class_name in os.listdir(combined_dir):
    class_combined_path = os.path.join(combined_dir, class_name)
    images = os.listdir(class_combined_path)
    random.shuffle(images)
    
    num_images = len(images)
    train_split = int(split_ratios['train'] * num_images)
    val_split = int(split_ratios['val'] * num_images)
    
    splits = {
        'train': images[:train_split],
        'val': images[train_split:train_split + val_split],
        'test': images[train_split + val_split:]
    }
    
    for split_name, split_images in splits.items():
        split_class_dir = os.path.join(output_dirs[split_name], class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        
        for img_name in split_images:
            source = os.path.join(class_combined_path, img_name)
            destination = os.path.join(split_class_dir, img_name)
            shutil.copy(source, destination)

print("Data split completed.")
for split_name, split_dir in output_dirs.items():
    print(f"{split_name} set:")
    for class_name in os.listdir(split_dir):
        class_split_dir = os.path.join(split_dir, class_name)
        print(f"  {class_name}: {len(os.listdir(class_split_dir))} images")
