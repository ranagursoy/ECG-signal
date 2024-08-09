import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.applications.xception import Xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU devices: {gpus}")
    except RuntimeError as e:
        print(e)

# Function to load and process image data
def get_image_arrays(data):
    images = []
    for i in data:
        img = Image.open(i).convert('L')  # Convert to grayscale
        img = img.resize((480, 80))  # Resize to 299x299 as expected by Xception
        img_array = np.array(img)
        images.append(img_array)
    return np.expand_dims(np.array(images), axis=-1)  # Add channel dimension

# Load file paths for each class
KAH_path = glob('KAH/*.jpeg')
KKAH_path = glob('KKAH/*.jpeg')
Normal_path = glob('Normal/*.jpeg')

# Load and process images
class0_imgs = get_image_arrays(KAH_path)
class1_imgs = get_image_arrays(KKAH_path)
class2_imgs = get_image_arrays(Normal_path)

# Labels
class0_labels = np.zeros(len(class0_imgs))
class1_labels = np.ones(len(class1_imgs))
class2_labels = np.full(len(class2_imgs), 2)

# Concatenate data
X = np.concatenate([class0_imgs, class1_imgs, class2_imgs], axis=0)
y = np.concatenate([class0_labels, class1_labels, class2_labels], axis=0)

# Normalize images
X = X / 255.0  # Normalize to [0, 1] range

# Convert labels to one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# Convert grayscale images to 3 channels by repeating the grayscale image
X_train = np.repeat(X_train, 3, axis=-1)
print(X_train.shape)
X_test = np.repeat(X_test, 3, axis=-1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Load Xception model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(80, 480, 3))

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of Xception to prevent them from being trained
for layer in base_model.layers:
    layer.trainable = False  # Initially freeze layers

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Directory for results
results_dir = 'training_results_xception'
os.makedirs(results_dir, exist_ok=True)

# Checkpoint to save the best model
checkpoint_path = os.path.join(results_dir, 'best_model_xception.h5')
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')

# Learning rate reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(results_dir, 'training_history_xception.csv'))

# Confusion matrix and classification report
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)

y_test_classes = np.argmax(y_test, axis=1)  # Convert one-hot to class indices

confusion_mtx = confusion_matrix(y_test_classes, Y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_dir, 'confusion_matrix_xception.png'))
plt.close()

report = classification_report(y_test_classes, Y_pred_classes)
with open(os.path.join(results_dir, 'classification_report_xception.txt'), 'w') as f:
    f.write(report)

print("Model training and evaluation complete. Results saved in 'training_results_xception' directory.")
