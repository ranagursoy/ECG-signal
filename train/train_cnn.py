import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Enable GPU memory growth if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU devices: {gpus}")
    except RuntimeError as e:
        print(e)

def get_image_arrays(data_path, size=(480, 80)):
    """Load and process images from the given file paths."""
    images = []
    for filepath in glob(os.path.join(data_path, '*.jpeg')):
        img = Image.open(filepath).convert('L')  # Convert to grayscale
        img = img.resize(size)  # Resize images
        img_array = np.array(img)
        images.append(img_array)
    return np.array(images)

# Load images
KAH_images = get_image_arrays('KAH')
KKAH_images = get_image_arrays('KKAH')
Normal_images = get_image_arrays('Normal')

# Create labels
KAH_labels = np.zeros(len(KAH_images))
KKAH_labels = np.ones(len(KKAH_images))
Normal_labels = np.full(len(Normal_images), 2)

# Stack images and labels
X = np.concatenate([KAH_images, KKAH_images, Normal_images], axis=0)
y = np.concatenate([KAH_labels, KKAH_labels, Normal_labels], axis=0)
X = np.expand_dims(X, -1)  # Add channel dimension for CNN

# Normalize pixel values
X = X / 255.0

# One-hot encode labels
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(80, 480, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Results directory
results_dir = 'training_results'
os.makedirs(results_dir, exist_ok=True)

# Checkpoint to save the best model
checkpoint_path = os.path.join(results_dir, 'best_model.h5')
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# Reduce learning rate callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Fit the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save training history
history_df = pd.DataFrame(history.history)
history_path = os.path.join(results_dir, 'training_history.csv')
history_df.to_csv(history_path)

# Predictions
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
confusion_mtx = confusion_matrix(y_true, Y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
conf_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')
plt.savefig(conf_matrix_path)
plt.close()

# Classification report
report = classification_report(y_true, Y_pred_classes, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_path = os.path.join(results_dir, 'classification_report.csv')
report_df.to_csv(report_path)

print("Model training and evaluation complete. Results saved in 'training_results' directory.")
