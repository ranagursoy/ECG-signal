import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping

# Image dimensions and VGG16 model preparation
img_height, img_width = 128, 128
input_shape = (img_height, img_width, 3)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False  # Freeze the base model

# Feature extraction model
feature_extractor = keras.models.Sequential([
    base_model,
    keras.layers.Flatten()
])

# Autoencoder model definition
input_layer = keras.Input(shape=input_shape)
x = feature_extractor(input_layer)
encoded = keras.layers.Dense(128, activation='relu')(x)
encoded = keras.layers.Dense(64, activation='relu')(encoded)
decoded = keras.layers.Dense(128, activation='relu')(encoded)
decoded = keras.layers.Dense(np.prod(input_shape), activation='sigmoid')(decoded)
decoded = keras.layers.Reshape(input_shape)(decoded)
autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Function to load images from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpeg"):  
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=(img_height, img_width))
            img_array = img_to_array(img) / 255.0  
            images.append(img_array)
    return np.array(images)

# Load images
normal_images = load_images_from_folder("C:/Users/ranag/Documents/GitHub/EKG-signal/Normal_v5")
anomalous_images = load_images_from_folder("C:/Users/ranag/Documents/GitHub/EKG-signal/KAH_v5")

# Split data
normal_train, normal_temp = train_test_split(normal_images, test_size=0.3, random_state=42)
normal_val, normal_test = train_test_split(normal_temp, test_size=0.5, random_state=42)
anomalous_train, anomalous_temp = train_test_split(anomalous_images, test_size=0.3, random_state=42)
anomalous_val, anomalous_test = train_test_split(anomalous_temp, test_size=0.5, random_state=42)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5,  
    min_delta=0.001,  
    restore_best_weights=True, 
    verbose=1  
)

# Train the model
history = autoencoder.fit(
    normal_train, normal_train,
    epochs=50,
    batch_size=16,
    validation_data=(normal_val, normal_val),
    callbacks=[early_stopping],  
    verbose=1
)

# Evaluate on train set
reconstructed_train_normal = autoencoder.predict(normal_train)
reconstructed_train_anomalous = autoencoder.predict(anomalous_train)
mse_train_normal = np.mean(np.square(normal_train - reconstructed_train_normal), axis=(1, 2, 3))
mse_train_anomalous = np.mean(np.square(anomalous_train - reconstructed_train_anomalous), axis=(1, 2, 3))
threshold_train = np.percentile(mse_train_normal, 8)  # Calculate threshold at 8th percentile for train

predictions_train_normal = (mse_train_normal <= threshold_train).astype(int)
predictions_train_anomalous = (mse_train_anomalous > threshold_train).astype(int)

all_predictions_train = np.concatenate([predictions_train_normal, predictions_train_anomalous])
all_true_labels_train = np.concatenate([np.zeros_like(predictions_train_normal), np.ones_like(predictions_train_anomalous)])

# Confusion Matrix and Classification Report for train data
conf_matrix_train = confusion_matrix(all_true_labels_train, all_predictions_train)
class_report_train = classification_report(all_true_labels_train, all_predictions_train)

# Save results for train set
output_folder = "model_results_" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(output_folder, exist_ok=True)

# Save the confusion matrix and classification report for train data
with open(os.path.join(output_folder, 'train_classification_report.txt'), 'w') as f:
    f.write("Train Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix_train))
    f.write("\n\nTrain Classification Report:\n")
    f.write(class_report_train)

# Plot and save the confusion matrix for train data
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Train Confusion Matrix')
plt.savefig(os.path.join(output_folder, 'train_confusion_matrix.png'))
plt.close()

# Evaluate on test set
reconstructed_test_normal = autoencoder.predict(normal_test)
reconstructed_test_anomalous = autoencoder.predict(anomalous_test)
mse_test_normal = np.mean(np.square(normal_test - reconstructed_test_normal), axis=(1, 2, 3))
mse_test_anomalous = np.mean(np.square(anomalous_test - reconstructed_test_anomalous), axis=(1, 2, 3))
threshold_test = np.percentile(mse_test_normal, 8)  # Calculate threshold at 8th percentile for test

predictions_test_normal = (mse_test_normal <= threshold_test).astype(int)
predictions_test_anomalous = (mse_test_anomalous > threshold_test).astype(int)

all_predictions_test = np.concatenate([predictions_test_normal, predictions_test_anomalous])
all_true_labels_test = np.concatenate([np.zeros_like(predictions_test_normal), np.ones_like(predictions_test_anomalous)])

# Confusion Matrix and Classification Report for test data
conf_matrix_test = confusion_matrix(all_true_labels_test, all_predictions_test)
class_report_test = classification_report(all_true_labels_test, all_predictions_test)

# Save the confusion matrix and classification report for test data
with open(os.path.join(output_folder, 'test_classification_report.txt'), 'w') as f:
    f.write("Test Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix_test))
    f.write("\n\nTest Classification Report:\n")
    f.write(class_report_test)

# Plot and save the confusion matrix for test data
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Test Confusion Matrix')
plt.savefig(os.path.join(output_folder, 'test_confusion_matrix.png'))
plt.close()

# Save the model
autoencoder.save(os.path.join(output_folder, 'autoencoder_model.h5'))
