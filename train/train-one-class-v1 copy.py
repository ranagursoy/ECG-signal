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

# Evaluate on test set using different thresholds
reconstructed_test_normal = autoencoder.predict(normal_test)
reconstructed_test_anomalous = autoencoder.predict(anomalous_test)
mse_test_normal = np.mean(np.square(normal_test - reconstructed_test_normal), axis=(1, 2, 3))
mse_test_anomalous = np.mean(np.square(anomalous_test - reconstructed_test_anomalous), axis=(1, 2, 3))

output_folder = "model_results_" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(output_folder, exist_ok=True)

thresholds = range(10, 30)  # From 70 to 95 percentiles
for threshold in thresholds:
    percentile_threshold = np.percentile(mse_test_normal, threshold)
    predictions_test_normal = (mse_test_normal <= percentile_threshold).astype(int)
    predictions_test_anomalous = (mse_test_anomalous > percentile_threshold).astype(int)
    
    all_predictions_test = np.concatenate([predictions_test_normal, predictions_test_anomalous])
    all_true_labels_test = np.concatenate([np.zeros_like(predictions_test_normal), np.ones_like(predictions_test_anomalous)])
    
    conf_matrix = confusion_matrix(all_true_labels_test, all_predictions_test)
    class_report = classification_report(all_true_labels_test, all_predictions_test)
    
    # Save results for each threshold
    threshold_folder = os.path.join(output_folder, f"threshold_{threshold}")
    os.makedirs(threshold_folder, exist_ok=True)
    
    with open(os.path.join(threshold_folder, 'classification_report.txt'), 'w') as f:
        f.write("Test Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix))
        f.write("\n\nTest Classification Report:\n")
        f.write(class_report)
    
    # Plot and save the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix at Threshold {threshold}')
    plt.savefig(os.path.join(threshold_folder, 'confusion_matrix.png'))
    plt.close()

# Plot and save training and validation loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history.get('accuracy', []), label='Training Accuracy')
plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.savefig(os.path.join(output_folder, 'training_validation_loss_accuracy.png'))
plt.close()

# Save the model
autoencoder.save(os.path.join(output_folder, 'autoencoder_model.h5'))
