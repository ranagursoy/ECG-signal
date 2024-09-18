from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import os
from glob import glob

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU devices: {gpus}")
    except RuntimeError as e:
        print(e)

# Set the image size
IMAGE_SIZE = [224, 224]

# Define the paths for training and validation datasets
train_path = 'C:/Users/asil_/Downloads/archive/ECG_Image_data/train'
valid_path = 'C:/Users/asil_/Downloads/archive/ECG_Image_data/test'

# Load the VGG16 model without the top layers
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze all the layers in the VGG16 model
for layer in vgg.layers:
    layer.trainable = False

# Get the number of classes (folders)
folders = glob('C:/Users/asil_/Downloads/archive/ECG_Image_data/train/*')
num_classes = len(folders)
print(folders)

# Add custom layers on top of VGG16
x = Flatten()(vgg.output)
prediction = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=vgg.input, outputs=prediction)

# Display the model architecture
model.summary()

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Adjust the learning rate as per your need
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

# Flow data from directories
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

val_generator = test_val_datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_val_datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for getting correct labels when predicting
)

# Directory for results
results_dir = 'training_results_vgg16_6_classes_1'
os.makedirs(results_dir, exist_ok=True)

checkpoint_path = os.path.join(results_dir, 'best_model_vgg16.keras')
checkpoint = ModelCheckpoint(
    checkpoint_path,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')

# Learning rate reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(results_dir, 'training_history_vgg16.csv'))

# Evaluate the model on train data
train_generator.reset()
Y_train_pred = model.predict(train_generator)
Y_train_pred_classes = np.argmax(Y_train_pred, axis=1)
y_train_classes = train_generator.classes  # True labels

# Confusion matrix and classification report for train data
confusion_mtx_train = confusion_matrix(y_train_classes, Y_train_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx_train, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Train Data)')
plt.savefig(os.path.join(results_dir, 'confusion_matrix_train_vgg16.png'))
plt.close()

report_train = classification_report(y_train_classes, Y_train_pred_classes, target_names=list(train_generator.class_indices.keys()))
with open(os.path.join(results_dir, 'classification_report_train_vgg16.txt'), 'w') as f:
    f.write(report_train)

# Evaluate the model on validation data
val_generator.reset()
Y_val_pred = model.predict(val_generator)
Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
y_val_classes = val_generator.classes  # True labels

# Confusion matrix and classification report for validation data
confusion_mtx_val = confusion_matrix(y_val_classes, Y_val_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx_val, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Validation Data)')
plt.savefig(os.path.join(results_dir, 'confusion_matrix_val_vgg16.png'))
plt.close()

report_val = classification_report(y_val_classes, Y_val_pred_classes, target_names=list(val_generator.class_indices.keys()))
with open(os.path.join(results_dir, 'classification_report_val_vgg16.txt'), 'w') as f:
    f.write(report_val)

# Evaluate the model on test data
test_generator.reset()
Y_pred = model.predict(test_generator)
Y_pred_classes = np.argmax(Y_pred, axis=1)
y_test_classes = test_generator.classes  # True labels

# Handle the case where there are missing classes in predictions
unique_true_labels = np.unique(y_test_classes)
unique_pred_labels = np.unique(Y_pred_classes)

# Tüm sınıflar (6 sınıf olduğunu varsayıyoruz)
all_classes = np.array(range(len(test_generator.class_indices)))

# Eksik sınıfları doldur
missing_true_labels = np.setdiff1d(all_classes, unique_true_labels)
missing_pred_labels = np.setdiff1d(all_classes, unique_pred_labels)

# Gerçek sınıflara eksik sınıfları ekle
for label in missing_true_labels:
    y_test_classes = np.append(y_test_classes, label)
    Y_pred_classes = np.append(Y_pred_classes, label)

# Tahmin edilen sınıflara eksik sınıfları ekle
for label in missing_pred_labels:
    Y_pred_classes = np.append(Y_pred_classes, label)
    y_test_classes = np.append(y_test_classes, label)

# Confusion matrix and classification report for test data
confusion_mtx = confusion_matrix(y_test_classes, Y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Data)')
plt.savefig(os.path.join(results_dir, 'confusion_matrix_vgg16.png'))
plt.close()

report = classification_report(y_test_classes, Y_pred_classes, target_names=list(test_generator.class_indices.keys()))
with open(os.path.join(results_dir, 'classification_report_vgg16.txt'), 'w') as f:
    f.write(report)

print("Model training and evaluation complete. Results saved in 'training_results_vgg16_6_classes_1' directory.")
