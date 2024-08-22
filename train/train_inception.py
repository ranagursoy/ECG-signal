import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

# Directories for training, validation, and test sets
train_dir = 'C:/Users/ranag/Desktop/tüm_ekg/çizgisiz/thr/eşitlenmiş/train'
val_dir = 'C:/Users/ranag/Desktop/tüm_ekg/çizgisiz/thr/eşitlenmiş/val'
test_dir = 'C:/Users/ranag/Desktop/tüm_ekg/çizgisiz/thr/eşitlenmiş/test'

# Flow data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical'
)

val_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical'
)

test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical',
    shuffle=False  # Important for getting correct labels when predicting
)

# Load InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze the first 20 layers of InceptionV3 to prevent them from being trained
for layer in base_model.layers[:3]:
    layer.trainable = False

# Keep the rest of the layers trainable
for layer in base_model.layers[3:]:
    layer.trainable = True

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)  # Daha büyük bir Dense katmanı
x = Dropout(0.2)(x)  # Dropout'u artırmak
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

# Directory for results
results_dir = 'training_results_inception_9'
os.makedirs(results_dir, exist_ok=True)

# Checkpoint to save the best model
checkpoint_path = os.path.join(results_dir, 'best_model_inception.h5')
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=False)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')

# Learning rate reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(results_dir, 'training_history_inception.csv'))

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
plt.savefig(os.path.join(results_dir, 'confusion_matrix_train_inception.png'))
plt.close()

report_train = classification_report(y_train_classes, Y_train_pred_classes, target_names=list(train_generator.class_indices.keys()))
with open(os.path.join(results_dir, 'classification_report_train_inception.txt'), 'w') as f:
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
plt.savefig(os.path.join(results_dir, 'confusion_matrix_val_inception.png'))
plt.close()

report_val = classification_report(y_val_classes, Y_val_pred_classes, target_names=list(val_generator.class_indices.keys()))
with open(os.path.join(results_dir, 'classification_report_val_inception.txt'), 'w') as f:
    f.write(report_val)

# Evaluate the model on test data
test_generator.reset()
Y_pred = model.predict(test_generator)
Y_pred_classes = np.argmax(Y_pred, axis=1)
y_test_classes = test_generator.classes  # True labels

# Confusion matrix and classification report for test data
confusion_mtx = confusion_matrix(y_test_classes, Y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Data)')
plt.savefig(os.path.join(results_dir, 'confusion_matrix_inception.png'))
plt.close()

report = classification_report(y_test_classes, Y_pred_classes, target_names=list(test_generator.class_indices.keys()))
with open(os.path.join(results_dir, 'classification_report_inception.txt'), 'w') as f:
    f.write(report)

print("Model training and evaluation complete. Results saved in 'training_results_inception_9' directory.")
