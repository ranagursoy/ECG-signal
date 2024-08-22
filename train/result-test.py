import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model dosyalarının bulunduğu dizin
model_dir = r'D:\tüm_ekg\result-çizgisiz-eşitlenmemiş'

# Verinin olduğu dizinler
validation_dir = r'D:\tüm_ekg\result-çizgisiz-eşitlenmemiş\val'  # Doğrulama verisinin yolu

# Data generator hazırlığı
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='binary'
)

# Doğrulama veri setindeki toplam görüntü sayısını yazdır
print(f"Found {validation_generator.samples} images belonging to {validation_generator.num_classes} classes.")

best_val_accuracy = 0
best_model_file = None

# Dizin altındaki tüm .keras dosyalarını bulalım
model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]

# Tüm modelleri yükleyip, doğrulama seti üzerinde doğruluğunu kontrol edelim
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    model = tf.keras.models.load_model(model_path)
    
    # Modeli doğrulama seti üzerinde değerlendir
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
    
    print(f"Model: {model_file}, Val Accuracy: {val_accuracy:.4f}")
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_file = model_file

# En iyi modeli ve doğrulama doğruluğunu yazdır
if best_model_file:
    print(f"\nEn iyi model: {best_model_file}, Val Accuracy: {best_val_accuracy:.4f}")
else:
    print("Hiçbir model için val_accuracy bilgisi bulunamadı.")
