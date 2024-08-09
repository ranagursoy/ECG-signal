import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_v5_signal(image_path, plot_histogram=False):
    # Görüntüyü yükle
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Eğer görüntü None ise, dosya yolu kontrol et
    if image is None:
        raise ValueError("Görüntü yüklenemedi. Dosya yolunu kontrol edin.")
    
    # Görüntü histogramını çıkar ve eşik değer belirle
    if plot_histogram:
        plt.hist(image.ravel(), 256, [0,256])
        plt.title("Histogram")
        plt.show()
    
    # Otomatik eşikleme kullanarak görüntüyü ikili hale getir
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Sinyali çıkar
    # Belki EKG sinyalinin bulunduğu yatay konumu ayarlamak gerekebilir
    row_index = int(binary_image.shape[0] * 0.8)  # Örneğin yüksekliğin %80'i
    extracted_signal = binary_image[row_index, :]
    
    # Görüntüleme (opsiyonel)
    plt.figure(figsize=(10, 2))
    plt.plot(extracted_signal, color='black')
    plt.title('Extracted V5 Signal')
    plt.xlabel('Time (pixels)')
    plt.ylabel('Amplitude (pixel intensity)')
    plt.show()
    
    return extracted_signal

# Fonksiyonu çağır
v5_signal = extract_v5_signal('deneme.jpeg', plot_histogram=True)
