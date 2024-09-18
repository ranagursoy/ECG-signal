# ECG Signal Classification

This repository contains a deep learning-based approach to classify ECG signals into various categories using the **InceptionV3** model with transfer learning. The primary goal of this project is to accurately classify ECG data to assist in the diagnosis of cardiac conditions.

## Project Overview

### InceptionV3 Model
The **InceptionV3** model was developed to provide high performance and efficiency in image classification tasks. This model architecture breaks down large convolutions (such as 7x7) into smaller and asymmetrical convolutions (like 3x3 and 1x7), reducing computational costs while enhancing learning capacity. Furthermore, auxiliary classifiers are utilized to stabilize the training process by preventing gradient loss. These innovations make InceptionV3 a highly efficient model widely adopted in modern image classification projects.

### Transfer Learning with InceptionV3
In this study, the InceptionV3-based model was retrained for ECG signal classification using **transfer learning**. Pretrained weights from the **ImageNet** dataset were retained, freezing all but the last three layers to leverage InceptionV3's strong feature extraction capabilities. The model input dimensions were set to **299x299 pixels**, matching the InceptionV3 architecture.

A **GlobalAveragePooling2D** layer was added to convert the spatial dimensions into dense feature vectors, followed by a **Dense** layer for final classification using a **Softmax** activation function. This allowed the model to output probability distributions across the classes.

### Dataset and Data Preparation
The dataset used in this project includes ECG images categorized into various classes such as:
- **Myocardial Infarction**
- **Abnormal Heartbeat**
- **History of Myocardial Infarction**
- **Normal ECG**

To handle class imbalances, the dataset was balanced to match the smallest class (172 samples). The data was then split into **70% training**, **15% validation**, and **15% testing** sets.

### Model Training
The model was trained for **200 epochs** using hyperparameters optimized through **Grid Search**. The best parameters identified were:
- **Batch Size**: 32
- **Learning Rate**: 0.01
- **Dropout Rate**: 0.5
- **Unfrozen Layers**: -10

During training, the following callback mechanisms were employed:
- **EarlyStopping**: Stopped training if validation loss did not improve for 10 consecutive epochs.
- **ModelCheckpoint**: Saved the best model based on validation accuracy and loss.

### Results
The InceptionV3-based model achieved the following results on the test dataset:
- **Overall Accuracy**: 93.27%
- **Precision (Micro Average)**: 93.27%
- **Recall (Micro Average)**: 93.27%
- **F1 Score (Micro Average)**: 93.27%
- **Precision (Macro Average)**: 93.61%
- **Recall (Macro Average)**: 93.27%
- **F1 Score (Macro Average)**: 93.19%
- **Test Loss**: 0.2413

The analysis of the **confusion matrix** revealed that the "History of Myocardial Infarction" and "Normal" classes were classified with **100% accuracy**. On the other hand, the "Myocardial Infarction" and "Abnormal Heartbeat" classes showed some level of misclassification, with true positive rates of 88.46% and 84.62%, respectively.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/ranagursoy/ECG-signal.git
## Citation

This project and its findings were based on the work published in the following paper:

- **Rana Gürsoy**, et al., "Classification of ECG Signals using Deep Learning," *arXiv*, 2024. [DOI: https://doi.org/10.48550/arXiv.2408.16800](https://doi.org/10.48550/arXiv.2408.16800)
