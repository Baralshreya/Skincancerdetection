# 🩺 Skin Cancer Detection using EfficientNet-B3

This project focuses on **automated skin cancer classification** using **deep learning**, specifically the **EfficientNet-B3** architecture.  
The model is trained and fine-tuned on the **HAM10000** dataset to classify **seven types of skin lesions**.  
Our approach achieves a validation accuracy of **92.52%**, demonstrating strong potential for assisting dermatologists in early diagnosis of skin cancer.

---

## 📖 Overview

Skin cancer is one of the most common and deadly forms of cancer worldwide. Early detection significantly increases survival rates, but manual diagnosis is often time-consuming and subjective.  
This project uses **transfer learning, Fine tuning** and **deep convolutional neural networks (CNNs)** to build an efficient, accurate, and scalable skin cancer detection model.

---

## 🎯 Objectives

- To develop a deep learning model for **multi-class classification** of dermoscopic images.
- Pre processing of Uneven dataset of **HAM10000 dataset**. 
- To fine-tune a **pre-trained EfficientNet-B3** model for the **HAM10000 dataset**.  
- To evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix.

---

## 🧠 Dataset

**Dataset:** [HAM10000 – Human Against Machine with 10,000 Training Images](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

| Class | Description |
|-------|--------------|
| akiec | Actinic keratoses and intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

**Preprocessing Steps:**
- Image resizing to 224×224  
- Normalization (0–1 range)  
- Data augmentation: rotation, flip, zoom, brightness adjustment  
- Label encoding (one-hot)

---

## ⚙️ Model Architecture

**Base model:** EfficientNet-B3 (pretrained on ImageNet)  
**Framework:** TensorFlow / Keras  

Layers added:
- GlobalAveragePooling2D  
- Dropout (0.3)  
- Dense (512, ReLU)  
- Output (7, Softmax)

**Loss Function:** Categorical Crossentropy  
**Optimizer:** Adam (lr = 1e-4)  
**Batch Size:** 32  
**Epochs:** 50  

---

## 📊 Results

| Metric | Validation Set |
|---------|----------------|
| Accuracy | **92.52%** |
| Precision | 0.91 |
| Recall | 0.90 |
| F1-Score | 0.90 |

**Confusion Matrix and ROC curves** are provided in the results section of the notebook.

---

## 🖼️ Sample Predictions

| Image | True Label | Predicted Label |
|-------|-------------|-----------------|
| ![Example1](results/sample1.png) | Melanoma | Melanoma |
| ![Example2](results/sample2.png) | Nevus | Nevus |
| ![Example3](results/sample3.png) | BCC | BCC |

---

## 💻 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/SkincancerDetection.git
cd SkincancerDetection
pip install -r requirements.txt
