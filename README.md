# 📌 Coffee Bean Maturity Classification using CNN
## Klasifikasi Tingkat Kematangan Biji Kopi Arabika berdasarkan Parameter Warna (RGB dan HSV) menggunakan CNN

This project aims to classify the maturity level of Arabica coffee beans using a **Convolutional Neural Network (CNN)**. The classification is based on **RGB and HSV color parameters**, following the Agtron coffee color scale.

## 📂 Dataset
- **Total images**: 1200 segmented coffee bean images  
- **Classes**:
  - **Light Roast** (400 images)
  - **Medium Roast** (400 images)
  - **Dark Roast** (400 images)
- **Preprocessing**:
  - Image size: **256x256 pixels**
  - Normalization: **RGB (0-255) → (0-1)**
  - Conversion to **HSV** format
  - Image segmentation using **Canny Edge Detection & Contour Detection**

## 🛠Technologies Used
- **Python** (TensorFlow, Keras, OpenCV)
- **CNN Model**:
  - 5 Convolutional layers with **Batch Normalization & Dropout**
  - **Adam optimizer** (Learning rate = 0.00001)
  - Training for **100 epochs** with early stopping
- **Evaluation Metrics**:
  - **Accuracy**
  - **Precision, Recall, and F1-Score**
  - **Confusion Matrix**

## 📜 Citation
```bibtex
@thesis{Tataming_2025,
  author      = {Tataming, Amadeus Gavriel},
  year        = {2025},
  title       = {Klasifikasi Tingkat Kematangan Biji Kopi Arabika berdasarkan Parameter Warna (RGB dan HSV) menggunakan CNN},
  institution = {Universitas Airlangga},
  type        = {Undergraduate thesis}
}
```
## 📩 Contact
📧 **Email**: amadeustataming@gmail.com
🔗 **LinkedIn**: [Amadeus Gavriel Tataming](https://linkedin.com/in/amadeusgaviel)


