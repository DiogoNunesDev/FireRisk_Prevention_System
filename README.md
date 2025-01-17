# SkyBlaze: AI-Powered Wildfire Risk Analysis 🌲🔥

**SkyBlaze** is an AI-driven project designed to analyze aerial images and identify wildfire risk zones. This project demonstrates the application of advanced deep learning techniques for wildfire prevention and management. 

---

## 🌟 Features

- **Input-Output Showcase**: Visual results of wildfire risk analysis from aerial images.
- **Deep Learning Model**: Powered by a U-Net architecture tailored for segmentation tasks.
- **Custom Labeled Dataset**: Created with precision using the LabelMe annotation tool.
- **Future Direction**: Expanding to analyze vegetation dryness for more accurate risk predictions.

---

## 🔎 Results Showcase

### Aerial Image Input
![Input Example](https://via.placeholder.com/600x300)

### Predicted Risk Zone Output
![Output Example](https://via.placeholder.com/600x300)

---

## 🧠 The AI Model: U-Net

The U-Net model was chosen for its exceptional performance in image segmentation tasks. Here's a high-level view of the architecture:

![U-Net Model Architecture](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/UNet%20Architecture.jpg)

### Why U-Net?

- **Precision**: Highly accurate in segmenting risk zones from aerial images.
- **Efficiency**: Optimized for handling high-resolution input data.

---

## 🖼️ The Dataset

The dataset consists of labeled aerial images, annotated using **LabelMe** for precise segmentation. Here are some examples:

### Raw Aerial Image
![Raw Image Example](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/readme/Original%20Image.png)

### Labeled Data
![Labeled Image Example](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/readme/Mask%20Labeled%20Image.png)

The dataset enables the model to learn fine-grained distinctions between fire-prone and safe zones.

---

## 🏋️‍♂️ Training Process

The U-Net model was trained on the labeled dataset following a structured process:

1. **Data Preprocessing**:
   - Resized images to [dimensions] for consistency.
   - Applied data augmentation techniques (e.g., rotations, flips) to improve model robustness.

2. **Model Training**:
   - Framework: [TensorFlow].
   - Optimizer: [Adam] with an initial learning rate of [0.0001].
   - Loss Function: Dice loss for better handling of imbalanced data.

