# SkyBlaze: AI-Powered Wildfire Risk Analysis 🌲🔥

**SkyBlaze** is an AI-driven project designed to analyze aerial images and identify fire risk in properties. This project demonstrates the application of advanced deep learning techniques for wildfire prevention and management. 

---

## 🌟 Features

- **Input-Output Showcase**: Visual results of wildfire risk analysis from aerial images.
- **Deep Learning Model**: Powered by a U-Net architecture tailored for segmentation tasks.
- **Custom Labeled Dataset**: Created with precision using the LabelMe annotation tool.
- **Future Direction**: Expanding to analyze vegetation dryness for more accurate risk predictions.

---

## 🔎 Results Showcase

### Aerial Image Input
![Input Example](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/Test/test.jpg)

### Prediction
![Output Example](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/Test/output.jpg)

- **Red**: Building Class
- **Green**: Vegetation Class
- **Purple**: Road Class

---

## 🧠 The AI Model: U-Net

The U-Net model was chosen for its exceptional performance in image segmentation tasks. Here's a high-level view of the architecture:

![U-Net Model Architecture](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/readme/UNet%20Architecture.jpg)

### Why U-Net?

- **Precision**: Highly accurate in segmenting risk zones from aerial images.
- **Efficiency**: Optimized for handling high-resolution input data.

---

## 🖼️ The Dataset

The dataset consists of 41 labeled aerial images, annotated using **LabelMe** for precise segmentation. Here is a example:

### Classes
- There are 5 Classes: Building, Vegetation, Water, Road and Material (Fire prone materials)

### Raw Aerial Image
![Raw Image Example](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/readme/Original%20Image.png)

### Labeled Data
![Labeled Image Example](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/readme/Mask%20Labeled%20Image.png)

The dataset enables the model to learn fine-grained distinctions between the various components the compose a property.

   - Applied data augmentation techniques (e.g., rotations, flips) to increase image count.
   - Converted Labeled Images into Masks for model Training.


---

## 🏋️‍♂️ Training Process

The U-Net model was trained on the labeled dataset following a structured process:

1. **Data Preprocessing**:
   - Resized images to [dimensions] for consistency.

2. **Model Training**:
   - Framework: [TensorFlow].
   - Optimizer: [Adam] with an initial learning rate of [0.0001].
   - Loss Function: Dice loss for better handling of imbalanced data.

## 📊 Evaluation Metrics

To ensure reliable predictions, the U-Net model was evaluated using several metrics:

1. **IoU (Intersection over Union)**: Measures the overlap between the predicted and ground-truth segmentation masks.
2. **Dice Coefficient**: Quantifies the similarity between predicted masks and the actual regions.
3. **Pixel Accuracy**: Tracks the percentage of correctly predicted pixels.

---

## 🚀 Future Direction

SkyBlaze will continue to grow with the following planned improvements:

- **Vegetation Dryness Analysis**: Integrating additional data sources to assess vegetation dryness and predict wildfire risk more accurately.
- **Scalability**: Deploying the system for use across diverse geographical regions.


