# SkyBlaze: AI-Powered Wildfire Risk Analysis 🌲🔥

**SkyBlaze** is an AI-driven project designed to analyze aerial images and assess wildfire risk in properties.  
It leverages **state-of-the-art semantic segmentation** models to detect critical landscape features that influence fire spread.

---

## Features

- **Semantic Segmentation** of aerial images into 6 key land-cover classes.  
- **Deep Learning Model**: [DeepLabV3+](https://arxiv.org/abs/1802.02611) with a **ResNet101 backbone**, implemented in **PyTorch**.  
- **Custom Labeled Dataset**: 41 aerial images annotated with **LabelMe** for pixel-level precision.  
- **Scalable Approach**: Designed to move from drone-based imagery to satellite sources, with potential integration of LiDAR and multispectral sensors.  

---

## Classes & Color Coding

The model segments images into the following classes:

1. **Road** – (0, 0, 255) 🔴  
2. **Tree** – (0, 255, 0) 🟢  
3. **Grass/Shrubs** – (144, 238, 144) 🌿  
4. **Building** – (125, 0, 125) 🟣  
5. **Water** – (255, 0, 0) 🔵  
6. **Bare Soil** – (19, 69, 139) 🟤  

---

## Results Showcase

### Input Aerial Image
![Input Example](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/Test/image.png)

### Predicted Segmentation
![Output Example](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/Test/output.jpg)

---

## The AI Model: DeepLabV3+ with ResNet101

SkyBlaze uses **DeepLabV3+**, a leading architecture for semantic segmentation, with **ResNet101** as the feature extractor.

### Why DeepLabV3+?
- **Atrous Convolutions**: Captures multi-scale context.  
- **Encoder-Decoder Design**: Refines segmentation boundaries.  
- **Robust Backbone**: ResNet101 ensures strong feature representation.  

---

## The Dataset

- **102 labeled aerial images** from properties.  
- Annotated with **LabelMe** to create pixel-wise segmentation masks.  
- Data Augmentation applied: rotations and flips.  

### Example

**Raw Aerial Image**  
![Raw Image](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/readme/Original%20Image.png)  

**Labeled Segmentation Mask**  
![Labeled Image](https://github.com/DiogoNunesDev/FireRisk_Prevention_System/blob/main/readme/Mask%20Labeled%20Image.png)  

---

## Training Process

1. **Preprocessing**  
   - Images resized to **512×512** for consistency.  
   - Normalization applied for better convergence.  

2. **Model Training**  
   - Framework: **PyTorch**  
   - Optimizer: **Adam** (lr = 0.0001)  
   - Loss Function: **Cross-Entropy + Dice Loss** (to handle class imbalance)  

---

## Evaluation Metrics

To measure performance, the model was evaluated using:

- **IoU (Intersection over Union)**: Overlap between predicted and ground-truth masks.  
- **Dice Coefficient**: Similarity between predicted and actual regions.  
- **Pixel Accuracy**: Percentage of correctly predicted pixels.  

---

## Future Direction

- **Bigger Dataset**: Expand training data with drone & satellite imagery.  
- **Vegetation Health Analysis**: Integrate dryness & fuel load indicators for risk prediction.  
- **Regional Deployment**: Scale to municipalities and insurance companies for wildfire prevention planning.  

---
