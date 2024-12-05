CLASS_MAP = {
    "road": 0,
    "building": 1,
    "vegetation": 2,
    "material": 3,
    "water": 4,
}

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

# Paths
model_path = "unet_final_model.h5" 
image_path = "../Data/Full/Image_43.jpg" 

input_shape = (512, 896, 3)
alpha = 0.3

# Updated class colors to match your specification
class_colors = [
    (0, 0, 0),       # Road: black
    (255, 0, 0),     # Building: red
    (0, 255, 0),     # Vegetation: green
    (255, 255, 0),   # Material: yellow
    (0, 255, 255),   # Water: cyan
]  

model = load_model(model_path, compile=False)

image = cv2.imread(image_path)
original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
image = cv2.resize(image, (input_shape[1], input_shape[0])) 
image = image / 255.0  
image = np.expand_dims(image, axis=0)  

pred = model.predict(image)[0] 
pred_mask = np.argmax(pred, axis=-1)  # Getting the class with the highest probability

# Creating the overlay
overlay = np.zeros_like(original_image, dtype=np.uint8)
for class_idx, color in enumerate(class_colors):
    overlay[pred_mask == class_idx] = color  

# Blending the original image with the overlay
blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(blended)
plt.axis("off")

plt.tight_layout()
plt.show()
