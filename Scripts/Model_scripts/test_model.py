import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

CLASS_MAP = {
    "Road": 0, "Tree": 1, "Grass/Shrubs": 2, "Building": 3, "Water": 4, "Bare Soil": 5
}

# Paths
model_path = "../unet_final_model.h5"
image_path = "../../Test/test_2.jpg"

input_shape = (512, 896, 3)
alpha = 0.9

# Updated class colors to match your specification
class_colors = {
    0: (0, 0, 255),        # Road
    1: (0, 255, 0),        # Tree
    2: (255, 255, 0),      # Grass/Shrubs
    3: (125, 0, 125),      # Building
    4: (255, 0, 0),        # Water
    5: (0, 255, 255),      # Bare Soil
}

model = load_model(model_path, compile=False)

image = cv2.imread(image_path)

# Resize the image to match the model's input size
original_image = cv2.resize(image, (input_shape[1], input_shape[0])) 
image = cv2.resize(image, (input_shape[1], input_shape[0])) 
image = image / 255.0  # Normalize
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make prediction
pred = model.predict(image)[0]
pred_mask = np.argmax(pred, axis=-1)  # Get the class with the highest probability

# Create an empty overlay image
overlay = np.zeros_like(original_image, dtype=np.uint8)

# Assign colors to the overlay image based on predicted mask
for class_idx, color in class_colors.items():
    overlay[pred_mask == class_idx] = color

# Blend the original image with the overlay using the alpha value
blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)

# Save the output image
cv2.imwrite("../../Test/output.jpg", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

# Display the results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis("off")

# Predicted Mask
plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")

# Overlayed Image
plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(blended)
plt.axis("off")

plt.tight_layout()
plt.show()
