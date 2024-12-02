import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Paths
model_path = "unet_final_model.h5"  # Adjust as needed
image_path = "../Data/Full/Image_163.png"  # Replace with your test image path

# Constants
input_shape = (512, 896, 3)
alpha = 0.5  # Transparency for overlay
class_colors = [
    (0, 0, 0),       # Background (black)
    (255, 0, 0),     # Class 1 (red)
    (0, 255, 0),     # Class 2 (green)
    (0, 0, 255),     # Class 3 (blue)
    (255, 255, 0),   # Class 4 (yellow)
]  # Add colors for all classes in your dataset

# Load the model
model = load_model(model_path, compile=False)

# Load and preprocess the image
image = cv2.imread(image_path)
original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
image = cv2.resize(image, (input_shape[1], input_shape[0]))  # Resize to match model input
image = image / 255.0  # Normalize
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Predict
pred = model.predict(image)[0]  # Remove batch dimension
pred_mask = np.argmax(pred, axis=-1)  # Convert probabilities to class indices

# Create the overlay
overlay = np.zeros_like(original_image, dtype=np.uint8)
for class_idx, color in enumerate(class_colors):
    overlay[pred_mask == class_idx] = color

# Blend the original image with the overlay
blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)

# Plot the results
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
