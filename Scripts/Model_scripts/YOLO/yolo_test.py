from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Define class colors for YOLO classes
CLASS_COLORS = {
    0: (128, 0, 128),    # Road: Purple
    1: (0, 255, 0),      # Tree: Green
    2: (0, 0, 255),      # Grass/Shrubs: Blue
    3: (255, 0, 0),      # Building: Red
    4: (0, 255, 255),    # Water: Cyan
    5: (255, 255, 0),    # Bare Soil: Yellow
    6: (255, 255, 255)   # Unknown (Background): White
}

# YOLO class names
CLASS_NAMES = ["Road", "Tree", "Grass/Shrubs", "Building", "Water", "Bare Soil", "Car", "Unknown"]

# Load YOLO model
model = YOLO("../../../runs/segment/train/weights/best.pt")

# Function to load image using PIL and convert it to RGB
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB format
    return np.array(image)  # Convert to NumPy array

# Define image path
image_path = "../../../Data/Val/images/Image_10.jpg"
original_image = load_image(image_path)

# Run YOLO inference **without resizing**
results = model(original_image, save=False)

# Create an empty overlay with the same size as the original image
overlay = np.zeros_like(original_image, dtype=np.uint8)

# Extract segmentation masks from results
for result in results:
    masks = result.masks  # Get mask data
    classes = result.boxes.cls.cpu().numpy().astype(int)  # Get class labels
    scores = result.boxes.conf.cpu().numpy()  # Get confidence scores

    if masks is not None:
        mask_tensor = masks.data.cpu().numpy()  # Convert to numpy array
        
        # Create an empty class mask with the same dimensions as the image
        class_mask = np.full((original_image.shape[0], original_image.shape[1]), -1, dtype=int)
        
        for i, mask in enumerate(mask_tensor):
            class_id = classes[i]  # Get the predicted class
            confidence = scores[i]  # Get the confidence score
            color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Default to white if class not found
            
            # Resize mask to match original image dimensions
            mask_resized = np.array(Image.fromarray(mask).resize((original_image.shape[1], original_image.shape[0])))
            mask_resized = (mask_resized > 0.5).astype(np.uint8)
            
            # Assign class with highest confidence to each pixel
            mask_indices = np.where(mask_resized == 1)
            for y, x in zip(mask_indices[0], mask_indices[1]):
                if class_mask[y, x] == -1 or confidence > scores[class_mask[y, x]]:
                    class_mask[y, x] = i
                    overlay[y, x] = color

# Blend the original image with the overlay
alpha = 0.6  # Transparency factor
blended = (original_image * (1 - alpha) + overlay * alpha).astype(np.uint8)

# Save the final overlay image
Image.fromarray(blended).save("segmentation_overlay.jpg")

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Overlay")
plt.imshow(blended)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Segmentation Masks Only")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()

print("Segmentation overlay saved as segmentation_overlay.jpg")
