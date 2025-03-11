from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define class colors for YOLO classes
CLASS_COLORS = {
    0: (128, 0, 128),    # Road: Purple
    1: (0, 255, 0),      # Tree: Green
    2: (34, 139, 34),    # Grass/Shrubs: Dark Green
    3: (255, 0, 0),      # Building: Red
    4: (0, 255, 255),    # Water: Cyan
    5: (139, 69, 19),    # Bare Soil: Brown
}

# YOLO class names
CLASS_NAMES = ["Road", "Tree", "Grass/Shrubs", "Building", "Water", "Bare Soil"]

# Load YOLO model
model = YOLO("runs/segment/train/weights/best.pt")

# Load the original image
image_path = "./Data/Images/Resized_Images/Image_1.jpg"
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
orig_h, orig_w, _ = original_image.shape  # Get original image size

# Run YOLO inference
results = model(image_path, save=False)

# Create an empty overlay with the same size as the original image
overlay = np.zeros_like(original_image, dtype=np.uint8)

# Extract segmentation masks from results
for result in results:
    masks = result.masks  # Get mask data
    classes = result.boxes.cls.cpu().numpy().astype(int)  # Get class labels

    if masks is not None:
        mask_tensor = masks.data.cpu().numpy()  # Convert to numpy array
        
        for i, mask in enumerate(mask_tensor):
            class_id = classes[i]  # Get the predicted class
            
            if class_id in CLASS_COLORS:
                color = CLASS_COLORS[class_id]  # Get class-specific color
            else:
                color = (255, 255, 255)  # Default to white if class is unknown

            # Resize mask to original image size
            mask_resized = cv2.resize(mask, (orig_w, orig_h))
            mask_resized = (mask_resized > 0.5).astype(np.uint8)  # Threshold mask

            # Apply the color to the mask
            for c in range(3):  # Apply to RGB channels
                overlay[:, :, c] = np.where(mask_resized == 1, color[c], overlay[:, :, c])

# Blend the original image with the overlay
alpha = 0.6  # Transparency factor
blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)

# Save the final overlay image
cv2.imwrite("segmentation_overlay.jpg", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

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
