import os
import cv2
import numpy as np

# CLASS_MAP with indexing from 1 to 5
CLASS_MAP = {
    "road": 1,
    "building": 2,
    "vegetation": 3,
    "material": 4,
    "water": 5
}

NUM_CLASSES = len(CLASS_MAP)  # We are using indices from 1 to 5

# Function to calculate class percentages
def calculate_class_weights(masks_path):
    # Initialize the count array for each class, plus a slot for background (index 0)
    class_pixel_count = np.zeros(NUM_CLASSES + 1, dtype=np.float64)  # Store pixel count for each class (including background)
    total_pixels = 0  # Total number of pixels in the dataset
    
    # Iterate over mask files
    mask_files = [os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.endswith('.png')]
    print(f"Found {len(mask_files)} masks in {masks_path}")
    
    for mask_file in mask_files:
        # Load mask as grayscale
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load {mask_file}")
            continue
        
        # Count pixels per class
        for cls_id in range(NUM_CLASSES + 1):  # We check for class 0 (background) too
            class_pixel_count[cls_id] += np.sum(mask == cls_id)
        
        # Update total pixels
        total_pixels += mask.size
    
    # Calculate class percentages (excluding background class from display)
    class_percentages = (class_pixel_count[1:] / total_pixels) * 100  # Skip index 0
    
    # Compute weights (inverse proportional to class frequency)
    class_weights = total_pixels / (NUM_CLASSES * class_pixel_count[1:])  # Skip index 0
    normalized_weights = class_weights / np.sum(class_weights)  # Optional normalization
    
    # Display results
    print("\nClass Distribution:")
    for cls_name, cls_id in CLASS_MAP.items():
        print(f"  {cls_name.capitalize()}: {class_percentages[cls_id - 1]:.2f}%")  # Adjust for 0-based indexing in class_percentages
    
    print("\nClass Weights:")
    for cls_name, cls_id in CLASS_MAP.items():
        print(f"  {cls_name.capitalize()}: {normalized_weights[cls_id - 1]:.4f}")  # Adjust for 0-based indexing in class_weights
    
    return class_percentages, normalized_weights

# Path to the masks folder
masks_path = '../Masks'  # Adjust this path to your actual folder

# Run the function
class_percentages, class_weights = calculate_class_weights(masks_path)
