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

NUM_CLASSES = len(CLASS_MAP) 

def calculate_class_weights(masks_path):
    class_pixel_count = np.zeros(NUM_CLASSES + 1, dtype=np.float64) 
    total_pixels = 0  
    
    mask_files = [os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.endswith('.png')]
    print(f"Found {len(mask_files)} masks in {masks_path}")
    
    for mask_file in mask_files:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load {mask_file}")
            continue
        
        for cls_id in range(NUM_CLASSES + 1):  
            class_pixel_count[cls_id] += np.sum(mask == cls_id)
        
        total_pixels += mask.size
    

    class_percentages = (class_pixel_count[1:] / total_pixels) * 100  
    
    class_weights = total_pixels / (NUM_CLASSES * class_pixel_count[1:])  
    normalized_weights = class_weights / np.sum(class_weights)  
    
    print("\nClass Distribution:")
    for cls_name, cls_id in CLASS_MAP.items():
        print(f"  {cls_name.capitalize()}: {class_percentages[cls_id - 1]:.2f}%") 
    
    print("\nClass Weights:")
    for cls_name, cls_id in CLASS_MAP.items():
        print(f"  {cls_name.capitalize()}: {normalized_weights[cls_id - 1]:.4f}")  
    
    return class_percentages, normalized_weights

masks_path = '../Masks' 

class_percentages, class_weights = calculate_class_weights(masks_path)
