import cv2
import numpy as np
import os

label_map = {
    1: "Road",
    2: "Tree",
    3: "Grass/Shrubs",
    4: "Building",
    5: "Water",
    6: "Bare Soil",
    7: "Car",
    8: "Unknown"
}

def mask_to_yolo_segmentation(mask_path, output_txt_path):
    """Convert a segmentation mask into YOLO segmentation annotation format."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Error loading {mask_path}")
        return

    height, width = mask.shape
    unique_labels = np.unique(mask)

    with open(output_txt_path, "w") as f:
        for label in unique_labels:
            if label == 0 or label > 8:
                continue  # Skip background and out-of-range labels
            
            mask_binary = (mask == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3:
                    continue  # Ignore invalid contours
                
                points = []
                for point in contour:
                    x, y = point[0]
                    x_norm = x / width
                    y_norm = y / height
                    points.append(f"{x_norm:.6f} {y_norm:.6f}")
                
                annotation = f"{label - 1} " + " ".join(points) + "\n"
                f.write(annotation)

    print(f"Saved YOLO segmentation annotation: {output_txt_path}")

def process_masks(mask_dir, output_txt_dir):
    """Convert all mask images to YOLO segmentation annotation files."""
    os.makedirs(output_txt_dir, exist_ok=True)

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        txt_output_path = os.path.join(output_txt_dir, mask_file.replace(".png", ".txt"))
        mask_to_yolo_segmentation(mask_path, txt_output_path)

# Set directories
mask_directory = "../../Data/Masks/Full_Data"  # Replace with actual directory containing masks
yolo_output_directory = "../../Data/Yolo_Labels/labels"

# Convert mask images to YOLO segmentation format
process_masks(mask_directory, yolo_output_directory)

print("Conversion to YOLO segmentation format complete.")
