import json
import numpy as np
import cv2
import os

label_map = {
    "Road": 1,
    "Tree": 2,
    "Grass/Shrubs": 3,
    "Building": 4,
    "Water": 5,
    "Bare Soil": 6,
}

def json_to_mask(json_file, mask_size=(512, 512)):
    """Convert a LabelMe JSON annotation to a pixel mask."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    mask = np.zeros(mask_size, dtype=np.uint8)

    for shape in data["shapes"]:
        label = shape["label"]
        points = np.array(shape["points"], dtype=np.int32)

        if label in label_map:
            cv2.fillPoly(mask, [points], label_map[label])

    return mask

def process_directory(input_dir, output_dir, mask_size=(512, 512)):
    """Read all JSON files from input directory and save masks to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        mask = json_to_mask(json_path, mask_size)
        
        mask_filename = os.path.join(output_dir, json_file.replace(".json", ".png"))
        cv2.imwrite(mask_filename, mask)
        print(f"Saved mask: {mask_filename}")

input_directory = "../../Data/JSON/Full_Data"  
output_directory = "../../Data/Masks/Full_Data"

process_directory(input_directory, output_directory)
