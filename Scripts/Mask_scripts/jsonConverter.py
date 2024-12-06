import json
import numpy as np
import cv2
import os
from tqdm import tqdm

CLASS_MAP = {
    "road": 1,
    "building": 2,
    "vegetation": 3,
    "material": 4,
    "water": 5,
}

def create_mask_from_json(json_path, image_shape, output_path):
    """
    Create a mask from a JSON file containing annotation shapes.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for shape in data['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)

        class_value = CLASS_MAP.get(label)
        if class_value is not None:
            cv2.fillPoly(mask, [points], class_value)

    cv2.imwrite(output_path, mask)

def process_jsons_in_order(labels_order_path, json_folder, output_folder, image_shape):
    """
    Process JSON files in the order specified in labels_order.txt and create corresponding masks.
    """
    os.makedirs(output_folder, exist_ok=True)

    with open(labels_order_path, 'r') as f:
        labels_order = [line.strip() for line in f]

    mask_counter = 1

    for label_name in tqdm(labels_order, desc="Processing JSON files"):
        json_path = os.path.join(json_folder, label_name + '.json')

        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found - {json_path}")
            continue

        mask_name = f"Mask_{mask_counter}.png"
        output_path = os.path.join(output_folder, mask_name)

        create_mask_from_json(json_path, image_shape, output_path)

        mask_counter += 1

labels_order_path = "./labels_order.txt"  
json_folder = "../Labels/Full"            
output_folder = "../Masks"                
image_shape = (512, 896)            
      
process_jsons_in_order(labels_order_path, json_folder, output_folder, image_shape)
