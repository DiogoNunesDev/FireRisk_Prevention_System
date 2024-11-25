import json
import numpy as np
import cv2
import os
from tqdm import tqdm

CLASS_MAP = {
    "water": 0,
    "road": 1,
    "building": 2,
    "vegetation": 3,
    "material": 4
}

def create_mask_from_json(json_path, image_shape, output_path):

    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for shape in data['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)

        class_value = CLASS_MAP.get(label)
        if class_value:
            cv2.fillPoly(mask, [points], class_value)

    cv2.imwrite(output_path, mask)

def process_all_jsons(json_folder, output_folder, image_shape):

    os.makedirs(output_folder, exist_ok=True)
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    json_files.sort()

    mask_counter = 1

    for json_file in tqdm(json_files, desc="Processing JSON files"):
        json_path = os.path.join(json_folder, json_file)
        mask_name = f"Mask_{mask_counter}.png"
        output_path = os.path.join(output_folder, mask_name)

        create_mask_from_json(json_path, image_shape, output_path)

        mask_counter += 1

json_folder = "../Labels/Full" 
output_folder = "../Masks"  
image_shape = (512, 896)  


process_all_jsons(json_folder, output_folder, image_shape)
