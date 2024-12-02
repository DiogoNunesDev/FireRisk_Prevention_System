import json
import numpy as np
import cv2
import os
from tqdm import tqdm

# Class mapping
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

    # Initialize an empty mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for shape in data['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)

        # Map the label to its class value
        class_value = CLASS_MAP.get(label)
        if class_value is not None:
            cv2.fillPoly(mask, [points], class_value)

    # Save the mask
    cv2.imwrite(output_path, mask)

def process_jsons_in_order(labels_order_path, json_folder, output_folder, image_shape):
    """
    Process JSON files in the order specified in labels_order.txt and create corresponding masks.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Read the order of labels from the file
    with open(labels_order_path, 'r') as f:
        labels_order = [line.strip() for line in f]

    # Initialize counter for mask filenames
    mask_counter = 1

    # Loop through the ordered labels and create masks
    for label_name in tqdm(labels_order, desc="Processing JSON files"):
        json_path = os.path.join(json_folder, label_name + '.json')

        # Verify the JSON file exists
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found - {json_path}")
            continue

        # Construct the output mask file name
        mask_name = f"Mask_{mask_counter}.png"
        output_path = os.path.join(output_folder, mask_name)

        # Create the mask
        create_mask_from_json(json_path, image_shape, output_path)

        # Increment the mask counter
        mask_counter += 1

# Paths and parameters
labels_order_path = "./labels_order.txt"  # Path to the file specifying the order
json_folder = "../Labels/Full"            # Folder containing the JSON files
output_folder = "../Masks"                # Folder where masks will be saved
image_shape = (512, 896)                  # Shape of the output masks

# Process JSON files and create masks
process_jsons_in_order(labels_order_path, json_folder, output_folder, image_shape)
