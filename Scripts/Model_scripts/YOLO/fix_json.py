import json
import os

# Directory where JSON files are stored
json_files = [
    "./okok/Original/Image_54.json",
    "./okok/Original/Image_55.json",
    "./okok/Original/Image_56.json",
    "./okok/Original/Image_57.json",
    "./okok/Original/Image_58.json",
    "./okok/Original/Image_59.json",
    "./okok/Original/Image_60.json",
    "./okok/Original/Image_61.json",
    "./okok/Original/Image_62.json",
    "./okok/Original/Image_63.json",
    "./okok/Original/Image_64.json",
    "./okok/Original/Image_65.json",
    "./okok/Original/Image_66.json",
    "./okok/Original/Image_67.json",
    "./okok/Original/Image_68.json",
]

def merge_labels(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    
    for shape in data["shapes"]:
        if shape["label"] in ["Buiding"]:
            shape["label"] = "Building"
    
    # Save the modified JSON file
    output_file = json_file
    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)
    
    print(f"Processed: {json_file} -> {output_file}")

# Process each JSON file
for json_file in json_files:
    merge_labels(json_file)

print("All JSON files have been updated with merged labels.")
