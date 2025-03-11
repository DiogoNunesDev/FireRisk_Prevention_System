import json
import os

input_folder = "../../Data/JSON/Full_Data" 
output_folder = "../../Data/Yolo_Labels"  
os.makedirs(output_folder, exist_ok=True)

class_names = ["Road", "Tree", "Grass/Shrubs", "Building", "Water", "Bare Soil"]

for file_name in os.listdir(input_folder):
  if file_name.endswith(".json"):
    with open(os.path.join(input_folder, file_name), "r") as f:
        data = json.load(f)

    txt_file = os.path.join(output_folder, file_name.replace(".json", ".txt"))
    with open(txt_file, "w") as f:
      for shape in data["shapes"]:
        label = shape["label"]
        if label not in class_names:
            continue  
        
        class_id = class_names.index(label) 
        points = shape["points"]

        norm_points = [(x / 512, y / 512) for x, y in points]
        poly_string = " ".join([f"{x} {y}" for x, y in norm_points])

        f.write(f"{class_id} {poly_string}\n")
