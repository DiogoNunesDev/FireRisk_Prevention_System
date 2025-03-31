import os
import json

folder_path = "../../Data/JSON/Full_Data"  

labels_to_remove = ['Unknown', 'Car']

for filename in os.listdir(folder_path):
  if filename.endswith('.json'):
    file_path = os.path.join(folder_path, filename)

    with open(file_path, 'r') as f:
      data = json.load(f)

    original_len = len(data.get("shapes", []))
    data["shapes"] = [shape for shape in data.get("shapes", []) if shape["label"] not in labels_to_remove]
    new_len = len(data["shapes"])

    if original_len != new_len:
      with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
        print(f"Updated: {filename} — removed {original_len - new_len} shape(s)")

print("Done cleaning all files.")
