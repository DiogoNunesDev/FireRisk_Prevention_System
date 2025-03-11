import os
import json

input_folder = "../../Data/JSON/Original"  
output_folder = "../../Data/JSON/Full_Data"  

os.makedirs(output_folder, exist_ok=True)

image_width = 512  
image_height = 512  

def flip_horizontal(points, img_width):
    return [[img_width - x, y] for x, y in points]

def flip_vertical(points, img_height):
    return [[x, img_height - y] for x, y in points]

def flip_diagonal(points, img_width, img_height):
    return [[img_width - x, img_height - y] for x, y in points]

for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        input_json_path = os.path.join(input_folder, filename)
        base_filename = filename.split('.')[0]  

        with open(input_json_path, 'r') as f:
            data = json.load(f)

        for flip_type, flip_function in [
            ("H", lambda points: flip_horizontal(points, image_width)),
            ("V", lambda points: flip_vertical(points, image_height)),
            ("D", lambda points: flip_diagonal(points, image_width, image_height))
        ]:
            flipped_data = json.loads(json.dumps(data))  # Deep copy
            for shape in flipped_data['shapes']:
                if 'points' in shape:
                    shape['points'] = flip_function(shape['points'])

            output_json_path = os.path.join(output_folder, f"{base_filename}_{flip_type}.json")
            with open(output_json_path, 'w') as f:
                json.dump(flipped_data, f, indent=4)

            print(f"Flipped JSON saved: {output_json_path}")

print("All JSON transformations completed successfully!")
