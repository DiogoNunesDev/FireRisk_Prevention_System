import os
import json

#input_folder = "../Labels/Original"
#input_folder = "../Labels/Horizontal_Flip"
#output_folder = "../Labels/Horizontal_Flip"
output_folder = "../Labels/H_V_Flip"
#output_folder = "../Labels/Vertical_Flip"

os.makedirs(output_folder, exist_ok=True)

image_width = 1920  
image_height = 1080  

def flip_horizontal(points, image_width):
    return [[image_width - x, y] for x, y in points]

def flip_vertical(points, image_height):
    return [[x, image_height - y] for x, y in points]

for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        input_json_path = os.path.join(input_folder, filename)
        filename = filename.split('.')[0] + '_H_V.json'
        output_json_path = os.path.join(output_folder, filename)  
        
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        
        for shape in data['shapes']:
            if 'points' in shape:
                
                shape['points'] = flip_horizontal(shape['points'], image_width)
                
                shape['points'] = flip_vertical(shape['points'], image_height)
        
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f'Flipped JSON saved: {output_json_path}')
