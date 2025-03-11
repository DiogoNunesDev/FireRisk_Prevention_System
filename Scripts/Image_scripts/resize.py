import os
from PIL import Image

input_folder = "../../New Data"
output_folder = "../../New Data"  

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.avif', '.JPG')):
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename)  
        
        original_image = Image.open(input_image_path)
        resized_image = original_image.resize((896, 512))
        
        resized_image.save(output_image_path)
        
        print(f'Processed and saved: {output_image_path}')
