import os
from PIL import Image

input_folder = "../Data/Data_Horizontally_Flipped"
#input_folder = "../Data/Original"
#output_folder = "..\\Data\\Data_Vertically_Flipped\\"
output_folder = "../Data/Data_H_V_Flipped"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.avif', '.JPG')):
        input_image_path = os.path.join(input_folder, filename)
        filename = filename.split('.')[0] + '_V.jpg'
        output_image_path = os.path.join(output_folder, filename)  
        
        original_image = Image.open(input_image_path)
        
        flipped_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)
        
        flipped_image.save(output_image_path)
        
        print(f'Flipped and saved: {output_image_path}')
