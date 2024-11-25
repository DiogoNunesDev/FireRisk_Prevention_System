import os
from tqdm import tqdm
import cv2

def rename_images(image_folder, output_folder):
    """
    Renames images in a folder to sequential filenames (Image_1, Image_2, etc.).

    Args:
        image_folder (str): Path to the folder containing the images.
        output_folder (str): Path to save the renamed images.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get a list of image files (extensions: .jpg, .png, etc.)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Sort images to maintain consistent order
    image_files.sort()
    
    # Sequential counter for naming images
    image_counter = 1

    for image_file in tqdm(image_files, desc="Renaming Images"):
        # Original image path
        image_path = os.path.join(image_folder, image_file)
        
        # New sequential name
        new_name = f"Image_{image_counter}.png"  # Save all as .png
        
        # Destination path
        output_path = os.path.join(output_folder, new_name)
        
        # Read the image and save with new name (to ensure consistent format)
        image = cv2.imread(image_path)
        cv2.imwrite(output_path, image)
        
        image_counter += 1

    print(f"Renamed and saved {image_counter - 1} images in {output_folder}")

# Parameters
image_folder = "../Data/Full"  # Replace with the folder containing original images
output_folder = "../Data/Full_2"  # Replace with the folder to save renamed images

# Rename images sequentially
rename_images(image_folder, output_folder)
