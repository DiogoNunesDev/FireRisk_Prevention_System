import cv2
import numpy as np
import matplotlib.pyplot as plt

CLASS_COLORS = {
    0: (0, 0, 0),           # Unassigned pixels (Black)
    1: (0, 0, 255),         # Road (Red)
    2: (125, 0, 125),       # Building (Purple)
    3: (0, 255, 0),         # Vegetation (Green)
    4: (0, 255, 255),       # Material (Yellow)
    5: (255, 255, 0),       # Water (Cyan)
}

def create_colored_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color

    return color_mask

def display_images(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Error: Couldn't load the image or mask.")
        return

    if image.shape[:2] != mask.shape:
        print("Error: Image and mask dimensions don't match.")
        return

    color_mask = create_colored_mask(mask)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
    plt.title('Classified Mask')
    plt.axis('off')

    
    plt.show()

# Scanner for user input
while True:
    try:
        
        image_number = input("Enter the image number to inspect (or 'q' to quit): ")

        if image_number.lower() == 'q':
            print("Exiting the program.")
            break

        image_number = int(image_number)

        image_path = f'../Data/Full/Image_{image_number}.jpg'   
        mask_path = f'../Masks/Mask_{image_number}.png'         


        display_images(image_path, mask_path)

    except ValueError:
        print("Invalid input. Please enter a valid image number or 'q' to quit.")