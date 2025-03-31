import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define class colors based on your label mapping
CLASS_COLORS = {
    0: (0, 0, 0),        # Unassigned pixels (Black)
    1: (0, 0, 255),      # Road (Red)
    2: (0, 255, 0),      # Tree (Green)
    3: (144, 238, 144),  # Grass/Shrubs (Light Green)
    4: (125, 0, 125),    # Building (Purple)
    5: (0, 0, 255),      # Water (Blue)
    6: (139, 69, 19),    # Bare Soil (Brown)
    7: (169, 169, 169),  # Car (Gray)
    8: (255, 255, 255)   # Unknown (White)
}

def create_colored_mask(mask):
    """Convert a grayscale mask to a color-coded visualization."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color

    return color_mask

def display_images(image_path, mask_path):
    """Display an image with its corresponding color-coded mask."""
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

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
    plt.title('Classified Mask')
    plt.axis('off')

    plt.show()

# Interactive loop to inspect images
while True:
    try:
        image_number = input("Enter the image number to inspect (or 'q' to quit): ")

        if image_number.lower() == 'q':
            print("Exiting the program.")
            break

        image_number = int(image_number)

        # Adjust paths based on directory structure
        image_path = f'../../Data/Images/Full_Data/Image_{image_number}.jpg'   
        mask_path = f'../../Data/Masks/Full_Data/Image_{image_number}.png'         

        display_images(image_path, mask_path)

    except ValueError:
        print("Invalid input. Please enter a valid image number or 'q' to quit.")
