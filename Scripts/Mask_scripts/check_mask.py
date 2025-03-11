import cv2
import numpy as np
import matplotlib.pyplot as plt

CLASS_COLORS = {
    #Format: BLUE | GREEN | RED 
    1: (19, 69, 139),     # Road 
    2: (240, 32, 160),      # Building
    3: (0, 255, 0),     # Vegetation
    4: (0, 255, 255),       # Material 
    5: (255, 0, 0),       # Water 
    0: (0, 0, 0),           # Unassigned 
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

    # Display the original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Display of mask image (color-coded mask)
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
    plt.title('Color Mapped Mask')
    plt.axis('off')

    # Show the images
    plt.show()

# Example usage:
image_path = '../../Data/Full_Data/Image_1.jpg'   # Path to the input image
mask_path = '../../Masks/Mask_1.png'         # Path to the corresponding mask

# Call the function
display_images(image_path, mask_path)
