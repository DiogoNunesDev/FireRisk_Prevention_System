import cv2
import numpy as np
import matplotlib.pyplot as plt

CLASS_COLORS = {
    1: (255, 0, 0),         # Road (Red)
    2: (0, 255, 0),         # Building (Green)
    3: (0, 0, 255),         # Vegetation (Blue)
    4: (0, 255, 255),       # Material (Yellow)
    5: (255, 255, 0),       # Water (Cyan)
    0: (0, 0, 0),           # Unassigned (Black)
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
image_path = '../../Data/Full/Image_35.jpg'   # Path to the input image
mask_path = '../../Masks/Mask_35.png'         # Path to the corresponding mask

# Call the function
display_images(image_path, mask_path)
