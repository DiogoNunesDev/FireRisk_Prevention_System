import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from unet import UNet  # Importa o teu UNet em PyTorch
from deeplabv3_plus import DeepLabV3Plus
from hrnet import HRNetSegmentation


model_path = "./deeplab_best_iou_model.pth"
#model_path = "./deeplab_final_model.pth"
image_path = "../../Test/image.png"
image_path = "C:\\Users\\diogo\\OneDrive\\Ambiente de Trabalho\\SkyBlaze\\readme\\Original Image.png"
output_path = "../../Test/output.jpg"
mask_output_path = "../../Test/pred_mask.png"

input_shape = (512, 512)
alpha = 0.8
n_labels = 7 

class_colors = {
    0: (0, 0, 0),        # Unassigned pixels (Black)
    1: (255, 0, 0),      # Road (Red)
    2: (0, 255, 0),      # Tree (Green)
    3: (144, 238, 144),  # Grass/Shrubs (Light Green)
    4: (125, 0, 125),    # Building (Purple)
    5: (0, 0, 255),      # Water (Blue) 
    6: (19, 69, 139),    # Bare Soil (Brown)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = UNet(in_channels=3, num_classes=n_labels)
model = DeepLabV3Plus(num_classes=n_labels).to(device)
#model = HRNetSegmentation(num_classes=n_labels, pretrained=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

image = cv2.imread(image_path)
original_image = cv2.resize(image, (input_shape[1], input_shape[0]))
image_resized = original_image.astype(np.float32) / 255.0
image_tensor = torch.tensor(np.transpose(image_resized, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

colored_mask = np.zeros_like(original_image, dtype=np.uint8)
for class_idx, color in class_colors.items():
    colored_mask[pred_mask == class_idx] = color

blended = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)

cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
cv2.imwrite(mask_output_path, colored_mask)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
