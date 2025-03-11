import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

images_path = "./Data/Full_Data/Image_46.jpg"
labels_path = "./Yolo Labels/Image_46.txt"

class_names = ["Road", "Tree", "Grass/Shrubs", "Building", "Water", "Bare Soil"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

def visualize_yolo_annotation(image_file, label_file):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    h, w, _ = image.shape 

    with open(label_file, "r") as file:
        lines = file.readlines()

    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        points = np.array([float(x) for x in data[1:]]).reshape(-1, 2)
        
        points[:, 0] *= w
        points[:, 1] *= h
        points = points.astype(np.int32)

        cv2.polylines(image, [points], isClosed=True, color=colors[class_id], thickness=2)
        cv2.putText(image, class_names[class_id], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


if os.path.exists(images_path) and os.path.exists(labels_path):
    visualize_yolo_annotation(images_path, labels_path)
else:
    print("Sample image or label file not found! Check your dataset paths.")
