import os
import shutil

# Paths
source_folder = "../../Data/Yolo_Labels/labels"
destination_folder = "../../Data/Val/labels"

image_names = [
    "Image_1", "Image_10", "Image_16", "Image_17", "Image_18",
    "Image_19", "Image_26", "Image_28", "Image_31", "Image_35",
    "Image_42", "Image_43", "Image_44", "Image_45", "Image_48",
    "Image_49", "Image_50", "Image_51", "Image_52", "Image_54",
    "Image_55", "Image_6", "Image_60", "Image_63", "Image_65",
    "Image_76", "Image_79", "Image_80", "Image_95", "Image_97",
    "Image_98"
]

os.makedirs(destination_folder, exist_ok=True)

for name in image_names:
    txt_file = f"{name}.txt"
    src_path = os.path.join(source_folder, txt_file)
    dst_path = os.path.join(destination_folder, txt_file)

    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"Moved: {txt_file}")
    else:
        print(f"File not found: {txt_file}")

print("Selected label files have been moved.")
