import os
import shutil

def move_selected_label_files(source_dir, target_dir, selected_files):
    """Move specified YOLO label files from source directory to target directory."""
    os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists
    
    for label_file in selected_files:
        src_path = os.path.join(source_dir, label_file)
        dst_path = os.path.join(target_dir, label_file)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"Moved: {label_file} -> {target_dir}")
        else:
            print(f"File not found: {label_file}")

source_directory = "../../Data/Yolo_Labels/labels"  
target_directory = "../../Data/Val/labels"  

selected_files = [
    "Image_1.txt", "Image_6.txt", "Image_10.txt", "Image_16.txt", "Image_17.txt", "Image_18.txt", "Image_19.txt",
    "Image_26.txt", "Image_28.txt", "Image_31.txt", "Image_35.txt", "Image_42.txt", "Image_43.txt", "Image_44.txt", 
    "Image_45.txt", "Image_48.txt", "Image_49.txt", "Image_50.txt", "Image_51.txt", "Image_52.txt", "Image_54.txt", 
    "Image_55.txt", "Image_60.txt", "Image_63.txt", "Image_65.txt", "Image_76.txt", "Image_79.txt", "Image_80.txt", 
    "Image_95.txt", "Image_97.txt", "Image_98.txt"
]

# Move selected label files
move_selected_label_files(source_directory, target_directory, selected_files)

print("Selected label files moved successfully.")
