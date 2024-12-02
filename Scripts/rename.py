import os

# Path to the folder containing the files
directory = "../Data/Full"

# Path to your file_order.txt
file_order_path = "./file_order.txt"

# Read the file names from file_order.txt
with open(file_order_path, "r") as file:
    file_order = [line.strip() for line in file]

count = 1

for file_name in file_order:
    full_path = os.path.join(directory, file_name)

    if os.path.exists(full_path):
        new_name = f"Image_{count}.jpg"
        new_path = os.path.join(directory, new_name)

        os.rename(full_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")

        count += 1
    else:
        print(f"File not found: {file_name}")
