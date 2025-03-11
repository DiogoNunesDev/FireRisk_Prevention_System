import requests
import base64
import cv2
import numpy as np
import time

# Replace with your Hugging Face Space API URL
HF_API_URL = "https://diogonunesdev-firerisk-prev-system-deployment.hf.space/predict"  

# Path to the image file
IMAGE_PATH = "../Data/Original/Terrain_38.jpg"

# Open image and send it as a request
with open(IMAGE_PATH, "rb") as img_file:
    files = {"file": img_file}
    
    # Retry logic for potential cold starts
    for _ in range(3):  # Try up to 3 times
        response = requests.post(HF_API_URL, files=files)
        if response.status_code == 200:
            break
        print(f"Retrying... (status {response.status_code})")
        time.sleep(5)  # Wait before retrying

# Check if the request was successful
if response.status_code == 200:
    try:
        data = response.json()
        if "segmented_image" in data:
            encoded_image = data["segmented_image"]  # Extract the base64 image string

            # Decode base64 to numpy array
            decoded_data = base64.b64decode(encoded_image)
            np_arr = np.frombuffer(decoded_data, np.uint8)
            segmented_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if segmented_image is None:
                print("Decoding error: The image could not be loaded properly.")
            else:
                # Display image in popup window
                cv2.imshow("Segmented Image", segmented_image)
                cv2.waitKey(0)  # Wait for a key press to close
                cv2.destroyAllWindows()
        else:
            print("Error: 'segmented_image' key not found in response.")
    except Exception as e:
        print(f"Error processing response: {e}")
else:
    print("Request failed with status:", response.status_code)
    print("Response:", response.text)
