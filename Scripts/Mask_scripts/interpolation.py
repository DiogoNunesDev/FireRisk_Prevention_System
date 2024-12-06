import os
import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2
from tqdm import tqdm

def impute_unassigned_pixels(mask):

    unassigned = (mask == 0)

    indices = np.indices(mask.shape)

    distances, nearest_indices = distance_transform_edt(
        unassigned,
        return_indices=True
    )

    mask[unassigned] = mask[tuple(nearest_indices[:, unassigned])]

    return mask

def process_masks_in_folder(folder_path):

    mask_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    for mask_file in tqdm(mask_files, desc="Processing Masks"):
        mask_path = os.path.join(folder_path, mask_file)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        updated_mask = impute_unassigned_pixels(mask)

        cv2.imwrite(mask_path, updated_mask)

folder_path = "../Masks"

process_masks_in_folder(folder_path)
