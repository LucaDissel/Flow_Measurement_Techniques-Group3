import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images_top = []  # To store the top halves
    images_bottom = []  # To store the bottom halves
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is not None:
            # Calculate the splitting index
            split_index = img.shape[0] // 2
            
            # Divide the image into top and bottom halves
            top_half = img[:split_index, :]
            bottom_half = img[split_index:, :]
            
            # Append each half to its respective list
            images_top.append(top_half)
            images_bottom.append(bottom_half)
    return images_top, images_bottom


def compute_background(images):
    # Stack images along the third dimension and compute median
    stack = np.dstack(images)
    background = np.median(stack, axis=2)
    return background

def subtract_background(images, background):
    background_subtracted_images = []
    for img in images:
        # Ensure background is subtracted as float to prevent underflow
        diff = cv2.absdiff(img.astype(np.float32), background.astype(np.float32))
        # Normalize back to uint8
        norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        background_subtracted_images.append(norm_diff)
    return background_subtracted_images

# Specify the path to your folder with .tif images
folder_path = 'PIV_data'

# Load images
images = load_images_from_folder(folder_path)

# Compute the background
background = compute_background(images)

# Subtract the background from each image
background_subtracted_images = subtract_background(images, background)

# Example to display the first subtracted image (optional)
cv2.imshow('Background Subtracted Image', background_subtracted_images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
