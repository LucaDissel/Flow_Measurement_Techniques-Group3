import numpy as np
import cv2
import os
import sys

# Directory containing the images
folder_path = 'PIV data'
files = [os.path.join(folder_path, f'B0000{i}.tif') for i in range(1, 11)]

# Initialize an array to accumulate the images
sum_image = None

# Load each image and accumulate
for file in files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if sum_image is None:
        sum_image = np.zeros_like(img, dtype=np.float64)
    sum_image += img

# Compute the average image
average_image = sum_image / len(files)

# Load the first image
img_B00001 = cv2.imread(os.path.join(folder_path, 'B00001.tif'), cv2.IMREAD_GRAYSCALE)

# Subtract the average
diff_image = img_B00001 - average_image

sys.exit()

# Assuming the two images are stacked vertically
height = diff_image.shape[0] // 2
image1 = diff_image[:height, :]
image2 = diff_image[height:, :]

# Load the mask
mask = cv2.imread(os.path.join(folder_path, 'Mask.png'), cv2.IMREAD_COLOR)

# Create a binary mask where the red pixels are
red_mask = (mask[:,:,2] == 255) & (mask[:,:,1] == 0) & (mask[:,:,0] == 0)
# We will check if a block contains any masked pixels before processing it

def cross_correlate_blocks(img1, img2, mask, block_size=32):
    u, v = [], []
    mask_height = mask.shape[0] // 2  # Adjust if mask is not split like the images
    for i in range(0, img1.shape[0] - block_size + 1, block_size):
        for j in range(0, img1.shape[1] - block_size + 1, block_size):
            # Check mask for the current block
            if np.any(mask[i:i+block_size, j:j+block_size]):
                continue  # Skip this block if any red-masked pixels are present

            block1 = img1[i:i+block_size, j:j+block_size]
            block2 = img2[i:i+block_size, j:j+block_size]
            # Cross-correlation
            result = cv2.matchTemplate(block2, block1, method=cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            u.append(max_loc[0] - block_size//2)
            v.append(max_loc[1] - block_size//2)
    return np.array(u), np.array(v)

u, v = cross_correlate_blocks(image1, image2, red_mask[:height], block_size=32)

# Plot the data
import matplotlib.pyplot as plt
X, Y = np.meshgrid(np.arange(0, image1.shape[1], 32), np.arange(0, image1.shape[0], 32))
plt.figure()
plt.quiver(X, Y, u, v)
plt.title('Velocity Field')
plt.show()

