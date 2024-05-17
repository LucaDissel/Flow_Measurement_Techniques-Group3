import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import os
import sys

# ---------- Step 1: Load and average the images ----------
# Define the path to the images
folder_path = '..\\PIV_data'
file_names = [f'B0000{i}.tif' for i in range(1, 10)] + ['B00010.tif']
files = [os.path.join(folder_path, name) for name in file_names]

# Initialize an array to accumulate the images
sum_exposure = None

# Load each image and accumulate exposure values
for file in files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load {file}.")
        
    if sum_exposure is None:
        sum_exposure = np.zeros_like(img, dtype=np.float64)
    sum_exposure += img.astype(np.float64)  # Ensure data type is float for accurate summation

# Compute the average exposure
average_exposure = sum_exposure / len(files)

# ---------- Step 2: Subtract the average exposure from image B00001 ----------
# Load the first image (B00001)
img_B00001 = cv2.imread(os.path.join(folder_path, 'B00001.tif'), cv2.IMREAD_GRAYSCALE)

# Subtract the average
reduced_image = img_B00001 - average_exposure

# Ensure pixel values are within 0-255
reduced_image = np.clip(reduced_image, 0, 255).astype(np.uint8)

# Save the result
cv2.imwrite(os.path.join(folder_path, 'B00001-reduced.tif'), reduced_image)


# ---------- Step 3: Split image vertically ----------
# Assuming the two images are stacked vertically
height = reduced_image.shape[0] // 2
frame_a = reduced_image[:height, :]
frame_b = reduced_image[height:, :]

# ---------- Step 4: Apply the mask ----------
# Load the mask
mask = cv2.imread(os.path.join(folder_path, 'Mask.tif'), cv2.IMREAD_GRAYSCALE)

# Check if mask is loaded properly
if mask is None:
    print("Failed to load the mask.")
    sys.exit()

# Ensure red_mask is the same size as the images being processed
if mask.shape != frame_a.shape or mask.shape != frame_b.shape:
    print("Mask size does not match the image sizes.")
    sys.exit(1)

# Apply this mask to both split images
# Black areas (0) in  mask are areas to remove, white areas (255) to keep
frame_a_masked = cv2.bitwise_and(frame_a, frame_a, mask=mask)
frame_b_masked = cv2.bitwise_and(frame_b, frame_b, mask=mask)

# Save the results
cv2.imwrite(os.path.join(folder_path, 'B00001-frame_a.tif'), frame_a_masked)
cv2.imwrite(os.path.join(folder_path, 'B00001-frame_b.tif'), frame_b_masked)

# ---------- Step 5: Compute number of windows ----------
# Define window sizes
window_size = 32
search_size = int(window_size * 1.5)

# Dimensions of the image
rows, cols = frame_a_masked.shape

# Compute the number of windows
num_windows_y = int((rows - (search_size - window_size)) // window_size)
num_windows_x = int((cols - (search_size - window_size)) // window_size)

# Determine first window location
y0 = int((rows - num_windows_y * window_size) // 2)
x0 = int((cols - num_windows_x * window_size) // 2)

# Create arrays to store displacement
displacement_x = np.zeros((rows, cols))
displacement_y = np.zeros((rows, cols))

# ---------- Step 6: Cross-correlate ----------
for y in range(num_windows_y):
    for x in range(num_windows_x):
        # Define window in frame_a
        start_y_a = int(y0 + y * window_size)
        start_x_a = int(x0 + x * window_size)
        window_a = frame_a_masked[start_y_a:start_y_a + window_size, start_x_a:start_x_a + window_size]

        # Define search window in frame_b
        start_y_b = int(y0 + y * window_size - 0.25 * window_size)
        start_x_b = int(x0 + x * window_size - 0.25 * window_size)
        search_b = frame_b_masked[start_y_b:start_y_b + search_size, start_x_b:start_x_b + search_size]

        # Skip windows that include masked areas
        if np.any(mask[start_y_b:start_y_b + search_size, start_x_b:start_x_b + search_size] == 0):
            continue

        # Cross-correlate the window with the search window
        result = cv2.matchTemplate(search_b, window_a, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # Compute the displacement
        displacement_x[start_y_a:start_y_a + window_size, start_x_a:start_x_a + window_size] = max_loc[0] - (search_size - window_size) // 2
        displacement_y[start_y_a:start_y_a + window_size, start_x_a:start_x_a + window_size] = max_loc[1] - (search_size - window_size) // 2

# ---------- Step 7: Obtain flow velocity ----------
# Constants
pixel_pitch = 0.0000044         # meters
magnification = 0.047754667
dt = 0.000075                   # seconds

# Calculate magnitudes of displacements in meters
displacement_x_meters = (displacement_x * pixel_pitch) / magnification
displacement_y_meters = (displacement_y * pixel_pitch) / magnification

# Calculate velocity in meters per second
u = displacement_x_meters / dt
v = displacement_y_meters / dt
V = np.sqrt(u**2 + v**2)

# -------- Step 8: Visualize results ----------
# Create a grid with physical distances in mm
x = np.arange(0, cols) * (pixel_pitch / magnification * 1000)
y = np.arange(0, rows) * (pixel_pitch / magnification * 1000)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(10, 8))
cp = plt.contourf(X, Y, V, 25, cmap='viridis', zorder=1)
plt.quiver(X[::64, ::64], Y[::64, ::64], u[::64, ::64], v[::64, ::64], color='red', scale=500, zorder=2)

# Load and process the mask for overlay
mask_rgb = cv2.imread(os.path.join(folder_path, 'Mask.tif'), cv2.IMREAD_GRAYSCALE)
mask_overlay = np.zeros((*mask_rgb.shape, 4), dtype=np.uint8)
mask_overlay[mask_rgb == 0, :] = [200, 200, 200, 255]

# Overlay the mask on the plot
plt.imshow(mask_overlay, extent=[0, max(x), max(y), 0], aspect='auto', zorder=4)

# Create a dilation kernel (structuring element)
kernel = np.ones((search_size, search_size), np.uint8)

# Invert the mask to ensure the white area (255) is expanded
mask_inverted = cv2.bitwise_not(mask_rgb)

# Dilate the inverted mask to expand the white areas
mask_dilated = cv2.dilate(mask_inverted, kernel, iterations=1)

# Invert the dilated mask back to original form
mask_dilated_inverted = cv2.bitwise_not(mask_dilated)

# Create an RGBA overlay from the dilated mask
mask_overlay_white = np.zeros((*mask_dilated_inverted.shape, 4), dtype=np.uint8)
mask_overlay_white[mask_dilated_inverted == 0, :] = [255, 255, 255, 255]

# Overlay the mask on the plot
plt.imshow(mask_overlay_white, extent=[0, max(x), max(y), 0], aspect='auto', zorder=3)

plt.colorbar(cp)                                            # Adds a colorbar to show the magnitude scale
plt.title('Instantaneous velocity field at α=15°')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.show()
