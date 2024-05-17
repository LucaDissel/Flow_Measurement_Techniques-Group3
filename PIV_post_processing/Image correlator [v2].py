# ---------- Step 0: Inititalize ---------
# Import required modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import os
import sys

# Define path to data folder
folder_path = '..\\PIV_data'

# ---------- Step 1: Load image ----------
# Define path to image
img_path = os.path.join(folder_path, 'B00001.tif')

# Load image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Ensure pixel values are within 0-255
img = np.clip(img, 0, 255).astype(np.uint8)

# ---------- Step 2: Split image vertically ----------
# Assuming two images are stacked vertically
height = img.shape[0] // 2
frame_a = img[:height, :]
frame_b = img[height:, :]

# ---------- Step 3: Apply the mask ----------
# Define path to mask
mask_path = os.path.join(folder_path, 'Mask.tif')

# Load mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Check if mask is loaded properly
if mask is None:
    print("Failed to load the mask.")
    sys.exit()

# Ensure mask size matches image size
if mask.shape != frame_a.shape or mask.shape != frame_b.shape:
    print("Mask size does not match the image sizes.")
    sys.exit()

# Apply mask to both frames
frame_a_masked = cv2.bitwise_and(frame_a, frame_a, mask=mask)
frame_b_masked = cv2.bitwise_and(frame_b, frame_b, mask=mask)

# ---------- Step 4: Compute number of windows ----------
# Define window and search sizes
window_size = 32
search_size = int(window_size * 1.5)

# Obtain dimensions of image
rows, cols = frame_a_masked.shape

# Compute number of windows
num_windows_y = (rows - (search_size - window_size)) // window_size
num_windows_x = (cols - (search_size - window_size)) // window_size

# Determine starting location of first window
y0 = (rows - num_windows_y * window_size) // 2
x0 = (cols - num_windows_x * window_size) // 2

# Create arrays to store displacements
displacement_x = np.zeros((rows, cols))
displacement_y = np.zeros((rows, cols))

# ---------- Step 5: Cross-correlate ----------
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

        # Cross-correlate window from frame_a with search window from frame_b
        result = cv2.matchTemplate(search_b, window_a, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # Compute displacement
        displacement_x[start_y_a:start_y_a + window_size, start_x_a:start_x_a + window_size] = max_loc[0] - (search_size - window_size) // 2
        displacement_y[start_y_a:start_y_a + window_size, start_x_a:start_x_a + window_size] = max_loc[1] - (search_size - window_size) // 2

# ---------- Step 6: Obtain flow velocity ----------
# Constants for conversion
pixel_pitch = 0.0000044         # meters
magnification = 0.047754667     
dt = 0.000075                   # seconds

# Calculate displacements in meters
displacement_x_meters = (displacement_x * pixel_pitch) / magnification
displacement_y_meters = (displacement_y * pixel_pitch) / magnification

# Calculate velocity in meters per second
u = displacement_x_meters / dt      # x-component
v = displacement_y_meters / dt      # y-component
V = np.sqrt(u**2 + v**2)            # magnitude

# -------- Step 7: Visualize results ----------
# Create grid with physical distances in mm
x = np.arange(0, cols) * (pixel_pitch / magnification * 1000)
y = np.arange(0, rows) * (pixel_pitch / magnification * 1000)
X, Y = np.meshgrid(x, y)

# Determine visualization boundaries in px
x_start = x0
x_end = int(x0 + num_windows_x * window_size)
y_start = y0
y_end = int(y0 + num_windows_y * window_size)

# Plot velocity field
plt.figure(figsize=(10, 8))
cp = plt.contourf(X[y_start:y_end, x_start:x_end], Y[y_start:y_end, x_start:x_end], V[y_start:y_end, x_start:x_end], 50, cmap='viridis', zorder=1)
plt.quiver(X[y_start:y_end:64, x_start:x_end:64], Y[y_start:y_end:64, x_start:x_end:64], u[y_start:y_end:64, x_start:x_end:64], v[y_start:y_end:64, x_start:x_end:64], color='red', scale=500, zorder=2)

# Process mask for overlay
mask_overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)       # Initialize mask overlay with transparent background
mask_overlay[mask == 0, :] = [200, 200, 200, 255]               # Set gray color for masked areas

# Overlay mask on contour plot
plt.imshow(mask_overlay[y_start:y_end, x_start:x_end], extent=[0, max(x), max(y), 0], aspect='auto', zorder=4)

# Process mask for second extended overlay
kernel = np.ones((search_size, search_size), np.uint8)          # Create dilation kernel (structuring element) for extending mask
mask_inverted = cv2.bitwise_not(mask)                           # Invert mask to ensure black area is expanded
mask_dilated = cv2.dilate(mask_inverted, kernel, iterations=1)  # Dilate inverted mask
mask = cv2.bitwise_not(mask_dilated)                            # Invert dilated mask back to original form
mask_overlay_white = np.zeros((*mask.shape, 4), dtype=np.uint8) # Intialize mask overlay with transparent background
mask_overlay_white[mask == 0, :] = [255, 255, 255, 255]         # Set white color for extended masked area

# Overlay extended mask on contour plot
plt.imshow(mask_overlay_white[y_start:y_end, x_start:x_end], extent=[0, max(x), max(y), 0], aspect='auto', zorder=3)

# Final plot adjustments
plt.colorbar(cp)                                            
plt.title('Instantaneous velocity field at α=15°')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.show()
