# ---------- Step 0: Inititalize ---------
# Import required modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import os
import sys

# Define path to data folder
folder_path = '..\\PIV_data\\tif_img_pairs'

# Load files
files = [os.path.join(folder_path, f'B{str(i).zfill(5)}.tif') for i in range(1, 11)]

# Define path to mask
mask_path = '..\\PIV_data\\masks\Mask_AoA_15.tif'

# Load mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Check if mask is loaded properly
if mask is None:
    print("Failed to load the mask.")
    sys.exit()

# Define window and search sizes
window_size = 32
search_size = int(window_size * 1.5)
overlap = 0.5

# Initialize arrays to store cumulative displacements
total_dx = None
total_dy = None

# --------- Step 1: Process all images ---------
for img_path in files:
    
    # ---------- Step 1-a: Load image ----------
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # ---------- Step 1-b: Split image vertically ----------
    # Assuming two images are stacked vertically
    height = img.shape[0] // 2
    frame_a, frame_b = img[:height, :], img[height:, :]

    # ---------- Step 1-c: Apply the mask ----------
    # Check if mask match image size
    if not (mask.shape == frame_a.shape and mask.shape == frame_b.shape):
        print("Mask does not match the image sizes.")
        sys.exit()
    
    frame_a_masked = cv2.bitwise_and(frame_a, frame_a, mask=mask)
    frame_b_masked = cv2.bitwise_and(frame_b, frame_b, mask=mask)

    # ---------- Step 1-d: Compute number of windows ----------
    # Obtain dimensions of image
    rows, cols = frame_a_masked.shape

    # Compute number of windows in both directions
    num_windows_y = int((rows - (search_size - window_size)) // (window_size * overlap) - 1)
    num_windows_x = int((cols - (search_size - window_size)) // (window_size * overlap) - 1)

    # Determine starting location of first window
    y0 = (rows - (num_windows_y + 1) * overlap * window_size) // 2
    x0 = (cols - (num_windows_x + 1) * overlap * window_size) // 2

    # Initialize arrays to store locations and displacements
    x, y, dx, dy = np.array([]), np.array([]), np.array([]), np.array([])

    # ---------- Step 1-e: Cross-correlate ----------
    for Y in range(num_windows_y):
        for X in range(num_windows_x):
            # Define window in frame_a
            start_y_a = int(y0 + Y * window_size * overlap)
            start_x_a = int(x0 + X * window_size * overlap)
            window_a = frame_a_masked[start_y_a:start_y_a + window_size, start_x_a:start_x_a + window_size]

            # Compute center of frame_a
            y = np.append(y, start_y_a + 0.5 * window_size)
            x = np.append(x, start_x_a + 0.5 * window_size)

            # Define search window in frame_b
            start_y_b = int(y0 + Y * window_size * overlap - 0.25 * window_size)
            start_x_b = int(x0 + X * window_size * overlap - 0.25 * window_size)
            search_b = frame_b_masked[start_y_b:start_y_b + search_size, start_x_b:start_x_b + search_size]

            # Skip windows that include masked areas
            if np.any(mask[start_y_b:start_y_b + search_size, start_x_b:start_x_b + search_size] == 0):
                dx = np.append(dx, 0)
                dy = np.append(dy, 0)
            else:
                # Cross-correlate window from frame_a with search window from frame_b
                result = cv2.matchTemplate(search_b, window_a, cv2.TM_CCOEFF)
                _, _, _, max_loc = cv2.minMaxLoc(result)

                # Compute displacement in pixels
                dx = np.append(dx, max_loc[0] - (search_size - window_size) // 2)
                dy = np.append(dy, max_loc[1] - (search_size - window_size) // 2)
                
    # Accumulate displacements
    if total_dx is None:
        total_dx = dx
        total_dy = dy
    else:
        total_dx += dx
        total_dy += dy

# ---------- Step 2: Obtain flow velocities ----------
# Constants for conversion
pixel_pitch = 0.0000044         # meters
magnification = 0.047754667     
dt = 0.000075                   # seconds

# Compute average displacements
avg_dx = total_dx / len(files)
avg_dy = total_dy / len(files)

# Convert displacements to meters
avg_dx = (avg_dx * pixel_pitch) / magnification
avg_dy = (avg_dy * pixel_pitch) / magnification

# Calculate velocity in meters per second
u = avg_dx / dt             # x-component
v = avg_dy / dt             # y-component
V = np.sqrt(u**2 + v**2)    # magnitude

# Reshape into 2D arrays
u = u.reshape((num_windows_x, num_windows_y))
v = v.reshape((num_windows_x, num_windows_y))
V = V.reshape((num_windows_y, num_windows_x))

# -------- Step 3: Visualize results ----------
# Create grid in mm
X = x.reshape((num_windows_y, num_windows_x)) * pixel_pitch / magnification * 1000
Y = y.reshape((num_windows_y, num_windows_x)) * pixel_pitch / magnification * 1000

# Determine visualization boundaries in pixels for mask
x_start = int(x0)
x_end = int(x0 + ((num_windows_x + 1) / overlap) * window_size)
y_start = int(y0)
y_end = int(y0 + ((num_windows_y + 1) / overlap) * window_size)

# Plot velocity field
plt.figure(figsize=(14, 8))
cp = plt.contourf(X, Y, V, 500, cmap='viridis', zorder=1)
a = 5
plt.quiver(X[::a, ::a], Y[::a, ::a], u[::a, ::a], v[::a, ::a], color='red', scale=400, zorder=2)
plt.gca().invert_yaxis()

# Process mask for overlay
mask_overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)       # Initialize mask overlay with transparent background
mask_overlay[mask == 0, :] = [200, 200, 200, 255]               # Set gray color for masked areas

# Overlay mask on contour plot
plt.imshow(mask_overlay[y_start:y_end, x_start:x_end], extent=[np.amin(X), np.amax(X), np.amax(Y), np.amin(Y)], aspect='auto', zorder=4)

# Process mask for second extended overlay
kernel_size = int(np.sqrt(2*search_size**2))                    # Compute required kernel size
kernel = np.ones((kernel_size, kernel_size), np.uint8)          # Create dilation kernel (structuring element) for extending mask
mask_inverted = cv2.bitwise_not(mask)                           # Invert mask to ensure black area is expanded
mask_dilated = cv2.dilate(mask_inverted, kernel, iterations=1)  # Dilate inverted mask
mask = cv2.bitwise_not(mask_dilated)                            # Invert dilated mask back to original form
mask_overlay_white = np.zeros((*mask.shape, 4), dtype=np.uint8) # Intialize mask overlay with transparent background
mask_overlay_white[mask == 0, :] = [255, 255, 255, 255]         # Set white color for extended masked area

# Overlay extended mask on contour plot
plt.imshow(mask_overlay_white[y_start:y_end, x_start:x_end], extent=[np.amin(X), np.amax(X), np.amax(Y), np.amin(Y)], aspect='auto', zorder=3)

# Final plot adjustments
cbar = plt.colorbar(cp)
cbar.set_label('Velocity in m/s', rotation=270, labelpad=15)
plt.title('Mean velocity field at α=15°')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.show()
