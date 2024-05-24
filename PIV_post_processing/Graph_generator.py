# ---------- Step 0: Initialize ----------
import pandas as pd
import sys
from tabulate import tabulate
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------- Step 1: Choose data file ----------
# Data
data = {
    '#': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'AoA': [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 15, 15, 15, 15, 15],
    'WS': [16, 32, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
    'ov': [0, 0, 0, 50, 50, 50, 0, 0, 50, 50, 50, 50, 50, 50, 50],
    'MP/SP': ['SP', 'SP', 'SP', 'SP', 'MP', 'MP', 'SP', 'SP', 'MP', 'MP', 'MP', 'MP', 'MP', 'MP', 'MP'],
    'dt': [75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 10, 10, 75, 75, 75],
    'instant/mean': ['instant', 'instant', 'instant', 'instant', 'instant', 'mean_20', 'instant', 'mean_20', 'instant', 'mean_20', 'instant', 'mean_20', 'instant', 'mean_10', 'mean_100']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

# Prompt user for input
dataset_number = int(input("\nEnter the number corresponding to the dataset you want to read (1-15): "))

# Validate input
if dataset_number < 1 or dataset_number > 15:
    print("Invalid selection. Please run the script again and choose a number between 1 and 15.")
    sys.exit()
else:
    selected_data = df.iloc[dataset_number - 1]
    base_path = f"..\\PIV_data\\DAVIS_data\\AoA_{selected_data['AoA']}\\WS_{selected_data['WS']}x{selected_data['WS']}\\ov_{selected_data['ov']}\\{selected_data['MP/SP']}\\dt_{selected_data['dt']}\\{selected_data['instant/mean']}"
    
    if 'instant' in selected_data['instant/mean']:
        file_name = "B00001.dat"
        title_prefix = "Instantaneous"
    else:
        plot_type = input("Do you want to plot average [1] or the RMS [2]?: ")
        if plot_type == '1':
            file_name = "B00001.dat"
            title_prefix = "Mean"
        elif plot_type == '2':
            file_name = "B00002.dat"
            title_prefix = "RMS"
        else:
            print("Invalid selection. Please run the script again and choose [1] for average or [2] for RMS.")
            sys.exit()
    
    path = f"{base_path}\\{file_name}"
    print(f"\nPath to the selected data file: {path}")

# ---------- Step 2: Process data file --------
# Read data file
try:
    with open(path, 'r') as file:
        lines = file.readlines()
except FileNotFoundError:
    print(f"File not found at path: {path}.")
    sys.exit()
except Exception as e:
    print(f"An error occurred while reading the file: {e}.")
    sys.exit()

# Skip header
data = np.array([line.strip().split() for line in lines[3:]], dtype=float)

# Extract columns
x, y, u, v = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# Adjust x and y to start at 0
x -= np.min(x)
y -= np.min(y)

# Compute velocity magnitude
V = np.sqrt(u**2 + v**2)

# Reshape into 2D arrays
unique_x = np.unique(x)
unique_y = np.unique(y)
u = u.reshape(len(unique_y), len(unique_x))
v = v.reshape(len(unique_y), len(unique_x))
V = V.reshape(len(unique_y), len(unique_x))

# ---------- Step 3: Load mask -----------
# Define path to mask
mask_path = f'..\\PIV_data\\masks\Mask_AoA_{selected_data["AoA"]}.tif'
print(f"\nPath to selected mask: {mask_path}")

# Load mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Check if mask is loaded properly
if mask is None:
    print("Failed to load the mask.")
    sys.exit()

# -------- Step 4: Visualize results ----------
# Create grid
X, Y = np.meshgrid(unique_x, unique_y)

# Plot velocity field
plt.figure(figsize=(14, 8))
cp = plt.contourf(X, Y, V, 500, cmap='viridis', vmin=0, vmax=12, zorder=1)
a = 5
plt.quiver(X[::a, ::a], Y[::a, ::a], u[::a, ::a], v[::a, ::a], color='red', scale=400, zorder=2)
plt.gca().invert_yaxis()

# Process mask for overlay
mask_overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)       # Initialize mask overlay with transparent background
mask_overlay[mask == 0, :] = [200, 200, 200, 255]               # Set gray color for masked areas

# Overlay mask on contour plot
plt.imshow(mask_overlay, extent=[min(x), max(x), max(y), min(y)], aspect='auto', zorder=4)

# Process mask for second extended overlay
kernel_size = 50                                                # Kernel size
kernel = np.ones((kernel_size, kernel_size), np.uint8)          # Create dilation kernel (structuring element) for extending mask
mask_inverted = cv2.bitwise_not(mask)                           # Invert mask to ensure black area is expanded
mask_dilated = cv2.dilate(mask_inverted, kernel, iterations=1)  # Dilate inverted mask
mask = cv2.bitwise_not(mask_dilated)                            # Invert dilated mask back to original form
mask_overlay_white = np.zeros((*mask.shape, 4), dtype=np.uint8) # Intialize mask overlay with transparent background
mask_overlay_white[mask == 0, :] = [255, 255, 255, 255]         # Set white color for extended masked area

# Overlay extended mask on contour plot
plt.imshow(mask_overlay_white, extent=[min(x), max(x), max(y), min(y)], aspect='auto', zorder=3)

# Final plot adjustments
cbar = plt.colorbar(cp)
cbar.set_label('Velocity in m/s', rotation=270, labelpad=15)
plt.title(f'{title_prefix} velocity field at α={selected_data["AoA"]}°')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.show()
