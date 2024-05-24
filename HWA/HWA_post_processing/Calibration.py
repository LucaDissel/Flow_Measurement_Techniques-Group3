import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import os

# Directory containing the data
folder_path = '../HWA_data'  
folder_path = os.path.normpath(folder_path)     # Normalize path

# Check and list files in the directory
if os.path.exists(folder_path):
    print("Directory exists. Files in directory:")
    files = [f for f in os.listdir(folder_path) if f.startswith('Calibration') in f]
    print(files)
else:
    print("Directory does not exist. Check the path.")
    exit()

# Read and process each calibration file
velocities = np.arange(0, 21, 2)                # Velocity steps from 0 to 20 m/s
voltage_means = []

for speed in velocities:
    file_name = f'Calibration_{speed:03d}.txt'  # Format the filename
    file_path = os.path.join(folder_path, file_name)
    try:
        data = pd.read_csv(file_path, sep="\t", skiprows=24, header=None, names=['Time', 'Voltage'])
        voltage_mean = data['Voltage'].mean()
        voltage_means.append(voltage_mean)
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        voltage_means.append(np.nan)  # Append NaN for missing files

# Remove NaN values in case some files were missing
valid_indices = ~np.isnan(voltage_means)
valid_velocities = velocities[valid_indices]
valid_voltage_means = np.array(voltage_means)[valid_indices]

# Fit a polynomial to the voltage data
degrees = [4,6]
coeff = []
print(coeff)

for i,degree in enumerate(degrees):
    coeff.append(Polynomial.fit(valid_velocities, valid_voltage_means, deg=degree))

    # Plot the fit
    x_fit = np.linspace(0, 20, 100)
    y_fit = coeff[i](x_fit)

    # Output the coefficients
    print(f'Degree:',degree,"            Polynomial Coefficients:", coeff[i].convert().coef)



# def voltage_to_airspeed(voltage, coeffs):
#     airspeed = np.polyval(coeffs[::-1], voltage)  # coeffs should be in decreasing order for np.polyval
#     return airspeed


# for polynomial_coefficients in coeff:
#     polynomial_coefficients.convert().coef
#     estimated_airspeed = voltage_to_airspeed(voltage_means, polynomial_coefficients)
#     print(f"Estimated Airspeed: {estimated_airspeed:.2f} m/s")