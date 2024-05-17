import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data from the text file
folder_path = 'HWA_data'
# files = [os.path.join(folder_path, f'B0000{i}.tif') for i in range(1, 11)]
file_path = "HWA\\HWA_data\\Measurement_pos32_aoa0.txt"
data = pd.read_csv(file_path, sep="\t", skiprows=lambda x: x in range(23) or x==1, usecols=[0, 1])

# Assuming you have a calibration to convert voltage to wind speed (e.g., m/s per volt)
# Replace these coefficients with your actual calibration data
voltage_to_m_per_s = 2.5  # Example conversion factor

# Convert voltage readings to speed using the calibration factor
data['Speed (m/s)'] = data['Voltage'] * voltage_to_m_per_s

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(data['X_Value'], data['Speed (m/s)'], label='Wind Speed', color='blue')
plt.title('Wind Speed Measurement from HWA')
plt.xlabel('Time (s)')
plt.ylabel('Wind Speed (m/s)')
plt.legend()
plt.grid(True)
plt.show()