import numpy as np
import matplotlib.pyplot as plt

# =========================== PIV PARAMETERS DERIVATION ==========================

# Camera Specs
Npix_horizontal: int = 1628
Npix_vertical: int = 1236
Camera_aspect_ratio = Npix_horizontal / Npix_vertical  # horizontal / vertical
Pixel_pitch = 4.40  # in [micro m]

Max_recording_rate = 8.3  # in [Hz]
Min_separation_time = 1 / Max_recording_rate  # in [s]

Camera_focal_length = 35.  # in [mm]
fStop_min, fStop_max = 2.8, 32


# Object specs
Chord_length = 0.1  # in [m]
FOV_to_chord = 1.5
FOV_length = FOV_to_chord * Chord_length  # in [m]
FOV_height = FOV_length / Camera_aspect_ratio  # in [m]

Magnification_factor = Npix_horizontal * Pixel_pitch * 1e6 / FOV_length
Object_distance = Camera_focal_length * (1 + 1 / Magnification_factor)
Image_distance = Camera_focal_length * (1 + Magnification_factor)
