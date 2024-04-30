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

Magnification_factor = Npix_horizontal * Pixel_pitch * 1e-6 / FOV_length
Object_distance = Camera_focal_length * (1 + 1 / Magnification_factor)  # in [mm]
Image_distance = Camera_focal_length * (1 + Magnification_factor)  # in [mm]

foo = 0


Wavelength = 532.  # in [nm]

Particle_mean_diameter = 1.068  # in [micro m]


def particle_image_diameter(particle_size, magnification, wavelength, fstop):

    diffusion_diameter = 2.44 * (wavelength * 1e-3) * (1 + magnification) * fstop  # in [micro m]
    particle_image_size = np.sqrt((magnification * particle_size)**2 + diffusion_diameter**2)
    return particle_image_size


def focal_depth():
    pass


def pulse_time(window_size, magnification, velocity):
    # Quadrature rule
    pass


fStop_range = np.linspace(fStop_min, fStop_max, 1001)
Particle_image_diameter_range = particle_image_diameter(Particle_mean_diameter, Magnification_factor, Wavelength,
                                                        fStop_range)

plt.plot(fStop_range, Particle_image_diameter_range)
plt.hlines(Pixel_pitch * 2, fStop_min, fStop_max)
plt.hlines(Pixel_pitch * 3, fStop_min, fStop_max)
plt.xlabel("f#")
plt.ylabel(r"$d_{\tau}$")
plt.grid()
plt.show()

fStop = 9.7
