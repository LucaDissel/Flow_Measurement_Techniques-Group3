import numpy as np

def particle_image_diameter(particle_size, magnification, wavelength, fstop):

    diffusion_diameter = 2.44 * (wavelength) * (1 + magnification) * fstop  # in [micro m]
    particle_image_size = np.sqrt((magnification * particle_size)**2 + diffusion_diameter**2)
    return particle_image_size


def focal_depth():

    pass


def pulse_time(window_size, magnification, velocity):
    # Quadrature rule
    pass