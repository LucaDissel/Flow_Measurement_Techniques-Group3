import numpy as np


def particle_image_diameter(particle_size, magnification, wavelength, fstop):
    """Relation between aperture and particle diffusion.
    :argument particle_size: the mean size of the particle used [in micrometers]
    :argument magnification:  the magnification factor of the camera
    :argument wavelength: the wavelength of laser light [in micrometers]
    :argument fstop: the f-stop setting of the camera

    :return particle_image_size: the size of the particle as perceived in the image frame [in micrometers]"""
    diffusion_diameter = 2.44 * wavelength * (1 + magnification) * fstop  # in [micro m]
    particle_image_size = np.sqrt((magnification * particle_size)**2 + diffusion_diameter**2)
    return particle_image_size


def focal_depth(wavelength, fstop, magnification):
    """Relation between aperture and focal depth.
    :argument wavelength: the wavelength of laser light [in micrometers]
    :argument fstop: the f-stop setting of the camera
    :argument magnification: the magnification factor of the camera

    :return: dz: the focal depth [in micrometers]"""
    dz = 4.88 * wavelength * fstop**2 * ((magnification + 1) / magnification)**2
    return dz


def pulse_time(window_size, magnification, velocity):
    """By Quadrature Rule.
    :argument window_size: the interrogation window size in the image frame [in meters]
    :argument magnification: the magnification factor of the camera
    :argument velocity: the mean flow speed of the particles [in meters per second]

    :return dt: the time between image pairs required [in seconds]"""
    dt = window_size / (4 * magnification * velocity)
    return dt
