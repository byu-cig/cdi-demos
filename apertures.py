import LightPipes as lp
from LightPipes.units import *
import utils as ut
from skimage import transform, draw
from skimage import data as skidata
import numpy as np
from matplotlib import pyplot as plt


def double_pinhole(resolution=256, frame_size=20.0*mm, radius=0.4*mm, separation=2.0*mm, wavelength=5*um):
    # Create double pinhole
    F = lp.Begin(frame_size, wavelength, resolution)
    F1 = lp.CircAperture(radius / 2.0, -separation / 2.0, 0, F)
    F2 = lp.CircAperture(radius / 2.0, separation / 2.0, 0, F)
    F = lp.BeamMix(F1, F2)

    # Calculate the Fraunhofer distance
    D = separation + (2 * radius)
    d_fraun = 2 * D**2 / wavelength

    return F, d_fraun


def horse(resolution=256, frame_size=20.0*mm, horse_size=5.0*mm, wavelength=5*um):
    h_pix = int(resolution * horse_size / frame_size)
    h = transform.resize(np.logical_not(skidata.horse()), (h_pix, h_pix))
    h = ut.pad_to_size(h, resolution)

    F = lp.Begin(frame_size, wavelength, resolution)
    F.field = F.field * h

    d_fraun = 2 * horse_size**2 / wavelength

    return F, d_fraun


def zone_plate(resolution=512, frame_size=20.0*cm, r_0=1.0*cm, N_rings=20, wavelength=5*um):
    r_n = [np.sqrt(n+1)*r_0 for n in range(N_rings*2)]
    focus = 2 * r_n[-1] * (r_n[-1] - r_n[-2]) / wavelength

    F = lp.Begin(frame_size, wavelength, resolution)
    F_out = lp.Begin(frame_size, wavelength, resolution)
    F_out.field = F_out.field * 0
    for n in range(N_rings):
        F_out.field = F_out.field + lp.CircScreen(F, R=r_n[2*n]).field * lp.CircAperture(F, R=r_n[2*n+1]).field

    return F_out, focus


if __name__ == '__main__':
    field, _ = zone_plate(resolution=512)
    plt.imshow(lp.Intensity(field))
    plt.show()
