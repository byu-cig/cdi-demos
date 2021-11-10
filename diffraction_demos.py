import skimage.data as skdata
import utils as ut
import apertures as ap
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import LightPipes as lp
from progressbar import progressbar as pbar


def phase_problem():
    img = skdata.camera()
    dif = ut.log(ut.fft(img))
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'xticks': [], 'yticks': []}, tight_layout=True)
    ax1.set_title('Original image')
    ax1.imshow(img, cmap='gray')
    ax2.set_title('Diffraction pattern (with phase)')
    ax2.imshow(ut.comp_to_rgb(dif))
    ax3.set_title('Diffraction pattern (intensity only)')
    ax3.imshow(np.abs(dif), cmap='gray')

    save_to = Path(ut.get_save_dir())
    plt.imsave(save_to/'camera.png', img, cmap='gray')
    plt.imsave(save_to/'camera_diffraction_rgb.png', ut.comp_to_rgb(dif))
    plt.imsave(save_to/'camera_diffraction_amp.png', np.abs(dif), cmap='gray')


def evolve(aperture, *plots, prop_dist=0.2, Nz=1000, cmap='plasma', **aperture_kwargs):
    """
    Calculate and show the central slice of the intensity profile as it evolves from an aperture.

    :param aperture: aperture function (must be a function from apertures.py)
    :type aperture: function
    :param plots: strings that specify which plots to make (currently can be 'evolution' and/or 'full')
    :type plots: str
    :param prop_dist: maximum propagation distance *relative to fraunhofer distance*
    :type prop_dist: float
    :param cmap: matplotlib colormap to use for plots
    :type cmap: str
    :param aperture_kwargs: keyword arguments to pass to the aperture function
    """
    F, d_fraun = aperture(**aperture_kwargs)
    N = F.grid_dimension
    z_max = d_fraun * prop_dist

    z_range = np.linspace(0, z_max, Nz)
    out = np.zeros((Nz, N))

    # Calculate intensity at each z along propagation
    for i in pbar(range(len(z_range))):
        I = lp.Intensity(lp.Fresnel(z_range[i], F), 2)
        out[i] = I[N//2]

    out = np.rot90(out)

    if 'evolution' in plots:
        plt.figure()
        plt.imshow(out, cmap=cmap)
        plt.axis('off')
        plt.title('intensity pattern')
        plt.show()
        ut.save_image(out, cmap=cmap)

    if 'full' in plots:
        fig = plt.figure(tight_layout=True)
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax1.set_title('Z = 0 m')
        plt.axis('off')
        ax1.imshow(lp.Intensity(F, 2), cmap=cmap)
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax2.set_title(f'Z = {z_max:0.2} m')
        plt.axis('off')
        ax2.imshow(lp.Intensity(lp.Fresnel(z_max, F), 2), cmap=cmap)
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        ax3.set_title('Central slice evolution')
        plt.axis('off')
        ax3.imshow(out, cmap=cmap)
        plt.show()


if __name__ == '__main__':
    evolve(ap.zone_plate, 'full', 'evolution', Nz=1500, prop_dist=1.5, resolution=512)
