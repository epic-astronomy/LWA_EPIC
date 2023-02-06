import numpy as np
from lsl.common.stations import lwasv
from astropy.constants import c as speed_of_light


def gen_loc_lwasv(
    grid_size=64, grid_resolution=20 / 60.
):
    """
    Generate grid centered locations of LWASV stands compatible with DFT code.
    """

    lsl_locsf = np.array([(ant.stand.x, ant.stand.y, ant.stand.z) for ant in lwasv[::2]])

    lsl_locsf[[i for i, a in enumerate(lwasv[::2]) if a.stand.id == 256], :] = 0.0

    delta = (2 * grid_size * np.sin(np.pi * grid_resolution / 360)) ** -1
    # chan_wavelengths = speed_of_light.value / frequencies
    # sample_grid = chan_wavelengths * delta
    # sll = sample_grid[0] / chan_wavelengths[0]
    # lsl_locs = lsl_locs.T

    lsl_locsf = lsl_locsf / delta
    #sample_grid[np.newaxis, np.newaxis, :, np.newaxis]
    lsl_locsf -= np.min(lsl_locsf, axis=0, keepdims=True)

    # Centre locations slightly
    lsl_locsf += (grid_size - np.max(lsl_locsf, axis=0, keepdims=True)) / 2.

    # add ntime axis
    # final dimensions = 3, ntime, nchan, nant, npol
    # locc = np.broadcast_to(lsl_locsf, (ntime, 3, npol, nchan, lsl_locs.shape[1])).transpose(1, 0, 3, 4, 2).copy()
    return dict(delta=delta, locations=lsl_locsf.astype(np.double))


