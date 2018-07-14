import numpy as np

def ant_conv(ngrid, kernel, pos):
    """ Function to convolve single antenna onto grid
    Args:
        ngrid: Number of grid size per side (512x512 would have ngrid=512)
        kernel: Convolution kernel
        pos: Antenna position x and y, in units of grid pixels
    Returns:
        mat: 1D mapping from antenna to grid (this is one row of the total
             gridding matrix)
    """
    mat = np.zeros((ngrid, ngrid))

    # Do convolution
    pos = np.round(pos).astype(np.int)  # Nearest pixel for now
    mat[pos[0]:pos[0] + kernel.shape[0], pos[1]:pos[1] + kernel.shape[1]] = kernel

    return mat.reshape(-1)

def make_grid(delta, ngrid, kernel, pos, wavelength=None):
    """ Function to convolve antennas onto grid. This creates the gridding matrix
        used to map antenna signals to grid.
    Args:
        delta: Grid resolution, in wavelength
        ngrid: Number of grid size per side (512x512 would have ngrid=512)
        kernel: Convolution kernel. Can be single kernel (same for all antennas),
                or list of kernels.
        pos: Antenna positions (Nant, 2), in wavelengths (if wavelength==None) or
             meters (if wavelength given)
        wavelength: wavelength of observations in meters. If None (default), pos
                    is assumed to be in wavelength already.
    Returns:
        mat: Matrix mapping from antennas to grid.
    """

    mat = np.zeros((ngrid**2, pos.shape[0]))
    if not isinstance(kernel, list):
        kernel = [kernel for i in range(pos.shape[0])]
    # put positions in units of grid pixels
    pos[:, 0] -= pos[:, 0].min()
    pos[:, 1] -= pos[:, 1].min()
    if wavelength is not None:
        pos /= wavelength
    pos /= delta  # units of grid pixels

    for i, p in enumerate(pos):
        mat[:, i] = ant_conv(ngrid, kernel[i], p)
    return mat
