import numpy as NP
import scipy.constants as FCNST

################################################################################    

def inputcheck(locs, wavelength=1.0, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0,
               rmin=0.0, rmax=1.0, rotangle=0.0, pointing_center=None):

    """
    ----------------------------------------------------------------------------
    Checks inputs for compatibility for analytic aperture kernel estimation

    Inputs:

    locs    [numpy array] locations at which aperture kernel is to be estimated. 
            Must be a Mx2 numpy array where M is the number of locations, x- and 
            y-locations are stored in the first and second columns respectively.
            The units can be arbitrary but preferably that of distance. Must be
            specified, no defaults.

    wavelength
            [scalar or numpy array] Wavelength of the radiation. If it is a 
            scalar or numpy array of size 1, it is assumed to be identical for 
            all locations. If an array is provided, it must be of same size as
            the number of locations at which the aperture kernel is to be 
            estimated. Same units as locs. Default=1.0

    xmin    [scalar] Lower limit along the x-axis for the aperture kernel 
            footprint. Applicable in case of rectangular or square apertures. 
            Same units as locs. Default=-1.0

    xmax    [scalar] Upper limit along the x-axis for the aperture kernel 
            footprint. Applicable in case of rectangular or square apertures. 
            Same units as locs. Default=1.0

    ymin    [scalar] Lower limit along the y-axis for the aperture kernel 
            footprint. Applicable in case of rectangular or square apertures. 
            Same units as locs. Default=-1.0

    ymax    [scalar] Upper limit along the y-axis for the aperture kernel 
            footprint. Applicable in case of rectangular or square apertures. 
            Same units as locs. Default=1.0

    rmin    [scalar] Lower limit along the radial direction for the aperture 
            kernel footprint. Applicable in case of circular apertures. Same 
            units as locs. Default=0.0

    rmax    [scalar] Upper limit along the radial direction for the aperture 
            kernel footprint. Applicable in case of circular apertures. Same 
            units as locs. Default=1.0

    rotangle
            [scalar] Angle (in radians) by which the principal axis of the 
            aperture is rotated counterclockwise east of sky frame. 
            Applicable in case of rectangular, square and elliptical 
            apertures. Default=0.0

    pointing_center
            [numpy array] Pointing center to phase the aperture illumination 
            to. Must be a 2-element array denoting the x- and y-direction 
            cosines that obeys rules of direction cosines. Default=None 
            (zenith)

    Outputs:

    Dictionary consisting of corrected values of input under the following 
    keys:
    'locs'  [numpy array] Corrected locations for aperture kernel estimation. 
            Mx2 array for x and y coordinates of M locations. Same units as 
            locs
    'wavelength'
            [numpy array] Corrected wavelengths. 1x1 or Mx1 array. Same units
            as locs
    'xmin'  [scalar] Corrected lower limit along the x-axis for the aperture 
            kernel footprint. Applicable in case of rectangular or square 
            apertures. Same units as locs
    'xmax'  [scalar] Corrected upper limit along the x-axis for the aperture 
            kernel footprint. Applicable in case of rectangular or square 
            apertures. Same units as locs
    'ymin'  [scalar] Corrected lower limit along the y-axis for the aperture 
            kernel footprint. Applicable in case of rectangular or square 
            apertures. Same units as locs
    'ymax'  [scalar] Corrected upper limit along the y-axis for the aperture 
            kernel footprint. Applicable in case of rectangular or square 
            apertures. Same units as locs
    'rmin'  [scalar] Corrected lower limit along the radial direction for the 
            aperture kernel footprint. Applicable in case of circular apertures. 
            Same units as locs
    'rmax'  [scalar] Corrected upper limit along the radial direction for the 
            aperture kernel footprint. Applicable in case of circular apertures. 
            Same units as locs
    'rotangle'
            [scalar] Angle (in radians) by which the principal axis of the 
            aperture is rotated counterclockwise east of sky frame. 
    'pointing_center'
            [numpy array] Corrected pointing center to phase the aperture 
            illumination to. It is a 2-element array denoting the x- and 
            y-direction cosines
    ----------------------------------------------------------------------------
    """

    try:
        locs
    except NameError:
        raise NameError('x and y locations must be specified in locs')

    if not isinstance(locs, NP.ndarray):
        raise TypeError('locs must be a numpy array')

    outdict = {}
    locs = NP.squeeze(locs)
    locs_shape = locs.shape
    locs_size = locs.size

    if (locs.ndim < 1) or (locs.ndim > 3):
        raise ValueError('locs must be a one-, two- or three-dimensional array')

    if locs.ndim == 1:
        if (locs_size < 2) or (locs_size > 3):
            raise ValueError('locs must contain at least two elements and not more than three elements')
        locs = locs[:2].reshape(1,-1)
    else:
        if locs.ndim == 3:
            locs = locs[:,:,0]
        if locs.shape[1] > 2:
            locs = locs[:,:2]

    outdict['locs'] = locs.reshape(-1,2)

    if not isinstance(wavelength, (int, float, NP.ndarray)):
        raise TypeError('wavelength must be a scalar or numpy array')
    else:
        wavelength = NP.asarray(wavelength, dtype=NP.float).reshape(-1)
        if (wavelength.size != 1) and (wavelength.size != locs.shape[0]):
            raise ValueError('Dimensions of wavelength are incompatible with those of locs')
        if NP.any(wavelength <= 0.0):
            raise ValueError('wavelength(s) must be positive')

    outdict['wavelength'] = wavelength
    
    if not isinstance(xmin, (int,float)):
        raise TypeError('xmin must be a scalar')
    else:
        xmin = float(xmin)

    if not isinstance(xmax, (int,float)):
        raise TypeError('xmax must be a scalar')
    else:
        xmax = float(xmax)

    if xmin >= xmax:
        raise ValueError('xmin must be less than xmax')

    if not isinstance(ymin, (int,float)):
        raise TypeError('ymin must be a scalar')
    else:
        ymin = float(ymin)

    if not isinstance(ymax, (int,float)):
        raise TypeError('ymax must be a scalar')
    else:
        ymax = float(ymax)

    if ymin >= ymax:
        raise ValueError('ymin must be less than ymax')

    if not isinstance(rmin, (int,float)):
        raise TypeError('rmin must be a scalar')
    else:
        rmin = float(rmin)

    if not isinstance(rmax, (int,float)):
        raise TypeError('rmax must be a scalar')
    else:
        rmax = float(rmax)

    if rmin < 0.0:
        rmin = 0.0
    if rmin >= rmax:
        raise ValueError('rmin must be less than rmax')
    
    outdict['xmin'] = xmin
    outdict['xmax'] = xmax
    outdict['ymin'] = ymin
    outdict['ymax'] = ymax
    outdict['rmin'] = rmin
    outdict['rmax'] = rmax

    if not isinstance(rotangle, (int,float)):
        raise TypeError('Rotation angle must be a scalar')
    else:
        rotangle = float(rotangle)

    outdict['rotangle'] = rotangle
        
    if pointing_center is not None:
        if not isinstance(pointing_center, NP.ndarray):
            raise TypeError('pointing_center must be a numpy array')
        pointing_center = NP.squeeze(pointing_center)
        if pointing_center.size != 2:
            raise ValueError('pointing center must be a 2-element numpy array')
        if NP.sum(pointing_center**2) > 1.0:
            raise ValueError('pointing center incompatible with rules of direction cosines')
    else:
        pointing_center = NP.asarray([0.0, 0.0]).reshape(1,-1)

    outdict['pointing_center'] = pointing_center

    return outdict
        
################################################################################

def rect(locs, wavelength=1.0, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0,
         rotangle=0.0, pointing_center=None):

    """
    ----------------------------------------------------------------------------
    Rectangular aperture kernel estimation

    Inputs:

    locs    [numpy array] locations at which aperture kernel is to be estimated. 
            Must be a Mx2 numpy array where M is the number of locations, x- and 
            y-locations are stored in the first and second columns respectively.
            The units can be arbitrary but preferably that of distance. Must be 
            specified, no defaults

    wavelength
            [scalar or numpy array] Wavelength of the radiation. If it is a 
            scalar or numpy array of size 1, it is assumed to be identical for 
            all locations. If an array is provided, it must be of same size as
            the number of locations at which the aperture kernel is to be 
            estimated. Same units as locs. Default=1.0

    xmin    [scalar] Lower limit along the x-axis for the aperture kernel 
            footprint. Applicable in case of rectangular or square apertures. 
            Same units as locs. Default=-1.0

    xmax    [scalar] Upper limit along the x-axis for the aperture kernel 
            footprint. Applicable in case of rectangular or square apertures. 
            Same units as locs. Default=1.0

    ymin    [scalar] Lower limit along the y-axis for the aperture kernel 
            footprint. Applicable in case of rectangular or square apertures. 
            Same units as locs. Default=-1.0

    ymax    [scalar] Upper limit along the y-axis for the aperture kernel 
            footprint. Applicable in case of rectangular or square apertures. 
            Same units as locs. Default=1.0
    
    rotangle
            [scalar] Angle (in radians) by which the principal axis of the 
            aperture is rotated counterclockwise east of sky frame. 
            Default=0.0

    pointing_center
            [numpy array] Pointing center to phase the aperture illumination to.
            Must be a 2-element array denoting the x- and y-direction cosines
            that obeys rules of direction cosines. Default=None (zenith)

    ----------------------------------------------------------------------------
    """

    inpdict = inputcheck(locs, wavelength=wavelength, xmin=xmin, xmax=xmax,
                         ymin=ymin, ymax=ymax, rotangle=rotangle,
                         pointing_center=pointing_center)
    locs = inpdict['locs']
    wavelength = inpdict['wavelength']
    xmin = inpdict['xmin']
    xmax = inpdict['xmax']    
    ymin = inpdict['ymin']
    ymax = inpdict['ymax']
    rotangle = inpdict['rotangle']
    pointing_center = inpdict['pointing_center']

    kern = NP.zeros(locs.shape[0], dtype=NP.complex64)
    if ymax-ymin > xmax-xmin:
        rotangle = rotangle + NP.pi/2

    # Rotate all locations by -rotangle to get x- and y-values relative to
    # aperture frame

    rotmat = NP.asarray([[NP.cos(-rotangle), -NP.sin(-rotangle)],
                         [NP.sin(-rotangle),  NP.cos(-rotangle)]])
    locs = NP.dot(locs, rotmat.T)

    ind = NP.logical_and((locs[:,0] >= xmin) & (locs[:,0] <= xmax), (locs[:,1] >= ymin) & (locs[:,1] <= ymax))
    kern[ind] = NP.exp(-1j * 2*NP.pi/wavelength * NP.dot(locs, pointing_center.T))
    
    eps = 1e-10
    if NP.all(NP.abs(kern.imag) < eps):
        kern = kern.real

    return kern

################################################################################    
