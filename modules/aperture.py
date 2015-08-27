import numpy as NP
import scipy.constants as FCNST
import lookup_operations as LKP

################################################################################    

def parmscheck(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0,rmin=0.0, rmax=1.0,
               rotangle=0.0, pointing_center=None):

    """
    ----------------------------------------------------------------------------
    Checks aperture parameters for compatibility for analytic aperture kernel 
    estimation

    xmin    [scalar] Lower limit along the x-axis for the aperture kernel 
            footprint. Applicable in case of rectangular or square apertures. 
            Default=-1.0

    xmax    [scalar] Upper limit along the x-axis for the aperture kernel 
            footprint. Applicable in case of rectangular or square apertures. 
            Default=1.0

    ymin    [scalar] Lower limit along the y-axis for the aperture kernel 
            footprint. Applicable in case of rectangular apertures. 
            Default=-1.0

    ymax    [scalar] Upper limit along the y-axis for the aperture kernel 
            footprint. Applicable in case of rectangular apertures. 
            Default=1.0

    rmin    [scalar] Lower limit along the radial direction for the aperture 
            kernel footprint. Applicable in case of circular apertures.  
            Default=0.0

    rmax    [scalar] Upper limit along the radial direction for the aperture 
            kernel footprint. Applicable in case of circular apertures.  
            Default=1.0

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
    'xmin'  [scalar] Corrected lower limit along the x-axis for the aperture 
            kernel footprint. Applicable in case of rectangular or square 
            apertures. 
    'xmax'  [scalar] Corrected upper limit along the x-axis for the aperture 
            kernel footprint. Applicable in case of rectangular or square 
            apertures. 
    'ymin'  [scalar] Corrected lower limit along the y-axis for the aperture 
            kernel footprint. Applicable in case of rectangular
            apertures. 
    'ymax'  [scalar] Corrected upper limit along the y-axis for the aperture 
            kernel footprint. Applicable in case of rectangular
            apertures. 
    'rmin'  [scalar] Corrected lower limit along the radial direction for the 
            aperture kernel footprint. Applicable in case of circular apertures. 
            
    'rmax'  [scalar] Corrected upper limit along the radial direction for the 
            aperture kernel footprint. Applicable in case of circular apertures. 
            
    'rotangle'
            [scalar] Angle (in radians) by which the principal axis of the 
            aperture is rotated counterclockwise east of sky frame. 
    'pointing_center'
            [numpy array] Corrected pointing center to phase the aperture 
            illumination to. It is a 2-element array denoting the x- and 
            y-direction cosines
    ----------------------------------------------------------------------------
    """

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
    
    outdict = {}
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
            footprint. Applicable in case of rectangular apertures. 
            Same units as locs. Default=-1.0

    ymax    [scalar] Upper limit along the y-axis for the aperture kernel 
            footprint. Applicable in case of rectangular apertures. 
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
            kernel footprint. Applicable in case of rectangular
            apertures. Same units as locs
    'ymax'  [scalar] Corrected upper limit along the y-axis for the aperture 
            kernel footprint. Applicable in case of rectangular
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
    
    parmsdict = parmscheck(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                           rotangle=rotangle, pointing_center=pointing_center)
    
    outdict['xmin'] = parmsdict['xmin']
    outdict['xmax'] = parmsdict['xmax']
    outdict['ymin'] = parmsdict['ymin']
    outdict['ymax'] = parmsdict['ymax']
    outdict['rmin'] = parmsdict['rmin']
    outdict['rmax'] = parmsdict['rmax']
    outdict['rotangle'] = parmsdict['rotangle']
    outdict['pointing_center'] = parmsdict['pointing_center']

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
            all locations. If an array is provided, number of wavelengths must 
            equal number of locations. Same units as locs. Default=1.0

    xmin    [scalar] Lower limit along the x-axis for the aperture kernel 
            footprint. Same units as locs. Default=-1.0

    xmax    [scalar] Upper limit along the x-axis for the aperture kernel 
            footprint. Same units as locs. Default=1.0

    ymin    [scalar] Lower limit along the y-axis for the aperture kernel 
            footprint. Same units as locs. Default=-1.0

    ymax    [scalar] Upper limit along the y-axis for the aperture kernel 
            footprint. Same units as locs. Default=1.0
    
    rotangle
            [scalar] Angle (in radians) by which the principal axis of the 
            aperture is rotated counterclockwise east of sky frame. 
            Default=0.0

    pointing_center
            [numpy array] Pointing center to phase the aperture illumination to.
            Must be a 2-element array denoting the x- and y-direction cosines
            that obeys rules of direction cosines. Default=None (zenith)

    Outputs:

    kern    [numpy array] complex aperture kernel with a value for each 
            location in the input. 
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
    kern[ind] = NP.exp(-1j * 2*NP.pi/wavelength[ind] * NP.dot(locs[ind,:], pointing_center.T).ravel())
    
    eps = 1e-10
    if NP.all(NP.abs(kern.imag) < eps):
        kern = kern.real

    return kern

################################################################################

def square(locs, wavelength=1.0, xmin=-1.0, xmax=1.0, rotangle=0.0,
           pointing_center=None):

    """
    ----------------------------------------------------------------------------
    Square aperture kernel estimation

    Inputs:

    locs    [numpy array] locations at which aperture kernel is to be estimated. 
            Must be a Mx2 numpy array where M is the number of locations, x- and 
            y-locations are stored in the first and second columns respectively.
            The units can be arbitrary but preferably that of distance. Must be 
            specified, no defaults

    wavelength
            [scalar or numpy array] Wavelength of the radiation. If it is a 
            scalar or numpy array of size 1, it is assumed to be identical for 
            all locations. If an array is provided, number of wavelengths must 
            equal number of locations. Same units as locs. Default=1.0

    xmin    [scalar] Lower limit for the aperture kernel footprint. Same 
            units as locs. Default=-1.0

    xmax    [scalar] Upper limit for the aperture kernel footprint. Same 
            units as locs. Default=1.0

    rotangle
            [scalar] Angle (in radians) by which the principal axis of the 
            aperture is rotated counterclockwise east of sky frame. 
            Default=0.0

    pointing_center
            [numpy array] Pointing center to phase the aperture illumination to.
            Must be a 2-element array denoting the x- and y-direction cosines
            that obeys rules of direction cosines. Default=None (zenith)

    Outputs:

    kern    [numpy array] complex aperture kernel with a value for each 
            location in the input. 
    ----------------------------------------------------------------------------
    """

    kern = rect(locs, wavelength=wavelength, xmin=xmin, xmax=xmax, ymin=xmin,
                ymax=xmax, rotangle=rotangle, pointing_center=pointing_center)

    return kern

################################################################################

def circular(locs, wavelength=1.0, rmin=0.0, rmax=1.0, pointing_center=None):

    """
    ----------------------------------------------------------------------------
    Uniform circular aperture kernel estimation

    Inputs:

    locs    [numpy array] locations at which aperture kernel is to be estimated. 
            Must be a Mx2 numpy array where M is the number of locations, x- and 
            y-locations are stored in the first and second columns respectively.
            The units can be arbitrary but preferably that of distance. Must be
            specified, no defaults.

    wavelength
            [scalar or numpy array] Wavelength of the radiation. If it is a 
            scalar or numpy array of size 1, it is assumed to be identical for 
            all locations. If an array is provided, number of wavelengths must 
            equal number of locations. Same units as locs. Default=1.0

    rmin    [scalar] Lower limit along the radial direction for the aperture 
            kernel footprint. Applicable in case of circular apertures. Same 
            units as locs. Default=0.0

    rmax    [scalar] Upper limit along the radial direction for the aperture 
            kernel footprint. Applicable in case of circular apertures. Same 
            units as locs. Default=1.0

    pointing_center
            [numpy array] Pointing center to phase the aperture illumination 
            to. Must be a 2-element array denoting the x- and y-direction 
            cosines that obeys rules of direction cosines. Default=None 
            (zenith)

    Outputs:

    kern    [numpy array] complex aperture kernel with a value for each 
            location in the input. 
    ----------------------------------------------------------------------------
    """
    
    inpdict = inputcheck(locs, wavelength=wavelength, rmin=rmin, rmax=rmax,
                         pointing_center=pointing_center)
    locs = inpdict['locs']
    wavelength = inpdict['wavelength']
    rmin = inpdict['rmin']
    rmax = inpdict['rmax']
    pointing_center = inpdict['pointing_center']

    kern = NP.zeros(locs.shape[0], dtype=NP.complex64)

    radii = NP.sqrt(NP.sum(locs**2, axis=1))
    ind = (radii >= rmin) & (radii <= rmax) 
    kern[ind] = NP.exp(-1j * 2*NP.pi/wavelength[ind] * NP.dot(locs[ind,:], pointing_center.T).ravel())
    
    eps = 1e-10
    if NP.all(NP.abs(kern.imag) < eps):
        kern = kern.real

    return kern
    
################################################################################

class AntennaAperture(object):

    """
    ----------------------------------------------------------------------------
    Class to manage collective information on a group of antennas.

    Attributes:

    kernel_type [dictionary] denotes whether the kernel is analytic or based on
                a lookup table. It has two keys 'P1' and 'P2' - one for each 
                polarization. Under each key the allowed values are 'func' and
                'lookup' (default). If specified as None during initialization,
                it is set to 'lookup' under both polarizations.

    shape       [dictionary] denotes the shape of the aperture. It has two keys 
                'P1' and 'P2' - one for each polarization. Under each key the 
                allowed values are under each polarization are 'rect', 'square',
                'circular' or None. These apply only if the corresponding 
                kernel_type for the polarization is set to 'func' else the 
                shape will be set to None.

    xmin        [dictionary] Lower limit along the x-axis for the aperture 
                kernel footprint. Applicable in case of rectangular or square 
                apertures. It has two keys 'P1' and 'P2' - one for each 
                polarization. The value (default=-1.0) held by each key is a 
                scalar

    xmax        [dictionary] Upper limit along the x-axis for the aperture 
                kernel footprint. Applicable in case of rectangular or square 
                apertures. It has two keys 'P1' and 'P2' - one for each 
                polarization. The value (default=1.0) held by each key is a 
                scalar

    ymin        [dictionary] Lower limit along the y-axis for the aperture 
                kernel footprint. Applicable in case of rectangular 
                apertures. It has two keys 'P1' and 'P2' - one for each 
                polarization. The value (default=-1.0) held by each key is a 
                scalar

    ymax        [dictionary] Upper limit along the y-axis for the aperture 
                kernel footprint. Applicable in case of rectangular 
                apertures. It has two keys 'P1' and 'P2' - one for each 
                polarization. The value (default=1.0) held by each key is a 
                scalar

    rmin        [dictionary] Lower limit along the radial axis for the aperture 
                kernel footprint. Applicable in case of circular 
                apertures. It has two keys 'P1' and 'P2' - one for each 
                polarization. The value (default=0.0) held by each key is a 
                scalar

    rmax        [dictionary] Upper limit along the radial axis for the aperture 
                kernel footprint. Applicable in case of circular
                apertures. It has two keys 'P1' and 'P2' - one for each 
                polarization. The value (default=1.0) held by each key is a 
                scalar

    rotangle    [dictionary] Angle (in radians) by which the principal axis of the 
                aperture is rotated counterclockwise east of sky frame. 
                Applicable in case of rectangular, square and elliptical 
                apertures. It has two keys 'P1' and 'P2' - one for each 
                polarization. The value (default=0.0) held by each key is a 
                scalar

    wtsposxy    [dictionary] two-dimensional locations of the gridding weights 
                in wts for each polarization under keys 'P1' and 'P2'. The 
                locations are in ENU coordinate system as a list of 2-column 
                numpy arrays in units of distance. 

    wtsxy       [dictionary] The gridding weights for antenna. Different 
                polarizations 'P1' and 'P2' form the keys of this dictionary. 
                These values are in general complex. Under each key, the values 
                are maintained as a numpy array of complex antenna weights 
                corresponding to positions in the lookup table. It should be of 
                same size as the number of rows in wtsposxy
    
    Member functions:

    __init__()  Initializes an instance of class AntennaAperture which manages
                information about an antenna aperture

    compute()   Estimates the kernel for given locations based on the aperture 
                attributes

    Read the member function docstrings for details.
    ----------------------------------------------------------------------------
    """

    def __init__(self, kernel_type=None, shape=None, parms=None, lkpinfo=None):

        """
        ------------------------------------------------------------------------
        Initializes an instance of class AntennaAperture which manages
        information about an antenna aperture

        Class attributes initialized are:
        kernel_type, shape, xmin, xmax, ymin, ymax, rmin, emax, rotangle, 
        wtsposxy, wtsxy

        Read docstring of class AntennaAperture for details on these 
        attributes.

        Inputs:

        kernel_type [dictionary] denotes whether the kernel is analytic or based 
                    on a lookup table. It has two keys 'P1' and 'P2' - one for 
                    each polarization. Under each key the allowed values are 
                    'func' and 'lookup' (default). If specified as None,
                    it is set to 'lookup' under both polarizations.
    
        shape       [dictionary] denotes the shape of the aperture. It has two 
                    keys 'P1' and 'P2' - one for each polarization. Under each 
                    key the allowed values are under each polarization are 
                    'rect', 'square', 'circular' or None. These apply only if 
                    the corresponding kernel_type for the polarization is set 
                    to 'func' else the shape will be set to None.

        parms       [dictionary] denotes parameters of the aperture shape. It 
                    has two keys 'P1' and 'P2' - one for each polarization. 
                    Under each of these keys is another dictionary with the 
                    following keys and information:
                    'xmin'  [scalar] Lower limit along the x-axis for the 
                            aperture kernel footprint. Applicable in case of 
                            rectangular or square apertures. Default=-1.0
                    'xmax'  [scalar] Upper limit along the x-axis for the 
                            aperture kernel footprint. Applicable in case of 
                            rectangular or square apertures. Default=1.0
                    'ymin'  [scalar] Lower limit along the y-axis for the 
                            aperture kernel footprint. Applicable in case of 
                            rectangular apertures. Default=-1.0
                    'ymax'  [scalar] Upper limit along the y-axis for the 
                            aperture kernel footprint. Applicable in case of 
                            rectangular apertures. Default=1.0
                    'rmin'  [scalar] Lower limit along radial axis for the 
                            aperture kernel footprint. Applicable in case of 
                            circualr apertures. Default=0.0
                    'rmax'  [scalar] Upper limit along radial axis for the 
                            aperture kernel footprint. Applicable in case of 
                            circular apertures. Default=1.0
                    'rotangle'
                            [scalar] Angle (in radians) by which the principal 
                            axis of the aperture is rotated counterclockwise 
                            east of sky frame. Applicable in case of 
                            rectangular, square and elliptical apertures. It 
                            has two keys 'P1' and 'P2' - one for each 
                            polarization. The value (default=0.0) held by each 
                            key is a scalar

        lookup      [dicitonary] consists of weights information for each of 
                    the two polarizations under keys 'P1' and 'P2'. Each of 
                    the values under the keys is a string containing the full
                    path to a filename that contains the positions and 
                    weights for the antenna field illumination in the form of 
                    a lookup table as columns (x-loc [float], y-loc 
                    [float], wts[real], wts[imag if any]). 
        ------------------------------------------------------------------------
        """

        if kernel_type is None:
            kernel_type = {}
            for pol in ['P1','P2']:
                kernel_type[pol] = 'lookup'
        elif isinstance(kernel_type, dict):
            for pol in ['P1','P2']:
                if pol not in kernel_type:
                    kernel_type[pol] = 'lookup'
                elif kernel_type[pol] not in ['lookup', 'func']:
                    raise ValueError('Invalid value specified for kernel_type under polarization {0}'.format(pol))
        else:
            raise TypeError('kernel_type must be a dictionary')

        if shape is None:
            shape = {}
            for pol in ['P1','P2']:
                if kernel_type[pol] == 'lookup':
                    shape[pol] = None
                else:
                    shape[pol] = 'circular'
        elif isinstance(shape, dict):
             for pol in ['P1','P2']:
                if pol not in shape:
                    if kernel_type[pol] == 'lookup':
                        shape[pol] = None
                    else:
                        shape[pol] = 'circular'
                else:
                    if kernel_type[pol] != 'func':
                        raise ValueError('Values specified in kernel_type and shape are incompatible')
                    elif shape[pol] not in ['rect', 'square', 'circular']:
                        raise ValueError('Invalid value specified for shape under polarization {0}'.format(pol))
        else:
            raise TypeError('Aperture kernel shape must be a dictionary')
            
        if parms is None:
            parms = {}
            for pol in ['P1','P2']:
                parms[pol] = {}
                parms[pol]['xmin'] = -1.0
                parms[pol]['ymin'] = -1.0
                parms[pol]['rmin'] = 0.0
                parms[pol]['xmax'] = 1.0
                parms[pol]['ymax'] = 1.0
                parms[pol]['rmax'] = 1.0
                parms[pol]['rotangle'] = 0.0
        elif isinstance(parms, dict):
            for pol in ['P1','P2']:
                if pol not in parms:
                    parms[pol] = {}
                    parms[pol]['xmin'] = -1.0
                    parms[pol]['ymin'] = -1.0
                    parms[pol]['rmin'] = 0.0
                    parms[pol]['xmax'] = 1.0
                    parms[pol]['ymax'] = 1.0
                    parms[pol]['rmax'] = 1.0
                    parms[pol]['rotangle'] = 0.0
                elif isinstance(parms[pol], dict):
                    if 'xmin' not in parms[pol]: parms[pol]['xmin'] = -1.0
                    if 'ymin' not in parms[pol]: parms[pol]['ymin'] = -1.0
                    if 'rmin' not in parms[pol]: parms[pol]['rmin'] = 0.0
                    if 'xmax' not in parms[pol]: parms[pol]['xmax'] = 1.0
                    if 'ymax' not in parms[pol]: parms[pol]['ymax'] = 1.0
                    if 'rmax' not in parms[pol]: parms[pol]['rmax'] = 1.0
                    if 'rotangle' not in parms[pol]: parms[pol]['rotangle'] = 0.0
                else:
                    raise TypeError('Aperture parameters under polarization {0} must be a dictionary'.format(pol))
        else:
            raise TypeError('Aperture kernel parameters must be a dictionary')

        self.kernel_type = {}
        self.shape = {}
        self.xmin = {}
        self.ymin = {}
        self.rmin = {}
        self.xmax = {}
        self.ymax = {}
        self.rmax = {}
        self.rotangle = {}
            
        for pol in ['P1', 'P2']:
            self.kernel_type[pol] = kernel_type[pol]
            self.shape[pol] = shape[pol]
            parmsdict = parmscheck(xmin=parms[pol]['xmin'], xmax=parms[pol]['xmax'], ymin=parms[pol]['ymin'], ymax=parms[pol]['ymax'], rmin=parms[pol]['rmin'], rmax=parms[pol]['rmax'], rotangle=parms[pol]['rotangle'])
            self.xmin[pol] = parmsdict['xmin']
            self.ymin[pol] = parmsdict['ymin']
            self.rmin[pol] = parmsdict['rmin']
            self.xmax[pol] = parmsdict['xmax']
            self.ymax[pol] = parmsdict['ymax']
            self.rmax[pol] = parmsdict['rmax']
            self.rotangle[pol] = parmsdict['rotangle']

        self.wtsposxy = {}
        self.wtsxy = {}
        for pol in ['P1', 'P2']:
            self.wtsposxy[pol] = None
            self.wtsxy[pol] = None                
            if lkpinfo is not None:
                if pol in lkpinfo:
                    lkpdata = LKP.read_lookup(lkpinfo[pol]['file'])
                    self.wtsposxy[pol] = NP.hstack((lkpdata[0].reshape(-1,1),lkpdata[1].reshape(-1,1)))
                    self.wtsxy[pol] = lkpdata[2]
                    if lkpdata.shape[1] == 4:  # Read in the imaginary part
                        self.wtsxy[pol] += 1j * lkpdata[3]

    ############################################################################

    def compute(self, locs, wavelength=1.0, pointing_center=None, pol=None):

        """
        ------------------------------------------------------------------------
        Estimates the kernel for given locations based on the aperture 
        attributes

        Inputs:

        locs    [numpy array] locations at which aperture kernel is to be 
                estimated. Must be a Mx2 numpy array where M is the number of 
                locations, x- and y-locations are stored in the first and second 
                columns respectively. The units can be arbitrary but preferably 
                that of distance. Must be specified, no defaults.

        wavelength
                [scalar or numpy array] Wavelength of the radiation. If it is a 
                scalar or numpy array of size 1, it is assumed to be identical 
                for all locations. If an array is provided, it must be of same 
                size as the number of locations at which the aperture kernel is 
                to be estimated. Same units as locs. Default=1.0

        pointing_center
                [numpy array] Pointing center to phase the aperture illumination 
                to. Must be a 2-element array denoting the x- and y-direction 
                cosines that obeys rules of direction cosines. Default=None 
                (zenith)

        pol     [string or list] The polarization for which the kernel is to be 
                estimated. Can be set to 'P1' or 'P2' or  list containing both. 
                If set to None, kernel is estimated for all the polarizations. 
                Default=None

        Outputs:

        Dictionary containing two keys 'P1' and 'P2' - one for each 
        polarization. Under each of these keys, the kernel is returned as a 
        numpy array of possibly complex values for the specified locations
        ------------------------------------------------------------------------
        """

        if pol is None:
            pol = ['P1', 'P2']
        elif not isinstance(pol, list):
            if pol not in ['P1', 'P2']:
                raise ValueError('Invalid value specified for pol')
            pol = [pol]
        else:
            pol = set(pol)
            p = [item for item in pol if item in ['P1', 'P2']]
            pol = p

        kern = {}
        for p in pol:
            kern[p] = None
            if self.shape[p] is not None:
                if self.shape[p] == 'rect':
                    kern[p] = rect(locs, wavelength=wavelength, xmin=self.xmin[p],
                                   xmax=self.xmax[p], ymin=self.ymin[p],
                                   ymax=self.ymax[p], rotangle=self.rotangle[p],
                                   pointing_center=pointing_center)
                elif self.shape[p] == 'square':
                    kern[p] = square(locs, wavelength=wavelength,
                                     xmin=self.xmin[p], xmax=self.xmax[p],
                                     rotangle=self.rotangle[p],
                                     pointing_center=pointing_center)
                elif self.shape[p] == 'circular':
                    kern[p] = circular(locs, wavelength=wavelength,
                                       rmin=self.rmin[p], rmax=self.rmax[p],
                                       pointing_center=pointing_center)
                else:
                    raise ValueError('The analytic kernel shape specified in the shape attribute is not currently supported')

        return kern
            
    ############################################################################

        
        
            
        
            
        
