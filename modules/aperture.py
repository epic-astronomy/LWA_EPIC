import numpy as NP
import lookup_operations as LKP

################################################################################

def parmscheck(xmax=1.0, ymax=1.0, rmin=0.0, rmax=1.0, rotangle=0.0,
               pointing_center=None):

    """
    ----------------------------------------------------------------------------
    Checks aperture parameters for compatibility for analytic aperture kernel 
    estimation

    xmax    [scalar] Upper limit along the x-axis for the original aperture 
            kernel footprint. Applicable in case of original rectangular or 
            square apertures. Default=1.0. Lower limit along the x-axis is set 
            to -xmax. Length of the original rectangular footprint is 2*xmax

    ymax    [scalar] Upper limit along the y-axis for the original aperture 
            kernel footprint. Applicable in case of original rectangular 
            apertures. Default=1.0. Lower limit along the y-axis is set to 
            -ymax. Breadth of the original rectangular footprint is 2*ymax

    rmin    [scalar] Lower limit along the radial direction for the original 
            aperture kernel footprint. Applicable in case of original circular 
            apertures. Default=0.0

    rmax    [scalar] Upper limit along the radial direction for the original 
            aperture kernel footprint. Applicable in case of original circular 
            apertures. Default=1.0

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
    'xmax'  [scalar] Corrected upper limit along the x-axis for the original 
            aperture kernel footprint. Applicable in case of original 
            rectangular or square apertures. 
    'ymax'  [scalar] Corrected upper limit along the y-axis for the original 
            aperture kernel footprint. Applicable in case of original 
            rectangular apertures. 
    'rmin'  [scalar] Corrected lower limit along the radial direction for the 
            original aperture kernel footprint. Applicable in case of original 
            circular apertures. 
            
    'rmax'  [scalar] Corrected upper limit along the radial direction for the 
            original aperture kernel footprint. Applicable in case of original 
            circular apertures. 
            
    'rotangle'
            [scalar] Angle (in radians) by which the principal axis of the 
            aperture is rotated counterclockwise east of sky frame. 
    'pointing_center'
            [numpy array] Corrected pointing center to phase the aperture 
            illumination to. It is a 2-element array denoting the x- and 
            y-direction cosines
    ----------------------------------------------------------------------------
    """

    if not isinstance(xmax, (int,float)):
        raise TypeError('xmax must be a scalar')
    else:
        xmax = float(xmax)

    if not isinstance(ymax, (int,float)):
        raise TypeError('ymax must be a scalar')
    else:
        ymax = float(ymax)

    if xmax <= 0.0:
        raise ValueError('xmax must be positive')
    if ymax <= 0.0:
        raise ValueError('ymax must be positive')

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
    outdict['xmax'] = xmax
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

def inputcheck(locs, wavelength=1.0, xmax=1.0, ymax=1.0, rmin=0.0, rmax=1.0,
               rotangle=0.0, pointing_center=None):

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

    xmax    [scalar] Upper limit along the x-axis for the original aperture 
            kernel footprint. Applicable in case of original rectangular or 
            square apertures. Same units as locs. Default=1.0. Lower limit along 
            the x-axis is set to -xmax. Length of the original rectangular 
            footprint is 2*xmax

    ymax    [scalar] Upper limit along the y-axis for the original aperture 
            kernel footprint. Applicable in case of original rectangular 
            apertures. Same units as locs. Default=1.0. Lower limit along the 
            y-axis is set to -ymax. Breadth of the original rectangular 
            footprint is 2*ymax

    rmin    [scalar] Lower limit along the radial direction for the original 
            aperture kernel footprint. Applicable in case of original circular 
            apertures. Same units as locs. Default=0.0

    rmax    [scalar] Upper limit along the radial direction for the original 
            aperture kernel footprint. Applicable in case of original circular 
            apertures. Same units as locs. Default=1.0

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
    'xmax'  [scalar] Corrected upper limit along the x-axis for the original 
            aperture kernel footprint. Applicable in case of original 
            rectangular or square apertures. Same units as locs
    'ymax'  [scalar] Corrected upper limit along the y-axis for the original 
            aperture kernel footprint. Applicable in case of original 
            rectangular apertures. Same units as locs
    'rmin'  [scalar] Corrected lower limit along the radial direction for the 
            original aperture kernel footprint. Applicable in case of original 
            circular apertures. Same units as locs
    'rmax'  [scalar] Corrected upper limit along the radial direction for the 
            original aperture kernel footprint. Applicable in case of original 
            circular apertures. Same units as locs
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
        if wavelength.size == 1:
            wavelength = wavelength + NP.zeros(locs.shape[0])

    outdict['wavelength'] = wavelength
    
    parmsdict = parmscheck(xmax=xmax, ymax=ymax, rmin=rmin, rmax=rmax,
                           rotangle=rotangle, pointing_center=pointing_center)
    
    outdict['xmax'] = parmsdict['xmax']
    outdict['ymax'] = parmsdict['ymax']
    outdict['rmin'] = parmsdict['rmin']    
    outdict['rmax'] = parmsdict['rmax']
    outdict['rotangle'] = parmsdict['rotangle']
    outdict['pointing_center'] = parmsdict['pointing_center']

    return outdict
        
################################################################################

def rect(locs, wavelength=1.0, xmax=1.0, ymax=1.0, rotangle=0.0,
         pointing_center=None):

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

    xmax    [scalar] Upper limit along the x-axis for the aperture kernel 
            footprint. Same units as locs. Default=1.0. Lower limit along the 
            x-axis is set to -xmax. Length of the rectangular footprint is 
            2*xmax

    ymax    [scalar] Upper limit along the y-axis for the aperture kernel 
            footprint. Same units as locs. Default=1.0. Lower limit along the 
            y-axis is set to -ymax. Length of the rectangular footprint is 
            2*ymax
    
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

    inpdict = inputcheck(locs, wavelength=wavelength, xmax=xmax, ymax=ymax,
                         rotangle=rotangle, pointing_center=pointing_center)
    locs = inpdict['locs']
    wavelength = inpdict['wavelength']
    xmax = inpdict['xmax']    
    ymax = inpdict['ymax']
    xmin = -xmax
    ymin = -ymax
    rotangle = inpdict['rotangle']
    pointing_center = inpdict['pointing_center']

    kern = NP.zeros(locs.shape[0], dtype=NP.complex64)
    if ymax > xmax:
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

def square(locs, wavelength=1.0, xmax=1.0, rotangle=0.0,
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

    xmax    [scalar] Upper limit for the aperture kernel footprint. Same 
            units as locs. Default=1.0. Lower limit along the x-axis is set to 
            -xmax. Length of the square footprint is 2*xmax

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

    kern = rect(locs, wavelength=wavelength, xmax=xmax, ymax=xmax,
                rotangle=rotangle, pointing_center=pointing_center)

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

def auto_convolved_rect(locs, wavelength=1.0, xmax=1.0, ymax=1.0, rotangle=0.0,
                        pointing_center=None):

    """
    ----------------------------------------------------------------------------
    Aperture kernel estimation from rectangular auto-convolution 

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

    xmax    [scalar] Upper limit along the x-axis for the aperture kernel 
            footprint for the original rectangle. Same units as locs. 
            Default=1.0. Lower limit along the x-axis for the original 
            rectangle is set to -xmax. Length of the original rectangular 
            footprint is 2*xmax while that of the auto-convolved function is
            4*xmax

    ymax    [scalar] Upper limit along the y-axis for the aperture kernel 
            footprint for the original rectangle. Same units as locs. 
            Default=1.0. Lower limit along the y-axis of the original 
            rectangle is set to -ymax. Length of the original rectangular 
            footprint is 2*ymax while that of the auto-convolved function is
            4*ymax
    
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

    inpdict = inputcheck(locs, wavelength=wavelength, xmax=xmax, ymax=ymax,
                         rotangle=rotangle, pointing_center=pointing_center)
    locs = inpdict['locs']
    wavelength = inpdict['wavelength']
    xmax = inpdict['xmax']    
    ymax = inpdict['ymax']
    xmin = -xmax
    ymin = -ymax
    rotangle = inpdict['rotangle']
    pointing_center = inpdict['pointing_center']

    kern = NP.zeros(locs.shape[0], dtype=NP.complex64)
    if ymax > xmax:
        rotangle = rotangle + NP.pi/2

    # Rotate all locations by -rotangle to get x- and y-values relative to
    # aperture frame

    rotmat = NP.asarray([[NP.cos(-rotangle), -NP.sin(-rotangle)],
                         [NP.sin(-rotangle),  NP.cos(-rotangle)]])
    locs = NP.dot(locs, rotmat.T)

    ind = NP.logical_and((locs[:,0] >= 2*xmin) & (locs[:,0] <= 2*xmax), (locs[:,1] >= 2*ymin) & (locs[:,1] <= 2*ymax))
    amp_rect = 1.0
    overlap = (2*xmax - NP.abs(locs[ind,0])) * (2*ymax - NP.abs(locs[ind,1]))
    kern[ind] = amp_rect**2 * overlap * NP.exp(-1j * 2*NP.pi/wavelength[ind] * NP.dot(locs[ind,:], pointing_center.T).ravel())
    
    eps = 1e-10
    if NP.all(NP.abs(kern.imag) < eps):
        kern = kern.real

    return kern

################################################################################

def auto_convolved_square(locs, wavelength=1.0, xmax=1.0, ymax=1.0,
                          rotangle=0.0, pointing_center=None):

    """
    ----------------------------------------------------------------------------
    Aperture kernel estimation from square auto-convolution 

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

    xmax    [scalar] Upper limit along the x-axis for the aperture kernel 
            footprint for the original square. Same units as locs. 
            Default=1.0. Lower limit along the x-axis for the original 
            square is set to -xmax. Length of the original square 
            footprint is 2*xmax while that of the auto-convolved function is
            4*xmax

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

    kern = auto_convolved_rect(locs, wavelength=wavelength, xmax=xmax,
                               ymax=xmax, rotangle=rotangle,
                               pointing_center=pointing_center)

    return kern

################################################################################

def auto_convolved_circular(locs, wavelength=1.0, rmax=1.0,
                            pointing_center=None):

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

    rmax    [scalar] Upper limit along the radial direction for the aperture 
            kernel of the original circular footprint. Same units as locs. 
            Default=1.0

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
    
    inpdict = inputcheck(locs, wavelength=wavelength, rmin=0.0, rmax=rmax,
                         pointing_center=pointing_center)
    locs = inpdict['locs']
    wavelength = inpdict['wavelength']
    rmin = inpdict['rmin']
    rmax = inpdict['rmax']
    pointing_center = inpdict['pointing_center']

    kern = NP.zeros(locs.shape[0], dtype=NP.complex64)

    radii = NP.sqrt(NP.sum(locs**2, axis=1))
    cos_halftheta = 0.5 * radii / rmax
    theta = 2 * NP.arccos(cos_halftheta)
    ind = radii <= 2*rmax
    amp_circ = 1.0
    overlap = rmax**2 * (theta[ind] - NP.sin(theta[ind]))
    kern[ind] = amp_circ**2 * overlap * NP.exp(-1j * 2*NP.pi/wavelength[ind] * NP.dot(locs[ind,:], pointing_center.T).ravel())
    
    eps = 1e-10
    if NP.all(NP.abs(kern.imag) < eps):
        kern = kern.real

    return kern
    
################################################################################

class Aperture(object):

    """
    ----------------------------------------------------------------------------
    Class to manage collective information on aperture of an antenna or an
    interferometer

    Attributes:

    pol         [list] contains a list of polarizations. Two polarizations 
                ['P1', 'P2'] if the instance corresponds to antenna aperture or
                four cross-polarizations ['P11', 'P12', 'P21', 'P22'] if it 
                corresponds to an interferometer aperture

    kernel_type [dictionary] denotes whether the kernel is analytic or based on
                a lookup table. It has two or four keys (depending on attribute 
                pol) - one for each polarization. Under each key the allowed 
                values are 'func' and 'lookup' (default). If specified as None 
                during initialization, it is set to 'lookup' under all
                polarizations.

    shape       [dictionary] denotes the shape of the aperture. It has two or 
                four keys (depending on attribute pol) - one for each 
                polarization. Under each polarization key the allowed values 
                are 'rect', 'square', 'circular', 'auto_convolved_rect', 
                'auto_convolved_square', 'auto_convolved_circular' or None. 
                These apply only if the corresponding kernel_type for the 
                polarization is set to 'func' else the shape will be set to 
                None.

    xmax        [dictionary] Upper limit along the x-axis for the aperture 
                kernel footprint. Applicable in case of original rectangular 
                or square apertures. It has two or four keys (depending on 
                attribute pol) - one for each polarization. The value 
                (default=1.0) held by each key is a scalar. Lower limit along 
                the x-axis is set to -xmax. Length of the original 
                rectangular/square footprint is 2*xmax

    ymax        [dictionary] Upper limit along the y-axis for the aperture 
                kernel footprint. Applicable in case of original rectangular 
                apertures. It has two or four keys (depending on 
                attribute pol) - one for each polarization. The value 
                (default=1.0) held by each key is a scalar. Lower limit along 
                the y-axis is set to -ymax. Breadth of the original rectangular 
                footprint is 2*ymax

    rmin        [dictionary] Lower limit along the radial axis for the original 
                aperture kernel footprint. Applicable in case of original 
                circular apertures. It has two or four keys (depending on 
                attribute pol) - one for each polarization. The value 
                (default=0.0) held by each key is a scalar

    rmax        [dictionary] Upper limit along the radial axis for the original 
                aperture kernel footprint. Applicable in case of original 
                circular apertures. It has two or four keys (depending on 
                attribute pol) - one for each polarization. The value 
                (default=1.0) held by each key is a scalar

    rotangle    [dictionary] Angle (in radians) by which the principal axis of the 
                aperture is rotated counterclockwise east of sky frame. 
                Applicable in case of rectangular, square and elliptical 
                apertures. It has two or four keys (depending on 
                attribute pol) - one for each polarization. The value 
                (default=0.0) held by each key is a scalar

    lkpinfo     [dictionary] lookup table file location, one for each 
                polarization under the standard keys denoting polarization

    wtsposxy    [dictionary] two-dimensional locations of the gridding weights 
                in wts for each polarization key. The locations are in ENU 
                coordinate system as a list of 2-column numpy arrays in units 
                of distance. 

    wtsxy       [dictionary] The gridding weights for the aperture. Different 
                polarizations form the keys of this dictionary. 
                These values are in general complex. Under each key, the values 
                are maintained as a numpy array of complex aperture weights 
                corresponding to positions in the lookup table. It should be of 
                same size as the number of rows in wtsposxy
    
    Member functions:

    __init__()  Initializes an instance of class Aperture which manages
                information about an antenna or interferometer aperture

    compute()   Estimates the kernel for given locations based on the aperture 
                attributes

    Read the member function docstrings for details.
    ----------------------------------------------------------------------------
    """

    def __init__(self, pol_type='dual', kernel_type=None, shape=None,
                 parms=None, lkpinfo=None, load_lookup=True):

        """
        ------------------------------------------------------------------------
        Initializes an instance of class Aperture which manages
        information about an antenna or interferometer aperture

        Class attributes initialized are:
        pol, kernel_type, shape, xmax, ymax, rmin, emax, rotangle, 
        wtsposxy, wtsxy, lkpinfo

        Read docstring of class Aperture for details on these 
        attributes.

        Inputs:

        pol_type    [string] Specifies type of polarizations to be set. 
                    Currently accepted values are 'dual' (default) and 
                    'cross'. If set to 'dual', the attribute pol is set to
                    dual antenna polarizations ['P1', 'P2']. If set to 'cross'
                    attribute pol is set to cross-polarizations from antennas
                    ['P11', 'P12', 'P21', 'P22']

        kernel_type [dictionary] denotes whether the kernel is analytic or based 
                    on a lookup table. It has two or four keys (depending on 
                    attribute pol) - one for each polarization. Under each key 
                    the allowed values are 'func' and 'lookup' (default). If 
                    specified as None, it is set to 'lookup' under both 
                    polarizations.
    
        shape       [dictionary] denotes the shape of the aperture. It has two 
                    or four keys (depending on attribute pol) - one for each 
                    polarization. Under each key the allowed values are 
                    'rect', 'square', 'circular', 'auto_convolved_rect', 
                    'auto_convolved_square', 'auto_convolved_circular'  or None. 
                    These apply only if the corresponding kernel_type for the 
                    polarization is set to 'func' else the shape will be set to 
                    None.

        parms       [dictionary] denotes parameters of the original aperture 
                    shape. It has two or four keys (depending on attribute pol),
                    one for each polarization. Under each of these keys is 
                    another dictionary with the following keys and information:
                    'xmax'  [scalar] Upper limit along the x-axis for the 
                            original aperture kernel footprint. Applicable in 
                            case of original rectangular or square apertures. 
                            Lower limit along the x-axis is set to -xmax. 
                            Length of the original rectangular/square footprint 
                            is 2*xmax
                    'ymax'  [scalar] Upper limit along the y-axis for the 
                            original aperture kernel footprint. Applicable in 
                            case of original rectangular apertures. Default=1.0. 
                            Lower limit along the y-axis is set to -ymax. 
                            Breadth of the original rectangular footprint is 
                            2*ymax
                    'rmin'  [scalar] Lower limit along radial axis for the 
                            original aperture kernel footprint. Applicable in 
                            case of original circular apertures. Default=0.0
                    'rmax'  [scalar] Upper limit along radial axis for the 
                            original aperture kernel footprint. Applicable in 
                            case of original circular apertures. Default=1.0
                    'rotangle'
                            [scalar] Angle (in radians) by which the principal 
                            axis of the aperture is rotated counterclockwise 
                            east of sky frame. Applicable in case of 
                            rectangular, square and elliptical apertures. It 
                            has two keys 'P1' and 'P2' - one for each 
                            polarization. The value (default=0.0) held by each 
                            key is a scalar

        lkpinfo     [dictionary] consists of weights information for each of 
                    the polarizations under polarization keys. Each of 
                    the values under the keys is a string containing the full
                    path to a filename that contains the positions and 
                    weights for the aperture illumination in the form of 
                    a lookup table as columns (x-loc [float], y-loc 
                    [float], wts[real], wts[imag if any]). 

        load_lookup [boolean] If set to True (default), loads from the lookup 
                    table. If set to False, the values may be loaded later 
                    using member function compute()
        ------------------------------------------------------------------------
        """

        if pol_type not in ['dual', 'cross']:
            raise ValueError('Polarization type must be "dual" or "cross"')
        elif pol_type == 'dual':
            self.pol = ['P1', 'P2']
        else:
            self.pol = ['P11', 'P12', 'P21', 'P22']

        if kernel_type is None:
            kernel_type = {}
            for pol in self.pol:
                kernel_type[pol] = 'lookup'
        elif isinstance(kernel_type, dict):
            for pol in self.pol:
                if pol not in kernel_type:
                    kernel_type[pol] = 'lookup'
                elif kernel_type[pol] not in ['lookup', 'func']:
                    raise ValueError('Invalid value specified for kernel_type under polarization {0}'.format(pol))
        else:
            raise TypeError('kernel_type must be a dictionary')

        if shape is None:
            shape = {}
            for pol in self.pol:
                if kernel_type[pol] == 'lookup':
                    shape[pol] = None
                else:
                    shape[pol] = 'circular'
        elif isinstance(shape, dict):
             for pol in self.pol:
                if pol not in shape:
                    if kernel_type[pol] == 'lookup':
                        shape[pol] = None
                    else:
                        shape[pol] = 'circular'
                else:
                    if kernel_type[pol] != 'func':
                        raise ValueError('Values specified in kernel_type and shape are incompatible')
                    elif shape[pol] not in ['rect', 'square', 'circular', 'auto_convolved_rect', 'auto_convolved_square', 'auto_convolved_circular']:
                        raise ValueError('Invalid value specified for shape under polarization {0}'.format(pol))
        else:
            raise TypeError('Aperture kernel shape must be a dictionary')
            
        if parms is None:
            parms = {}
            for pol in self.pol:
                parms[pol] = {}
                parms[pol]['rmin'] = 0.0
                parms[pol]['xmax'] = 1.0
                parms[pol]['ymax'] = 1.0
                parms[pol]['rmax'] = 1.0
                parms[pol]['rotangle'] = 0.0
        elif isinstance(parms, dict):
            for pol in self.pol:
                if pol not in parms:
                    parms[pol] = {}
                    parms[pol]['rmin'] = 0.0
                    parms[pol]['xmax'] = 1.0
                    parms[pol]['ymax'] = 1.0
                    parms[pol]['rmax'] = 1.0
                    parms[pol]['rotangle'] = 0.0
                elif isinstance(parms[pol], dict):
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
        self.rmin = {}
        self.xmax = {}
        self.ymax = {}
        self.rmax = {}
        self.rotangle = {}
            
        for pol in self.pol:
            self.kernel_type[pol] = kernel_type[pol]
            self.shape[pol] = shape[pol]
            parmsdict = parmscheck(xmax=parms[pol]['xmax'], ymax=parms[pol]['ymax'], rmin=parms[pol]['rmin'], rmax=parms[pol]['rmax'], rotangle=parms[pol]['rotangle'])
            self.rmin[pol] = parmsdict['rmin']
            self.xmax[pol] = parmsdict['xmax']
            self.ymax[pol] = parmsdict['ymax']
            self.rmax[pol] = parmsdict['rmax']
            self.rotangle[pol] = parmsdict['rotangle']

        self.wtsposxy = {}
        self.wtsxy = {}
        self.lkpinfo = {}
        if lkpinfo is not None:
            if not isinstance(lkpinfo, dict):
                raise TypeError('Input parameter lkpinfo must be a dictionary')
            for pol in self.pol:
                self.wtsposxy[pol] = None
                self.wtsxy[pol] = None
                if pol in lkpinfo:
                    self.lkpinfo[pol] = lkpinfo[pol]
                    if load_lookup:
                        lkpdata = LKP.read_lookup(self.lkpinfo[pol])
                        self.wtsposxy[pol] = NP.hstack((lkpdata[0].reshape(-1,1),lkpdata[1].reshape(-1,1)))
                        self.wtsxy[pol] = lkpdata[2]
                        if len(lkpdata) == 4:  # Read in the imaginary part
                            self.wtsxy[pol] += 1j * lkpdata[3]

    ############################################################################

    def compute(self, locs, wavelength=1.0, pointing_center=None, pol=None,
                rmaxNN=None, load_lookup=False):

        """
        ------------------------------------------------------------------------
        Estimates the kernel for given locations based on the aperture 
        attributes for an analytic or lookup-based estimation

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
                estimated. Can be set to one or both of 'P1' and 'P2' for antenna 
                apertures or one or more or all of 'P11', 'P12', 'P21', 'P22' for
                interferometer apertures as a listk. If set to None, kernel is 
                estimated for all the polarizations. Default=None

        rmaxNN  [scalar] Search distance upper limit in case of kernel 
                estimation from a lookup table. Default=None means value in 
                attribute rmax is used.

        load_lookup
                [boolean] If set to True, loads from the lookup table. If set
                to False (default), uses the already loaded values during 
                initialization

        Outputs:

        Dictionary containing two or four keys (depending on attribute pol), 
        one for each polarization. Under each of these keys, the kernel 
        information is returned as a (complex) numpy array
        ------------------------------------------------------------------------
        """

        if pol is None:
            pol = self.pol
        elif not isinstance(pol, list):
            if pol not in self.pol:
                raise ValueError('Invalid value specified for pol')
            pol = [pol]
        else:
            pol = set(pol)
            p = [item for item in pol if item in self.pol]
            pol = p

        outdict = {}
        for p in pol:
            outdict[p] = None
            if self.kernel_type[p] == 'func':
                if self.shape[p] is not None:
                    if self.shape[p] == 'rect':
                        outdict[p] = rect(locs, wavelength=wavelength, xmax=self.xmax[p], ymax=self.ymax[p], rotangle=self.rotangle[p], pointing_center=pointing_center)
                    elif self.shape[p] == 'square':
                        outdict[p] = square(locs, wavelength=wavelength, xmax=self.xmax[p], rotangle=self.rotangle[p], pointing_center=pointing_center)
                    elif self.shape[p] == 'circular':
                        outdict[p] = circular(locs, wavelength=wavelength, rmin=self.rmin[p], rmax=self.rmax[p], pointing_center=pointing_center)
                    elif self.shape[p] == 'auto_convolved_rect':
                        outdict[p] = auto_convolved_rect(locs, wavelength=wavelength, xmax=self.xmax[p], ymax=self.ymax[p], rotangle=self.rotangle[p], pointing_center=pointing_center)
                    elif self.shape[p] == 'auto_convolved_square':
                        outdict[p] = auto_convolved_square(locs, wavelength=wavelength, xmax=self.xmax[p], rotangle=self.rotangle[p], pointing_center=pointing_center)
                    elif self.shape[p] == 'auto_convolved_circular':
                        outdict[p] = auto_convolved_circular(locs, wavelength=wavelength, rmin=self.rmin[p], rmax=self.rmax[p], pointing_center=pointing_center)
                    else:
                        raise ValueError('The analytic kernel shape specified in the shape attribute is not currently supported')
            else:
                if rmaxNN is None:
                    rmaxNN = self.rmax
                if not isinstance(rmaxNN, (int,float)):
                    raise TypeError('Input rmaxNN must be a scalar')
                else:
                    rmaxNN = float(rmaxNN)
                    if rmaxNN <= 0.0:
                        raise ValueError('Search radius upper limit must be positive')

                if p in self.lkpinfo:
                    if load_lookup:
                        lkpdata = LKP.read_lookup(self.lkpinfo[p])
                        self.wtsposxy[p] = NP.hstack((lkpdata[0].reshape(-1,1),lkpdata[1].reshape(-1,1)))
                        self.wtsxy[p] = lkpdata[2]
                        if len(lkpdata) == 4:  # Read in the imaginary part
                            self.wtsxy[p] += 1j * lkpdata[3]

                    # inpind, refind, distNN = LKP.find_1NN(self.wtsposxy[p], locs, distance_ULIM=rmaxNN, remove_oob=True)
                    inpind, nnval, distNN = LKP.lookup_1NN_new(self.wtsposxy[p], self.wtsxy[p], locs, distance_ULIM=rmaxNN, remove_oob=False)
                    outdict[p] = nnval

        return outdict
            
    ############################################################################

        
        
            
        
            
        
