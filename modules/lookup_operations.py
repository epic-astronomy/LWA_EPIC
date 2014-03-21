import numpy as NP
from astropy.io import ascii
try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT

#################################################################################

def gen_lookup(x, y, data, file):

    """
    -----------------------------------------------------------------------------
    Generate a lookup table given two-dimensional coordinates and values at these
    coordinates.

    Inputs:

    x          [numpy vector] x-coordinates for the lookup values.

    y          [numpy vector] y-coordinates for the lookup values

    data       [numpy vector] data values at the coordinates given by x and y. x,
               y, and data must all have the same size

    file       [string] Name of the output file to write the lookup table to. It 
               will be created as an ascii table with 'x', 'y', 'real_value' and 
               'imag_value' (if imagianry value present) being the column headers

    -----------------------------------------------------------------------------
    """

    try:
        x, y, data, file
    except NameError:
        raise NameError('To generate a lookup table, input parameters x, y, data, and file should be specified.')

    if (x.size != y.size) or (x.size != data.size):
        raise ValueError('x, y, and data must be of same size.')

    if not isinstance(file, str):
        raise TypeError('Input parameter file must be of string data type')

    if NP.iscomplexobj(data):
        try:
            ascii.write([x.ravel(), y.ravel(), data.ravel().real, data.ravel().imag], file, names=['x', 'y', 'real_value', 'imag_value'])
        except IOError:
            raise IOError('Could not write to specified file: '+file)
    else:
        try:
            ascii.write([x.ravel(), y.ravel(), data.ravel()], file, names=['x', 'y', 'real_value'])
        except IOError:
            raise IOError('Could not write to specified file: '+file)
    
#################################################################################

def read_lookup(file):

    """
    -----------------------------------------------------------------------------
    Read data from a lookup database.

    Inputs:

    file      [string] Input file containing the lookup data base.

    Outputs:

    [tuple] each element of the tuple is a numpy array. The elements in order are
            x-coordinates, y-coordinates, data value at those coordiantes. The
            data values are real or complex depending on whether the lookup table
            has an 'imag_value' column
    -----------------------------------------------------------------------------
    """

    if not isinstance(file, str):
        raise TypeError('Input parameter file must be of string data type')

    try:
        cols = ascii.read(file, data_start=1)
    except IOError:
        raise IOError('Could not read the specified file: '+file)

    if 'imag_value' in cols.colnames:
        return cols['x'].data, cols['y'].data, cols['real_value'].data+1j*cols['imag_value'].data
    else:
        return cols['x'].data, cols['y'].data, cols['real_value'].data

#################################################################################

def lookup(x, y, val, xin, yin, distance_ULIM=NP.inf, oob_value=0.0,
           remove_oob=True):

    """
    -----------------------------------------------------------------------------
    Perform a lookup operation based on nearest neighbour algorithm using
    KD-Trees when the lookup database and required coordinate locations are
    specified.

    Inputs:
    
    x       [numpy array] x-coordinates in the lookup table

    y       [numpy array] y-coordinates in the lookup table

    val     [numpy array] values at the x and y locations. x, y, and val must be
            of same size

    xin     [numpy array] x-coordinates at which values are required

    yin     [numpy array] y-coordinates at which values are required. xin and yin
            must be of same size

    distance_ULIM
            [scalar] A positive number for the upper bound on distance while
            searching for nearest neighbours. Neighbours outside of this upper
            bound are not searched for. Default = NP.inf (infinite distance upper 
            bound means nearest neighbours will be searched all the way out to 
            infinite distance)

    oob_value
            [scalar] Value to be returned at the location if the nearest
            neighbour for the location is not found inside distance_ULIM.
            Default = 0

    remove_oob
            [boolean] If set to True, results of nearest neighbour search and 
            lookup are returned only for those input locations which have a 
            neighbour within the distance upper bound. Locations with no 
            neighbours inside the distance upper bound will not be in the 
            returned results. Default = True. If set to False, even if no 
            nearest neighbours are found within the distance upper bound, an
            infinite distance and out of bound index from the lookup table are
            returned.

    Outputs: 

    Returns a tuple of nearest neighbour lookup value, the index of input
    location and nearest neighbour distance in the lookup table. The tuple
    consists of the following elements in this order.

    ibind   [numpy vector] indices of the input locations for which nearest 
            neighbours were found from the lookup table. If remove_oob is 
            not set to True, size of ibind is equal to size of xin or yin, 
            otherwise size of ibind is less than or equal to size of xin or yin. 
            Values of ibind equal to size of lookup table values indicates the 
            nearest neighbours for the corresponding input location could not be 
            found within the distance upper bound from the lookup table. The
            corresponding values in distNN are returned as inf or NP.inf

    nnval   [numpy vector] nearest neighbour lookup values for the given input
            locations. If remove_oob is not set to True, size of nnval is equal
            to size of xin or yin, otherwise size of nnval is less than or equal
            to size of xin or yin. In such a case, out of bound locations are
            filled with oob_value. Corresponding values in distNN are returned as
            inf or NP.inf

    distNN  [numpy vector] distance to the nearest neighbour in the lookup table
            for the given input locations. If remove_oob is not set to True, size 
            of distNN is equal to size of xin or yin, otherwise size of distNN is 
            less than or equal to size of xin or yin. In such a case, out of 
            bound locations are filled with inf or NP.inf. 
    
    -----------------------------------------------------------------------------
    """

    try:
        x, y, val, xin, yin
    except NameError:
        raise NameError('x, y, val, xin, and yin must be specified for lookup operation.')

    if (x.size != y.size) or (x.size != val.size):
        raise ValueError('x, y, and val must be of same size.')

    if xin.size != yin.size:
        raise ValueError('Input parameters xin and yin must have same size.')

    kdt = KDT(zip(x,y))
    dist, ind = kdt.query(NP.hstack((xin.reshape(-1,1), yin.reshape(-1,1))), distance_upper_bound=distance_ULIM)
    nnval = NP.zeros(ind.size, dtype=val.dtype)
    nnval[dist < distance_ULIM] = val[ind[dist < distance_ULIM]]
    nnval[dist >= distance_ULIM] = oob_value
    ibind = NP.arange(ind.size) # in-bound indices
    if remove_oob:
        nnval = nnval[dist <= distance_ULIM]
        ibind = ibind[dist <= distance_ULIM]
        distNN = dist[dist <= distance_ULIM]

    return ibind, nnval, distNN

#################################################################################
