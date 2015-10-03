import numpy as NP

################################################################################

def hexagon_generator(spacing, n_total=None, n_side=None, orientation=None, 
                      center=None):
    
    """
    ----------------------------------------------------------------------------
    Generate a grid of baseline locations filling a regular hexagon. 
    Primarily intended for HERA experiment.

    Inputs:
    
    spacing      [scalar] positive scalar specifying the spacing between
                 antennas. Must be specified, no default.

    n_total      [scalar] positive integer specifying the total number of
                 antennas to be placed in the hexagonal array. This value
                 will be checked if it valid for a regular hexagon. If
                 n_total is specified, n_side must not be specified. 
                 Default = None.

    n_side       [scalar] positive integer specifying the number of antennas
                 on the side of the hexagonal array. If n_side is specified,
                 n_total should not be specified. Default = None

    orientation  [scalar] counter-clockwise angle (in degrees) by which the 
                 principal axis of the hexagonal array is to be rotated. 
                 Default = None (means 0 degrees)

    center       [2-element list or numpy array] specifies the center of the
                 array. Must be in the same units as spacing. The hexagonal
                 array will be centered on this position.

    Outputs:

    Two element tuple with these elements in the following order:

    xy           [2-column array] x- and y-locations. x is in the first
                 column, y is in the second column. Number of xy-locations
                 is equal to the number of rows which is equal to n_total

    id           [numpy array of string] unique antenna identifier. Numbers
                 from 0 to n_antennas-1 in string format.

    Notes: 

    If n_side is the number of antennas on the side of the hexagon, then
    n_total = 3*n_side**2 - 3*n_side + 1
    ----------------------------------------------------------------------------
    """

    try:
        spacing
    except NameError:
        raise NameError('No spacing provided.')

    if not isinstance(spacing, (int, float)):
        raise TypeError('spacing must be scalar value')

    if spacing <= 0:
        raise ValueError('spacing must be positive')
        
    if orientation is not None:
        if not isinstance(orientation, (int,float)):
            raise TypeError('orientation must be a scalar')

    if center is not None:
        if not isinstance(center, (list, NP.ndarray)):
            raise TypeError('center must be a list or numpy array')
        center = NP.asarray(center)
        if center.size != 2:
            raise ValueError('center should be a 2-element vector')
        center = center.reshape(1,-1)

    n_center, n_side, n_total = hexagon_relations(n_total=n_total, n_side=n_side)

    xref = NP.arange(2*n_side-1, dtype=NP.float)
    xloc, yloc = [], []
    for i in range(1,n_side):
        x = xref[:-i] + i * NP.cos(NP.pi/3)   # Select one less antenna each time and displace
        y = i*NP.sin(NP.pi/3) * NP.ones(2*n_side-1-i)
        xloc += x.tolist() * 2   # Two lists, one for the top and the other for the bottom
        yloc += y.tolist()   # y-locations of the top list
        yloc += (-y).tolist()   # y-locations of the bottom list

    xloc += xref.tolist()   # Add the x-locations of central line of antennas
    yloc += [0.0] * int(2*n_side-1)   # Add the y-locations of central line of antennas

    if len(xloc) != len(yloc):
        raise ValueError('Sizes of x- and y-locations do not agree')

    xy = zip(xloc, yloc)
    if len(xy) != n_total:
        raise ValueError('Sizes of x- and y-locations do not agree with n_total')

    xy = NP.asarray(xy)
    xy = xy - NP.mean(xy, axis=0, keepdims=True)    # Shift the center to origin
    if orientation is not None:   # Perform any rotation
        angle = NP.radians(orientation)
        rot_matrix = NP.asarray([[NP.cos(angle), -NP.sin(angle)], 
                                 [NP.sin(angle), NP.cos(angle)]])
        xy = NP.dot(xy, rot_matrix.T)

    xy *= spacing    # Scale by the spacing
    if center is not None:   # Shift the center
        xy += center

    return (NP.asarray(xy), map(str, range(n_total)))

################################################################################

def MWA_128T(layout_file=None):

    """
    ----------------------------------------------------------------------------
    Read the MWA 128T antenna layout from the specified file containing the 
    layout

    Inputs:

    layout_file [string] String containing the filename including the full path
                to the layout file

    Outputs:

    xy          [2-column array] x- and y-locations. x is in the first
                column, y is in the second column. Number of xy-locations
                is equal to the number of rows which is equal to n_total

    id          [numpy array of string] unique antenna identifier. Numbers
                from 0 to n_antennas-1 in string format.
    ----------------------------------------------------------------------------
    """

    if layout_file is None:
        layout_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

    if not isinstance(layout_file, str):
        raise TypeError('layout_file must be a string')

    ant_info = NP.loadtxt(layout_file, skiprows=6, comments='#', usecols=(0,1,2,3))
    ant_id = ant_info[:,0].astype(int).astype(str)
    ant_locs = ant_info[:,1:]

    return (ant_locs, ant_id)

################################################################################

def hexagon_relations(n_total=None, n_side=None, n_center=None):
    
    """
    ----------------------------------------------------------------------------
    Generate a grid of baseline locations filling a regular hexagon. 
    Primarily intended for HERA experiment.

    Inputs:
    
    n_total      [scalar] positive integer specifying the total number of
                 antennas to be placed in the hexagonal array. This value
                 will be checked if it valid for a regular hexagon. If
                 n_total is specified, n_side must not be specified. 
                 Default = None.

    n_side       [scalar] positive integer specifying the number of antennas
                 on the side of the hexagonal array. If n_side is specified,
                 n_total should not be specified. Default = None

    n_center     [scalar] positive integer specifying the number of antennas 
                 in the central row of the array. Must be in the same units as 
                 spacing. 

    Outputs:

    Three element tuple with these elements in the following order:

    n_side       [scalar] positive integer specifying the number of antennas
                 on the side of the hexagonal array

    n_center     [scalar] positive integer specifying the number of antennas 
                 in the central row of the array. Will be in the same units as 
                 spacing. 

    n_total      [scalar] positive integer specifying the total number of
                 antennas to be placed in the hexagonal array

    Notes: 

    If n_side is the number of antennas on the side of the hexagon, then
    n_total = 3*n_side**2 - 3*n_side + 1 and n_center = 2*n_side - 1
    ----------------------------------------------------------------------------
    """

    nspec = (n_total is not None) + (n_side is not None) + (n_center is not None)
    if nspec != 1:
        raise ValueError('Too many or too little specifications')

    if n_total is not None:
        if not isinstance(n_total, int):
            raise TypeError('n_total must be an integer')
        if n_total <= 0:
            raise ValueError('n_total must be positive')
        sqroots = NP.roots([3.0, -3.0, 1.0-n_total])
        valid_ind = NP.logical_and(sqroots.real >= 1, sqroots.imag == 0.0)
        if NP.any(valid_ind):
            sqroot = int(round(sqroots[valid_ind]))
        else:
            raise ValueError('No valid root found for the quadratic equation with the specified n_total')

        n_side = sqroot
        if (3*n_side**2 - 3*n_side + 1 != n_total):
            raise ValueError('n_total is not a valid number for a hexagonal array')

        n_center = 2*n_side - 1
    elif n_side is not None:
        if not isinstance(n_side, int):
            raise TypeError('n_side must be an integer')
        if n_side <= 0:
            raise ValueError('n_side must be positive')
        n_total = 3*n_side**2 - 3*n_side + 1
        n_center = 2*n_side - 1
    else:
        n_side = (n_center + 1) / 2

    return (n_center, n_side, n_total)

################################################################################
