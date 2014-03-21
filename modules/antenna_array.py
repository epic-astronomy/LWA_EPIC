import numpy as NP
import scipy.constants as FCNST
from astropy.io import fits
import my_DSP_modules as DSP
import geometry as GEOM
import my_gridding_modules as GRD
import lookup_operations as LKP

#####################################################################  

class PolInfo:

    """
    ----------------------------------------------------------------------------
    Class to manage polarization information of an antenna. 

    Attributes:

    Et_P1:   A complex vector representing a time series of electric field
             for polarization P1

    Et_P2:   A complex vector representing a time series of electric field
             for polarization P2 which is orthogonal to P1
        
    flag_P1: Boolean value. True means P1 is to be flagged. Default = False

    flag_P2: Boolean value. True means P2 is to be flagged. Default = False

    pol_type: 'Linear' or 'Circular' polarization

    Ef_P1:   A complex vector representing the Fourier transform of Et_P1

    Ef_P2:   A complex vector representing the Fourier transform of Et_P2

    Member functions:

    __init__():    Initializes an instance of class PolInfo

    __str__():     Prints a summary of current attributes.

    temporal_F():  Perform a Fourier transform of an Electric field time series

    update():      Routine to update the Electric field and flag information.
    
    Read the member function docstrings for details. 
    ----------------------------------------------------------------------------
    """

    def __init__(self):
        """
        ------------------------------------------------------------------------
        Initialize the PolInfo Class which manages polarization information of
        an antenna. 

        Class attributes initialized are:
        Et_P1, Et_P2, flag_P1, flag_P2, pol_type, Ef_P1, Ef_P2
     
        Read docstring of class PolInfo for details on these attributes.
        ------------------------------------------------------------------------
        """

        self.Et_P1 = 0.0
        self.Et_P2 = 0.0
        self.flag_P1 = False
        self.flag_P2 = False
        self.pol_type = ''
        self.Ef_P1 = 0.0
        self.Ef_P2 = 0.0

    ############################################################################ 

    def __str__(self):
        return ' Instance of class "{0}" in module "{1}" \n flag (P1): {2} \n flag (P2): {3} \n Polarization type: {4} '.format(self.__class__.__name__, self.__module__, self.flag_P1, self.flag_P2, self.pol_type)

    ############################################################################ 

    def temporal_F(self, pol=None):

        """
        ------------------------------------------------------------------------
        Perform a Fourier transform of an Electric field time series after 
        doubling the length of the sequence with zero padding (in order to be 
        identical to what would be obtained from a XF oepration)

        Keyword Input(s):

        pol     polarization to be Fourier transformed. Set to 'P1' or 'P2'. If 
                None provided, both time series Et_P1 and Et_P2 are Fourier 
                transformed.

        Outputs:

        Electric field spectrum Ef_P1 and/or Ef_P2 depending on value of pol.
        ------------------------------------------------------------------------
        """

        if pol is None:
            Et_P1 = NP.pad(self.Et_P1, (0,len(self.Et_P1)), 'constant',
                           constant_values=(0,0))
            Et_P2 = NP.pad(self.Et_P2, (0,len(self.Et_P2)), 'constant',
                           constant_values=(0,0))
            self.Ef_P1 = DSP.FT1D(Et_P1, ax=0, use_real=False, shift=True)
            self.Ef_P2 = DSP.FT1D(Et_P2, ax=0, use_real=False, shift=True)
        elif pol in ['P1','p1','P2','p2','x','X','y','Y']:
            if pol in ['P1','p1','x','X']:
                Et_P1 = NP.pad(self.Et_P1, (0,len(self.Et_P1)), 'constant',
                               constant_values=(0,0))
                self.Ef_P1 = DSP.FT1D(Et_P1, ax=0, use_real=False, shift=True)
            else:
                Et_P2 = NP.pad(self.Et_P2, (0,len(self.Et_P2)), 'constant',
                               constant_values=(0,0))
                self.Ef_P2 = DSP.FT1D(Et_P2, ax=0, use_real=False, shift=True)
        else:
            raise ValueError('Polarization string unrecognized. Verify inputs. Aborting PolInfo.temporal_F()')

    ############################################################################

    def update(self, Et_P1=None, Et_P2=None, flag_P1=False, flag_P2=False, pol_type='Linear'):

        """
        ------------------------------------------------------------------------
        Routine to update the Electric field and flag information.

        Keyword input(s):

        Et_P1:         [Complex vector] The new electric field time series in
                       polarization P1 that will replace the current attribute

        Et_P2:         [Complex vector] The new electric field time series in 
                       polarization P2 that will replace the current attribute

        flag_P1:       The new flag for polarization P1

        flag_P2:       The new flag for polarization P2
                        
        pol_type:      'Linear' or 'Circular' polarization
        ------------------------------------------------------------------------
        """
        
        if Et_P1 is not None:
            self.Et_P1 = NP.asarray(Et_P1)
            self.temporal_F(pol='X')

        if Et_P2 is not None:
            self.Et_P2 = NP.asarray(Et_P2)
            self.temporal_F(pol='Y')

        if flag_P1 is not None: self.flag_P1 = flag_P1
        if flag_P2 is not None: self.flag_P2 = flag_P2
        if pol_type is not None: self.pol_type = pol_type

#####################################################################################

class Antenna:

    """
    ----------------------------------------------------------------------------
    Class to manage individual antenna information.

    Attributes:

    label:      [Scalar] A unique identifier (preferably a string) for the 
                antenna. 

    latitude:   [Scalar] Latitude of the antenna's location.

    location:   [Instance of GEOM.Point class] The location of the antenna in 
                local East, North, Up coordinate system.

    timestamp:  [Scalar] String or float representing the timestamp for the 
                current attributes

    t:          [vector] The time axis for the time series of electric fields

    f:          [vector] Frequency axis obtained by a Fourier Transform of
                the electric field time series. Same length as attribute t 

    f0:         [Scalar] Positive value for the center frequency in Hz.

    pol:        [Instance of class PolInfo] polarization information for the 
                antenna. Read docstring of class PolInfo for details

    wts_P1:     [List of 1-column Vectors] The gridding weights for antenna in 
                the local ENU coordinate system under polarization P1. These could 
                be complex. This is provided as a list of numpy vectors, where each 
                vector corresponds to a frequency channel. See wtspos_P1_scale.

    wts_P2:     [List of 1-column Vectors] The gridding weights for antenna in 
                the local ENU coordinate system under polarization P2. These could 
                be complex. This is provided as a list of numpy vectors, where each 
                vector corresponds to a frequency channel. See wtspos_P2_scale.

    wtspos_P1:  [List of 2-column numpy arrays] Each 2-column numpy array is the 
                position of the gridding weights for a corresponding frequency 
                channel for polarization P1. The size of the list must be the 
                as wts_P1 and the number of channels. See wtspos_P1_scale. Units
                are in number of wavelengths.

    wtspos_P2:  [List of 2-column numpy arrays] Each 2-column numpy array is the 
                position of the gridding weights for a corresponding frequency 
                channel for polarization P2. The size of the list must be the 
                as wts_P2 and the number of channels. See wtspos_P2_scale. Units
                are in number of wavelengths.

    wtspos_P1_scale [None or 'scale'] If None, numpy vectors in wts_P1 and 
                    wtspos_P1 are provided for each frequency channel. If set to
                    'scale' wts_P1 and wtspos_P1 contain a list of only one 
                    numpy array corresponding to a reference frequency. This is
                    scaled internally to correspond to the first channel.
                    The gridding positions are correspondingly scaled to all the 
                    frequency channels.

    wtspos_P2_scale [None or 'scale'] If None, numpy vectors in wts_P2 and 
                    wtspos_P2 are provided for each frequency channel. If set to
                    'scale' wts_P2 and wtspos_P2 contain a list of only one 
                    numpy array corresponding to a reference frequency. This is
                    scaled internally to correspond to the first channel.
                    The gridding positions are correspondingly scaled to all the 
                    frequency channels.

    gridinfo_P1     [Dictionary] Contains gridding information pertaining to the
                    antenna under polarization P1. It contains keys for each 
                    frequency channel number. Each of these keys holds another
                    dictionary. This sub-dictionary consists of the following 
                    keys which hold the information described below:

                    f:           the frequency [in Hz] corresponding to the channel
                                 number
                    flag:        [Boolean] flag for frequency channel. True means
                                 the frequency channel is to be flagged.
                    gridxy_ind   [List of tuples] Each tuple holds the index of the
                                 interpolated position (in local ENU coordinate 
                                 system) on the grid. 
                    illumination [Numpy vector] The voltage pattern contributed by
                                 the antenna at that frequency to the grid. This 
                                 could contain complex values. 
                    Ef           [Numpy vector] The voltage seen by the antenna on
                                 the grid. This could contain complex values. 

    gridinfo_P2     [Dictionary] Contains gridding information pertaining to the
                    antenna under polarization P2. It contains keys for each 
                    frequency channel number. Each of these keys holds another
                    dictionary. This sub-dictionary consists of the following 
                    keys which hold the information described below:

                    f:           the frequency [in Hz] corresponding to the channel
                                 number
                    flag:        [Boolean] flag for frequency channel. True means
                                 the frequency channel is to be flagged.
                    gridxy_ind   [List of tuples] Each tuple holds the index of the
                                 interpolated position (in local ENU coordinate 
                                 system) on the grid. 
                    illumination [Numpy vector] The voltage pattern contributed by
                                 the antenna at that frequency to the grid. This 
                                 could contain complex values. 
                    Ef           [Numpy vector] The voltage seen by the antenna on
                                 the grid. This could contain complex values. 
    
    blc_P1          [2-element numpy array] Bottom Left corner where the antenna
                    contributes non-zero weight to the grid in polarization P1

    trc_P1          [2-element numpy array] Top right corner where the antenna
                    contributes non-zero weight to the grid in polarization P1

    blc_P2          [2-element numpy array] Bottom Left corner where the antenna
                    contributes non-zero weight to the grid in polarization P2

    trc_P2          [2-element numpy array] Top right corner where the antenna
                    contributes non-zero weight to the grid in polarization P2

    Member Functions:

    __init__():      Initializes an instance of class Antenna

    __str__():       Prints a summary of current attributes

    channels():      Computes the frequency channels from a temporal Fourier 
                     Transform

    update():        Updates the antenna instance with newer attribute values

    save():          Saves the antenna information to disk. Needs serious 
                     development. 

    Read the member function docstrings for details.
    ----------------------------------------------------------------------------
    """

    def __init__(self, label, latitude, location, center_freq):
        """
        ------------------------------------------------------------------------
        Initialize the Antenna Class which manages an antenna's information 

        Class attributes initialized are:
        label, latitude, location, pol, t, timestamp, f0, f, wts_P1, wts_P2, 
        wtspos_P1, wtspos_P2, wtspos_P1_scale, wtspos_P2_scale, gridinfo_P1, 
        gridinfo_P2, blc_P1, trc_P1, blc_P2, trc_P2
     
        Read docstring of class Antenna for details on these attributes.
        ------------------------------------------------------------------------
        """

        try:
            label
        except NameError:
            raise NameError('Antenna label must be provided.')

        try:
            latitude
        except NameError:
            self.latitude = 0.0

        try:
            location
        except NameError:
            self.location = GEOM.Point()

        try:
            center_freq
        except NameError:
            raise NameError('Center frequency must be provided.')

        self.label = label
        self.latitude = latitude

        if isinstance(location, GEOM.Point):
            self.location = location
        elif isinstance(location, (list, tuple, NP.ndarray)):
            self.location = GEOM.Point(location)
        else:
            raise TypeError('Antenna position must be a 3-element tuple or an instance of GEOM.Point')

        self.pol = PolInfo()
        self.t = 0.0
        self.timestamp = 0.0
        self.f0 = center_freq
        self.f = self.f0

        self.wts_P1 = []
        self.wts_P2 = []
        self.wtspos_P1_scale = None
        self.wtspos_P1 = []
        self.wtspos_P2 = []
        self.wtspos_P2_scale = None
        
        self.gridinfo_P1 = {}
        self.gridinfo_P2 = {}

        self.blc_P1 = NP.asarray([self.location.x, self.location.y]).reshape(1,2)
        self.trc_P1 = NP.asarray([self.location.x, self.location.y]).reshape(1,2)
        self.blc_P2 = NP.asarray([self.location.x, self.location.y]).reshape(1,2)
        self.trc_P2 = NP.asarray([self.location.x, self.location.y]).reshape(1,2)

    #################################################################################

    def __str__(self):
        return ' Instance of class "{0}" in module "{1}" \n label: {2} \n location: {3}'.format(self.__class__.__name__, self.__module__, self.label, self.location.__str__())

    #################################################################################

    def channels(self):
        """
        ------------------------------------------------------------------------
        Computes the frequency channels from a temporal Fourier Transform 
        assuming the temporal sequence has doubled in length with zero 
        padding while maintaining the time resolution.

        Output(s):

        Frequencies corresponding to channels obtained by a Fourier Transform
        of the time series.
        ------------------------------------------------------------------------
        """

        return DSP.spectax(2*len(self.t), self.t[1]-self.t[0], shift=True)

    #################################################################################

    def update(self, label=None, Et_P1=None, Et_P2=None, t=None, timestamp=None,
               location=None, wtsinfo_P1=None, wtsinfo_P2=None, flag_P1=None,
               flag_P2=None, gridfunc_freq=None, ref_freq=None, pol_type='Linear',
               verbose=False):
        """
        ------------------------------------------------------------------------
        Routine to update all or some of the antenna information 

        Inputs:

        label      [scalar string] Antenna identifier
                   
        Et_P1      [Numpy vector] Electric field stream for P1 polarization. 
                   Should be of same length as t
                   
        Et_P2      [Numpy vector] Electric field stream for P2 polarization. 
                   Should be of same length as t
                   
        t          [Numpy vector] Time axis for the time series

        timestamp  [Scalar] Float or string that uniquely identifies the time 
                   series

        location   [instance of class GEOM.Point] Local ENU coordinates of the
                   antenna

        wtsinfo_P1 [List of dictionaries] Length of list is equal to the number
                   of frequency channels or one (equivalent to setting
                   wtspos_P1_scale to 'scale'.). The list is indexed by 
                   the frequency channel number. Each element in the list
                   consists of a dictionarycorresponding to that frequency
                   channel. Each dictionary consists of three items with the
                   following keys in no particular order:

                   wtspos      [2-column Numpy array, optional] u- and v- 
                               positions for the gridding weights. Units
                               are in number of wavelengths. It is 
                               recommended that sufficient padding is provided in 
                               wtspos and wts
                   wts         [Numpy array] Complex gridding weights. Size is
                               equal to the number of rows in wtspos above
                   orientation [scalar] Orientation (in radians) of the wtspos 
                               coordinate system relative to the local ENU 
                               coordinate system. It is measured North of East. 
                   lookup      [string] If set, refers to a file location
                               containing the wtspos and wts information above as
                               columns (x-loc [float], y-loc [float], wts
                               [real], wts[imag if any]). If set, wtspos and wts 
                               information are obtained from this lookup table 
                               and the wtspos and wts keywords in the dictionary
                               are ignored. Note that wtspos values are obtained
                               after dividing x- and y-loc lookup values by the
                               wavelength

        wtsinfo_P2 [List of dictionaries] Length of list is equal to the number
                   of frequency channels or one (equivalent to setting
                   wtspos_P2_scale to 'scale'.). The list is indexed by 
                   the frequency channel number. Each element in the list
                   consists of a dictionarycorresponding to that frequency
                   channel. Each dictionary consists of three items with the
                   following keys in no particular order:

                   wtspos      [2-column Numpy array, optional] u- and v- 
                               positions for the gridding weights. Units
                               are in number of wavelengths. It is 
                               recommended that sufficient padding is provided in 
                               wtspos and wts
                   wts         [Numpy array] Complex gridding weights. Size is
                               equal to the number of rows in wtspos above
                   orientation [scalar] Orientation (in radians) of the wtspos 
                               coordinate system relative to the local ENU 
                               coordinate system. It is measured North of East. 
                   lookup      [string] If set, refers to a file location
                               containing the wtspos and wts information above as
                               columns (x-loc [float], y-loc [float], wts
                               [real], wts[imag if any]). If set, wtspos and wts 
                               information are obtained from this lookup table 
                               and the wtspos and wts keywords in the dictionary
                               are ignored. Note that wtspos values are obtained
                               after dividing x- and y-loc lookup values by the
                               wavelength

        flag_P1    [Boolean] Flag for polarization P1 for the antenna

        flag_P2    [Boolean] Flag for polarization P2 for the antenna

        gridfunc_freq [String scalar] If set to None (not provided) or to 'scale'
                      assumes that wtspos_P1 and wtspos_P2 are given for a
                      reference frequency which need to be scaled for the frequency
                      channels. Will be ignored if the wtsinfo_P1 and wtsinfo_P2 
                      have sizes equal to the number of frequency channels.

        ref_freq   [Scalar] Positive value (in Hz) of reference frequency (used
                   if gridfunc_freq is set to None or 'scale') at which
                   wtspos_P1 and wtspos_P2 are provided. If set to None,
                   ref_freq is assumed to be equal to the center frequency in 
                   the class Antenna's attribute. 

        pol_type   [String scalar] Should be set to 'Linear' or 'Circular' to
                   denote the type of polarization. Default = 'Linear'

        verbose    [Boolean] Default = False. If set to True, prints some 
                   diagnotic or progress messages.

        ------------------------------------------------------------------------
        """

        if label is not None: self.label = label
        if location is not None: self.location = location
        if timestamp is not None: self.timestamp = timestamp

        if t is not None:
            self.t = t
            self.f = self.f0 + self.channels()           

        if (flag_P1 is not None) or (flag_P2 is not None) or (Et_P1 is not None) or (Et_P2 is not None):
            self.pol.update(Et_P1, Et_P2, flag_P1, flag_P2, pol_type)

        if wtsinfo_P1 is not None:
            self.wtspos_P1 = []
            self.wts_P1 = []
            angles = []
            if len(wtsinfo_P1) == len(self.f):
                self.wtspos_P1_scale = None
                # self.wts_P1 += [wtsinfo[1] for wtsinfo in wtsinfo_P1]
                angles += [wtsinfo['orientation'] for wtsinfo in wtsinfo_P1]
                for i in range(len(self.f)):
                    rotation_matrix = NP.asarray([[NP.cos(-angles[i]),  NP.sin(-angles[i])],
                                                  [-NP.sin(-angles[i]), NP.cos(-angles[i])]])
                    if ('lookup' not in wtsinfo_P1[i]) or (wtsinfo_P1[i]['lookup'] is None):
                        self.wts_P1 += [wtsinfo_P1[i]['wts']]
                        wtspos = wtsinfo_P1[i]['wtspos']
                    else:
                        lookupdata = LKP.read_lookup(wtsinfo_P1[i]['lookup'])
                        wtspos = NP.hstack((lookupdata[0].reshape(-1,1),lookupdata[1].reshape(-1,1))) * (self.f[i]/FCNST.c)
                        self.wts_P1 += [lookupdata[2]]
                        # lookupdata = NP.loadtxt(wtsinfo_P1[i]['lookup'], usecols=(1,2,3), dtype=(NP.float, NP.float, NP.complex))
                        # wtspos = NP.hstack((lookupdata[:,0].reshape(-1,1), lookupdata[:,1].reshape(-1,1)))
                        # self.wts_P1 += [lookupdata[:,2]]
                    self.wtspos_P1 += [ NP.dot(NP.asarray(wtspos), rotation_matrix.T) ]
                self.blc_P1 = NP.repeat(NP.asarray([self.location.x, self.location.y]).reshape(1,-1),len(self.f),axis=0) - NP.abs(NP.asarray([NP.amin((FCNST.c/self.f[i])*self.wtspos_P1[i],0) for i in range(len(self.f))]))
                self.trc_P1 = NP.repeat(NP.asarray([self.location.x, self.location.y]).reshape(1,-1),len(self.f),axis=0) + NP.abs(NP.asarray([NP.amax((FCNST.c/self.f[i])*self.wtspos_P1[i],0) for i in range(len(self.f))]))
            elif len(wtsinfo_P1) == 1:
                if (gridfunc_freq is None) or (gridfunc_freq == 'scale'):
                    self.wtspos_P1_scale = 'scale'
                    if ref_freq is None:
                        ref_freq = self.f0
                    angles = wtsinfo_P1[0]['orientation']
                    rotation_matrix = NP.asarray([[NP.cos(-angles),  NP.sin(-angles)],
                                                  [-NP.sin(-angles), NP.cos(-angles)]])
                    if ('lookup' not in wtsinfo_P1[0]) or (wtsinfo_P1[0]['lookup'] is None):
                        self.wts_P1 += [ wtsinfo_P1[0]['wts'] ]
                        wtspos = wtsinfo_P1[0]['wtspos']
                    else:
                        lookupdata = LKP.read_lookup(wtsinfo_P1[0]['lookup'])
                        wtspos = NP.hstack((lookupdata[0].reshape(-1,1),lookupdata[1].reshape(-1,1))) * (ref_freq/FCNST.c)
                        self.wts_P1 += [lookupdata[2]]
                        # lookupdata = NP.loadtxt(wtsinfo_P1[0]['lookup'], usecols=(1,2,3), dtype=(NP.float, NP.float, NP.complex))
                        # wtspos = NP.hstack((lookupdata[:,0].reshape(-1,1), lookupdata[:,1].reshape(-1,1)))
                        # self.wts_P1 += [lookupdata[:,2]]
                    self.wtspos_P1 += [ (self.f[0]/ref_freq) * NP.dot(NP.asarray(wtspos), rotation_matrix.T) ]
                # elif gridfunc_freq == 'noscale':
                #     self.wtspos_P1_scale = 'noscale'
                #     self.wts_P1 += [ wtsinfo_P1[1] ]
                #     angles += [ wtsinfo_P1[2] ]
                #     rotation_matrix = NP.asarray([[NP.cos(-angles),  NP.sin(-angles)],
                #                                   [-NP.sin(-angles), NP.cos(-angles)]])
                #     self.wtspos_P1 += [ NP.dot(NP.asarray(wtsinfo_P1[0][0]), rotation_matrix.T) ]
                else:
                    raise ValueError('gridfunc_freq must be set to None, "scale" or "noscale".')

                self.blc_P1 = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) - (FCNST.c/self.f[0]) * NP.abs(NP.amin(self.wtspos_P1[0], 0))
                self.trc_P1 = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) + (FCNST.c/self.f[0]) * NP.abs(NP.amax(self.wtspos_P1[0], 0))

            else:
                raise ValueError('Number of elements in wtsinfo_P1 is incompatible with the number of channels.')

        if wtsinfo_P2 is not None:
            self.wtspos_P2 = []
            self.wts_P2 = []
            angles = []
            if len(wtsinfo_P2) == len(self.f):
                self.wtspos_P2_scale = None
                # self.wts_P2 += [wtsinfo[1] for wtsinfo in wtsinfo_P2]
                angles += [wtsinfo['orientation'] for wtsinfo in wtsinfo_P2]
                for i in range(len(self.f)):
                    rotation_matrix = NP.asarray([[NP.cos(-angles[i]),  NP.sin(-angles[i])],
                                                  [-NP.sin(-angles[i]), NP.cos(-angles[i])]])
                    if ('lookup' not in wtsinfo_P2[i]) or (wtsinfo_P2[i]['lookup'] is None):
                        self.wts_P2 += [wtsinfo_P2[i]['wts']]
                        wtspos = wtsinfo_P2[i]['wtspos']
                    else:
                        lookupdata = LKP.read_lookup(wtsinfo_P2[i]['lookup'])
                        wtspos = NP.hstack((lookupdata[0].reshape(-1,1),lookupdata[1].reshape(-1,1))) * (self.f[i]/FCNST.c)
                        self.wts_P2 += [lookupdata[2]]
                        # lookupdata = NP.loadtxt(wtsinfo_P2[i]['lookup'], usecols=(1,2,3), dtype=(NP.float, NP.float, NP.complex))
                        # wtspos = NP.hstack((lookupdata[:,0].reshape(-1,1), lookupdata[:,1].reshape(-1,1)))
                        # self.wts_P2 += [lookupdata[:,2]]
                    self.wtspos_P2 += [ NP.dot(NP.asarray(wtspos), rotation_matrix.T) ]
                self.blc_P2 = NP.repeat(NP.asarray([self.location.x, self.location.y]).reshape(1,-1),len(self.f),axis=0) - NP.abs(NP.asarray([NP.amin((FCNST.c/self.f[i])*self.wtspos_P2[i],0) for i in range(len(self.f))]))
                self.trc_P2 = NP.repeat(NP.asarray([self.location.x, self.location.y]).reshape(1,-1),len(self.f),axis=0) + NP.abs(NP.asarray([NP.amax((FCNST.c/self.f[i])*self.wtspos_P2[i],0) for i in range(len(self.f))]))
            elif len(wtsinfo_P2) == 1:
                if (gridfunc_freq is None) or (gridfunc_freq == 'scale'):
                    self.wtspos_P2_scale = 'scale'
                    if ref_freq is None:
                        ref_freq = self.f0
                    angles = wtsinfo_P2[0]['orientation']
                    rotation_matrix = NP.asarray([[NP.cos(-angles),  NP.sin(-angles)],
                                                  [-NP.sin(-angles), NP.cos(-angles)]])
                    if ('lookup' not in wtsinfo_P2[0]) or (wtsinfo_P2[0]['lookup'] is None):
                        self.wts_P2 += [ wtsinfo_P2[0]['wts'] ]
                        wtspos = wtsinfo_P2[0]['wtspos']
                    else:
                        lookupdata = LKP.read_lookup(wtsinfo_P2[0]['lookup'])
                        wtspos = NP.hstack((lookupdata[0].reshape(-1,1),lookupdata[1].reshape(-1,1))) * (ref_freq/FCNST.c)
                        self.wts_P2 += [lookupdata[2]]
                        # lookupdata = NP.loadtxt(wtsinfo_P2[0]['lookup'], usecols=(1,2,3), dtype=(NP.float, NP.float, NP.complex))
                        # wtspos = NP.hstack((lookupdata[:,0].reshape(-1,1), lookupdata[:,1].reshape(-1,1)))
                        # self.wts_P2 += [lookupdata[:,2]]
                    self.wtspos_P2 += [ (self.f[0]/ref_freq) * NP.dot(NP.asarray(wtspos), rotation_matrix.T) ]
                # elif gridfunc_freq == 'noscale':
                #     self.wtspos_P2_scale = 'noscale'
                #     self.wts_P2 += [ wtsinfo_P2[1] ]
                #     angles += [ wtsinfo_P2[2] ]
                #     rotation_matrix = NP.asarray([[NP.cos(-angles),  NP.sin(-angles)],
                #                                   [-NP.sin(-angles), NP.cos(-angles)]])
                #     self.wtspos_P2 += [ NP.dot(NP.asarray(wtsinfo_P2[0][0]), rotation_matrix.T) ]
                else:
                    raise ValueError('gridfunc_freq must be set to None, "scale" or "noscale".')

                self.blc_P2 = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) - (FCNST.c/self.f[0]) * NP.abs(NP.amin(self.wtspos_P2[0], 0))
                self.trc_P2 = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) + (FCNST.c/self.f[0]) * NP.abs(NP.amax(self.wtspos_P2[0], 0))

            else:
                raise ValueError('Number of elements in wtsinfo_P2 is incompatible with the number of channels.')

        if verbose:
            print 'Updated antenna {0}.'.format(self.label)

    #############################################################################

    def save(self, antfile, pol=None, tabtype='BinTableHDU', overwrite=False,
             verbose=True):

        """
        -------------------------------------------------------------------------
        Saves the antenna information to disk. 

        Input:

        antfile     [string] antenna filename with full path. Will be appended 
                    with antenna label and '.fits' extension

        Keyword Input(s):

        pol         [string] indicates which polarization information to be 
                    saved. Allowed values are 'P1', 'P2' or None (default). If 
                    None, information on both polarizations are saved.

        tabtype     [string] indicates table type for one of the extensions in 
                    the FITS file. Allowed values are 'BinTableHDU' and 
                    'TableHDU' for binary ascii tables respectively. Default is
                    'BinTableHDU'.

        overwrite   [boolean] True indicates overwrite even if a file already 
                    exists. Default = False (does not overwrite)

        verbose     [boolean] If True (default), prints diagnostic and progress
                    messages. If False, suppress printing such messages.
        ----------------------------------------------------------------------------
        """

        try:
            antfile
        except NameError:
            raise NameError('No filename provided. Aborting Antenna.save().')

        filename = antfile + '.' + self.label + '.fits'

        if verbose:
            print '\nSaving information about antenna {0}...'.format(self.label)
            
        hdulist = []
        hdulist += [fits.PrimaryHDU()]
        hdulist[0].header['label'] = (self.label, 'Antenna label')
        hdulist[0].header['latitude'] = (self.latitude, 'Latitude of Antenna')
        hdulist[0].header['East'] = (self.location.x, 'Location of Antenna along local EAST')
        hdulist[0].header['North'] = (self.location.y, 'Location of Antenna along local NORTH')
        hdulist[0].header['Up'] = (self.location.z, 'Location of Antenna along local UP')
        hdulist[0].header['f0'] = (self.f0, 'Center frequency (Hz)')
        hdulist[0].header['tobs'] = (self.timestamp, 'Timestamp associated with observation.')
        hdulist[0].header.set('EXTNAME', 'Antenna ({0})'.format(self.label))

        if verbose:
            print '\tCreated a primary HDU.'

        cols = []
        cols += [fits.Column(name='time_sequence', format='D', array=self.t)]
        cols += [fits.Column(name='frequency', format='D', array=self.f)]
        columns = fits.ColDefs(cols, tbtype=tabtype)
        tbhdu = fits.new_table(columns)
        tbhdu.header.set('EXTNAME', 'GENERAL INFO')
        hdulist += [tbhdu]

        if (pol is None) or (pol == 'P1'):
            if verbose:
                print '\tWorking on polarization P1...'
                print '\t\tWorking on weights information...'
            cols = []
            for i in range(len(self.wts_P1)):
                if verbose:
                    print '\t\t\tProcessing channel # {0}'.format(i)
                cols += [fits.Column(name='wtspos[{0:0d}]'.format(i), format='2D()', array=self.wtspos_P1[0])]
                cols += [fits.Column(name='wts[{0:0d}]'.format(i), format='M()', array=self.wts_P1[0])]
            columns = fits.ColDefs(cols, tbtype=tabtype)
            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'wtsinfo_P1')
            tbhdu.header.set('X_BLC', self.blc_P1[0,0])
            tbhdu.header.set('Y_BLC', self.blc_P1[0,1])
            tbhdu.header.set('X_TRC', self.trc_P1[0,0])
            tbhdu.header.set('Y_TRC', self.trc_P1[0,1])
            hdulist += [tbhdu]
            if verbose:
                print '\t\tCreated separate extension HDU {0} with weights information'.format(tbhdu.header['EXTNAME'])
                print '\t\tWorking on gridding information...'

            cols = []
            for i in range(len(self.f)):
                if verbose:
                    print '\t\t\tProcessing channel # {0}'.format(i)
                cols += [fits.Column(name='gridxy[{0:0d}]'.format(i), format='2D()', array=NP.asarray(self.gridinfo_P1[i]['gridxy_ind']))]
                cols += [fits.Column(name='illumination[{0:0d}]'.format(i), format='M()', array=self.gridinfo_P1[i]['illumination'])]
                cols += [fits.Column(name='Ef[{0:0d}]'.format(i), format='M()', array=self.gridinfo_P1[i]['Ef'])]
            columns = fits.ColDefs(cols, tbtype=tabtype)
            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'gridinfo_P1')
            hdulist += [tbhdu]
            if verbose:
                print '\t\tCreated separate extension HDU {0} with weights information'.format(tbhdu.header['EXTNAME'])

        if (pol is None) or (pol == 'P2'):
            if verbose:
                print '\tWorking on polarization P2...'
                print '\t\tWorking on weights information...'
            cols = []
            for i in range(len(self.wts_P2)):
                if verbose:
                    print '\t\t\tProcessing channel # {0}'.format(i)
                cols += [fits.Column(name='wtspos[{0:0d}]'.format(i), format='2D()', array=self.wtspos_P2[0])]
                cols += [fits.Column(name='wts[{0:0d}]'.format(i), format='M()', array=self.wts_P2[0])]
            columns = fits.ColDefs(cols, tbtype=tabtype)
            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'wtsinfo_P2')
            tbhdu.header.set('X_BLC', self.blc_P2[0,0])
            tbhdu.header.set('Y_BLC', self.blc_P2[0,1])
            tbhdu.header.set('X_TRC', self.trc_P2[0,0])
            tbhdu.header.set('Y_TRC', self.trc_P2[0,1])
            hdulist += [tbhdu]
            if verbose:
                print '\t\tCreated separate extension HDU {0} with weights information'.format(tbhdu.header['EXTNAME'])
                print '\t\tWorking on gridding information...'

            cols = []
            for i in range(len(self.f)):
                if verbose:
                    print '\t\t\tProcessing channel # {0}'.format(i)
                cols += [fits.Column(name='gridxy[{0:0d}]'.format(i), format='2D()', array=NP.asarray(self.gridinfo_P2[i]['gridxy_ind']))]
                cols += [fits.Column(name='illumination[{0:0d}]'.format(i), format='M()', array=self.gridinfo_P2[i]['illumination'])]
                cols += [fits.Column(name='Ef[{0:0d}]'.format(i), format='M()', array=self.gridinfo_P2[i]['Ef'])]
            columns = fits.ColDefs(cols, tbtype=tabtype)
            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'gridinfo_P2')
            hdulist += [tbhdu]
            if verbose:
                print '\t\tCreated separate extension HDU {0} with weights information'.format(tbhdu.header['EXTNAME'])

        hdu = fits.HDUList(hdulist)
        hdu.writeto(filename, clobber=overwrite)

        if verbose:
            print '\tNow writing FITS file to disk:\n\t\t{0}'.format(filename)
            print '\tData for antenna {0} written successfully to FITS file on disk:\n\t\t{1}\n'.format(self.label, filename)

#####################################################################  

class AntennaPair(Antenna):
    def __init__(self, A1, A2):
        self.A1, self.A2 = A1, A2
        self.label = A1.label+'-'+A2.label
        self.location = A1.location-A2.location
        self.Et = DSP.XC(self.A1.Et, self.A2.Et)
        # self.t = (A1.t[1]-A1.t[0])*NP.asarray(range(0,len(self.Et)))
        self.Ef = DSP.FT1D(self.Et, ax=0, use_real=False, shift=False)
        self.f = DSP.spectax(len(self.Et), resolution=A1.t[1]-A1.t[0])
        self.t = NP.fft.fftfreq(len(self.Ef),self.f[1]-self.f[0])        
        self.mode = 'XF'
    
    def __str__(self):
        return ' Instance of class "{0}" in module "{1}" \n label: {2} \n baseline: {3}'.format(self.__class__.__name__, self.__module__, self.label, self.location.__str__())

    def XF(self):
        self.Ef = DSP.FT1D(self.Et, ax=0, use_real=False, shift=False)
        self.f = DSP.spectax(len(self.Et), resolution=self.t[1]-self.t[0])
        self.mode = 'XF'

    def FX(self):
        # Ensure power of 2 and sufficient Nyquist length with zero padding to get identical results to XF-correlator output
        zero_pad_length1 = 2**NP.ceil(NP.log2(len(self.A1.Et)+len(self.A2.Et)-1))-len(self.A1.Et)   # zero pad length for first sequence
        zero_pad_length2 = 2**NP.ceil(NP.log2(len(self.A1.Et)+len(self.A2.Et)-1))-len(self.A2.Et)   # zero pad length for second sequence
        self.Ef = DSP.FT1D(NP.append(self.A1.Et, NP.zeros(zero_pad_length1)), ax=0, use_real=False, shift=False)*NP.conj(DSP.FT1D(NP.append(self.A2.Et, NP.zeros(zero_pad_length2)), ax=0, use_real=False, shift=False))  # product of FT
        self.f = DSP.spectax(int(2**NP.ceil(NP.log2(len(self.A1.Et)+len(self.A2.Et)-1))), resolution=self.A1.t[1]-self.A1.t[0])
        self.mode = 'FX'
        self.t = NP.fft.fftfreq(len(self.Ef),self.f[1]-self.f[0])

################################################################################

class AntennaArray:

    """
    ----------------------------------------------------------------------------
    Class to manage collective information on a group of antennas.

    Attributes:

    antennas:    [Dictionary] Dicitonary consisting of keys which hold instances
                 of class Antenna. The keys themselves are identical to the
                 label attributes of the antenna instances they hold.

    ants_blc_P1  [2-element Numpy array] The coordinates of the bottom left 
                 corner of the array of antennas for polarization P1.

    ants_trc_P1  [2-element Numpy array] The coordinates of the top right 
                 corner of the array of antennas for polarization P1.

    ants_blc_P2  [2-element Numpy array] The coordinates of the bottom left 
                 corner of the array of antennas for polarization P2.

    ants_trc_P2  [2-element Numpy array] The coordinates of the top right 
                 corner of the array of antennas for polarization P2.

    grid_blc_P1  [2-element Numpy array] The coordinates of the bottom left 
                 corner of the grid constructed for the array of antennas
                 for polarization P1. This may differ from ants_blc_P1 due to
                 any extra padding during the gridding process.

    grid_trc_P1  [2-element Numpy array] The coordinates of the top right 
                 corner of the grid constructed for the array of antennas
                 for polarization P1. This may differ from ants_trc_P1 due to
                 any extra padding during the gridding process.

    grid_blc_P2  [2-element Numpy array] The coordinates of the bottom left 
                 corner of the grid constructed for the array of antennas
                 for polarization P2. This may differ from ants_blc_P2 due to
                 any extra padding during the gridding process.

    grid_trc_P2  [2-element Numpy array] The coordinates of the top right 
                 corner of the grid constructed for the array of antennas
                 for polarization P2. This may differ from ants_trc_P2 due to
                 any extra padding during the gridding process.

    gridx_P1     [Numpy array] x-locations of the grid lattice for P1 polarization

    gridy_P1     [Numpy array] y-locations of the grid lattice for P1 polarization

    gridx_P2     [Numpy array] x-locations of the grid lattice for P2 polarization

    gridy_P2     [Numpy array] y-locations of the grid lattice for P2 polarization

    grid_illuminaton_P1
                 [Numpy array] Electric field illumination for P1 polarization 
                 on the grid. Could be complex. Same size as the grid

    grid_illuminaton_P2
                 [Numpy array] Electric field illumination for P2 polarization 
                 on the grid. Could be complex. Same size as the grid

    grid_Ef_P1   [Numpy array] Complex Electric field of polarization P1 
                 projected on the grid. 

    grid_Ef_P2   [Numpy array] Complex Electric field of polarization P2 
                 projected on the grid. 

    f            [Numpy array] Frequency channels (in Hz)

    f0           [Scalar] Center frequency of the observing band (in Hz)

    Member Functions:

    __init__()        Initializes an instance of class AntennaArray which manages
                      information about an array of antennas.
                      
    __str__()         Prints a summary of current attributes
                      
    __add__()         Operator overloading for adding antenna(s)
                      
    __radd__()        Operator overloading for adding antenna(s)
                      
    __sub__()         Operator overloading for removing antenna(s)
                      
    add_antenna()     Routine to add antenna(s) to the antenna array instance. 
                      A wrapper for operator overloading __add__() and __radd__()
                      
    remove_antenna()  Routine to remove antenna(s) from the antenna array 
                      instance. A wrapper for operator overloading __sub__()
                      
    grid()            Routine to produce a grid based on the antenna array 

    grid_convolve()   Routine to project the electric field illumination pattern
                      and the electric fields on the grid. It can operate on the
                      entire antenna array or incrementally project the electric
                      fields and illumination patterns from specific antennas on
                      to an already existing grid.

    grid_unconvolve() Routine to de-project the electric field illumination 
                      pattern and the electric fields on the grid. It can operate 
                      on the entire antenna array or incrementally de-project the 
                      electric fields and illumination patterns from specific 
                      antennas from an already existing grid.

    update():         Updates the antenna array instance with newer attribute
                      values
                      
    save():           Saves the antenna array information to disk. 

    Read the member function docstrings for details.
    ----------------------------------------------------------------------------
    """

    def __init__(self):

        """
        ------------------------------------------------------------------------
        Initialize the AntennaArray Class which manages information about an 
        array of antennas.

        Class attributes initialized are:
        antennas, ants_blc_P1, ants_trc_P1, ants_blc_P2, ant_trc_P2, gridx_P1,
        gridy_P1, gridx_P2, gridy_P2, grid_illumination_P1, 
        grid_illumination_P2, grid_Ef_P1, grid_Ef_P2, f, f0
     
        Read docstring of class AntennaArray for details on these attributes.
        ------------------------------------------------------------------------
        """

        self.antennas = {}
        self.ants_blc_P1 = NP.zeros(2).reshape(1,-1)
        self.ants_trc_P1 = NP.zeros(2).reshape(1,-1)
        self.ants_blc_P2 = NP.zeros(2).reshape(1,-1)
        self.ants_trc_P2 = NP.zeros(2).reshape(1,-1)
        self.grid_blc_P1 = NP.zeros(2).reshape(1,-1)
        self.grid_trc_P1 = NP.zeros(2).reshape(1,-1)
        self.grid_blc_P2 = NP.zeros(2).reshape(1,-1)
        self.grid_trc_P2 = NP.zeros(2).reshape(1,-1)
        self.gridx_P1, self.gridy_P1 = None, None
        self.gridx_P2, self.gridy_P2 = None, None
        self.grid_illumination_P1 = None
        self.grid_illumination_P2 = None
        self.grid_Ef_P1 = None
        self.grid_Ef_P2 = None
        self.f = None
        self.f0 = None
        self.timestamp = None
        
    ################################################################################# 

    def __str__(self):
        printstr = '\n-----------------------------------------------------------------'
        printstr += '\n Instance of class "{0}" in module "{1}".\n Holds the following "Antenna" class instances with labels:\n '.format(self.__class__.__name__, self.__module__)
        printstr += '  '.join(sorted(self.antennas.keys()))
        printstr += '\n Antenna array bounds: blc = [{0[0]}, {0[1]}],\n                       trc = [{1[0]}, {1[1]}]'.format(self.ants_blc_P1, self.ants_trc_P1)
        printstr += '\n Grid bounds: blc = [{0[0]}, {0[1]}],\n              trc = [{1[0]}, {1[1]}]'.format(self.grid_blc_P1, self.grid_trc_P1)
        printstr += '\n-----------------------------------------------------------------'
        return printstr

    ################################################################################# 

    def __add__(self, others):

        """
        ----------------------------------------------------------------------------
        Operator overloading for adding antenna(s)
    
        Inputs:
    
        others     [Instance of class AntennaArray, dictionary holding instance(s)
                   of class Antenna, list of instances of class Antenna, or a single
                   instance of class Antenna] If a dictionary is provided, the keys
                   should be the antenna labels and the values should be instances 
                   of class Antenna. If a list is provided, it should be a list of 
                   valid instances of class Antenna. These instance(s) of class
                   Antenna will be added to the existing instance of AntennaArray
                   class.
        ----------------------------------------------------------------------------
        """

        retval = self
        if isinstance(others, AntennaArray):
            # for k,v in others.antennas.items():
            for k,v in others.antennas.iteritems():
                if k in retval.antennas:
                    print "Antenna {0} already included in the list of antennas.".format(k)
                    print "For updating, use the update() method. Ignoring antenna {0}".format(k)
                else:
                    retval.antennas[k] = v
                    print 'Antenna "{0}" added to the list of antennas.'.format(k)
        elif isinstance(others, dict):
            # for item in others.values():
            for item in others.itervalues():
                if isinstance(item, Antenna):
                    if item.label in retval.antennas:
                        print "Antenna {0} already included in the list of antennas.".format(item.label)
                        print "For updating, use the update() method. Ignoring antenna {0}".format(item.label)
                    else:
                        retval.antennas[item.label] = item
                        print 'Antenna "{0}" added to the list of antennas.'.format(item.label)
        elif isinstance(others, list):
            for i in range(len(others)):
                if isinstance(others[i], Antenna):
                    if others[i].label in retval.antennas:
                        print "Antenna {0} already included in the list of antennas.".format(others[i].label)
                        print "For updating, use the update() method. Ignoring antenna {0}".format(others[i].label)
                    else:
                        retval.antennas[others[i].label] = others[i]
                        print 'Antenna "{0}" added to the list of antennas.'.format(others[i].label)
                else:
                    print 'Element \# {0} is not an instance of class Antenna.'.format(i)
        elif isinstance(others, Antenna):
            if others.label in retval.antennas:
                print "Antenna {0} already included in the list of antennas.".format(others.label)
                print "For updating, use the update() method. Ignoring antenna {0}".format(others[i].label)
            else:
                retval.antennas[others.label] = others
                print 'Antenna "{0}" added to the list of antennas.'.format(others.label)
        else:
            print 'Input(s) is/are not instance(s) of class Antenna.'

        return retval

    ################################################################################# 

    def __radd__(self, others):

        """
        ----------------------------------------------------------------------------
        Operator overloading for adding antenna(s)
    
        Inputs:
    
        others     [Instance of class AntennaArray, dictionary holding instance(s)
                   of class Antenna, list of instances of class Antenna, or a single
                   instance of class Antenna] If a dictionary is provided, the keys
                   should be the antenna labels and the values should be instances 
                   of class Antenna. If a list is provided, it should be a list of 
                   valid instances of class Antenna. These instance(s) of class
                   Antenna will be added to the existing instance of AntennaArray
                   class.
        ----------------------------------------------------------------------------
        """

        return self.__add__(others)

    ################################################################################# 

    def __sub__(self, others):
        """
        ----------------------------------------------------------------------------
        Operator overloading for removing antenna(s)
    
        Inputs:
    
        others     [Instance of class AntennaArray, dictionary holding instance(s)
                   of class Antenna, list of instances of class Antenna, list of
                   strings containing antenna labels or a single instance of class
                   Antenna] If a dictionary is provided, the keys should be the
                   antenna labels and the values should be instances of class
                   Antenna. If a list is provided, it should be a list of valid
                   instances of class Antenna. These instance(s) of class Antenna
                   will be removed from the existing instance of AntennaArray class.
        ----------------------------------------------------------------------------
        """

        retval = self
        if isinstance(others, dict):
            for item in others.values():
                if isinstance(item, Antenna):
                    if item.label not in retval.antennas:
                        print "Antenna {0} does not exist in the list of antennas.".format(item.label)
                    else:
                        del retval.antennas[item.label]
                        print 'Antenna "{0}" removed from the list of antennas.'.format(item.label)
        elif isinstance(others, list):
            for i in range(0,len(others)):
                if isinstance(others[i], str):
                    if others[i] in retval.antennas:
                        del retval.antennas[others[i]]
                        print 'Antenna {0} removed from the list of antennas.'.format(others[i])
                elif isinstance(others[i], Antenna):
                    if others[i].label in retval.antennas:
                        del retval.antennas[others[i].label]
                        print 'Antenna {0} removed from the list of antennas.'.format(others[i].label)
                    else:
                        print "Antenna {0} does not exist in the list of antennas.".format(others[i].label)
                else:
                    print 'Element \# {0} has no matches in the list of antennas.'.format(i)                        
        elif others in retval.antennas:
            del retval.antennas[others]
            print 'Antenna "{0}" removed from the list of antennas.'.format(others)
        elif isinstance(others, Antenna):
            if others.label in retval.antennas:
                del retval.antennas[others.label]
                print 'Antenna "{0}" removed from the list of antennas.'.format(others.label)
            else:
                print "Antenna {0} does not exist in the list of antennas.".format(others.label)
        else:
            print 'No matches found in existing list of antennas.'

        return retval

    ################################################################################# 

    def add_antennas(self, A=None):

        """
        ----------------------------------------------------------------------------
        Routine to add antenna(s) to the antenna array instance. A wrapper for
        operator overloading __add__() and __radd__()
    
        Inputs:
    
        A          [Instance of class AntennaArray, dictionary holding instance(s)
                   of class Antenna, list of instances of class Antenna, or a single
                   instance of class Antenna] If a dictionary is provided, the keys
                   should be the antenna labels and the values should be instances 
                   of class Antenna. If a list is provided, it should be a list of 
                   valid instances of class Antenna. These instance(s) of class
                   Antenna will be added to the existing instance of AntennaArray
                   class.
        ----------------------------------------------------------------------------
        """

        if A is None:
            print 'No antenna(s) supplied.'
        elif isinstance(A, (list, Antenna)):
            self = self.__add__(A)
        else:
            print 'Input(s) is/are not instance(s) of class Antenna.'

    ################################################################################# 

    def remove_antennas(self, A=None):

        """
        ----------------------------------------------------------------------------
        Routine to remove antenna(s) from the antenna array instance. A wrapper for
        operator overloading __sub__()
    
        Inputs:
    
        A          [Instance of class AntennaArray, dictionary holding instance(s)
                   of class Antenna, list of instances of class Antenna, or a single
                   instance of class Antenna] If a dictionary is provided, the keys
                   should be the antenna labels and the values should be instances 
                   of class Antenna. If a list is provided, it should be a list of 
                   valid instances of class Antenna. These instance(s) of class
                   Antenna will be removed from the existing instance of AntennaArray
                   class.
        ----------------------------------------------------------------------------
        """

        if A is None:
            print 'No antenna specified for removal.'
        else:
            self = self.__sub__(A)

    ################################################################################# 

    def antenna_positions(self, sort=False):
        
        if not isinstance(sort, bool):
            raise TypeError('sort keyword has to be a Boolean value.')

        if sort:
            xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in sorted(self.antennas.keys())])
            labels = sorted(self.antennas.keys())
        else:
            xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in self.antennas.keys()])
            labels = self.antennas.keys()

        outdict = {}
        outdict['antennas'] = labels
        outdict['positions'] = xyz

        return outdict

    ################################################################################# 

    def grid(self, uvspacing=0.5, xypad=None, pow2=True, pol=None):

        """
        ----------------------------------------------------------------------------
        Routine to produce a grid based on the antenna array 

        Inputs:

        uvspacing   [Scalar] Positive value indicating the maximum uv-spacing
                    desirable at the lowest wavelength (max frequency). Default = 0.5

        xypad       [List] Padding to be applied around the antenna locations before
                    forming a grid. List elements should be positive. If it is a
                    one-element list, the element is applicable to both x and y axes.
                    If list contains three or more elements, only the first two
                    elements are considered one for each axis. Default = None.

        pow2        [Boolean] If set to True, the grid is forced to have a size a 
                    next power of 2 relative to the actual sie required. If False,
                    gridding is done with the appropriate size as determined by
                    uvspacing. Default = True.

        pol         [String] The polarization to be gridded. Can be set to 'P1' or
                    'P2'. If set to None, gridding for both 'P1' and 'P2' is
                    performed. 
        ----------------------------------------------------------------------------
        """

        if self.f is None:
            self.f = self.antennas.itervalues().next().f

        if self.f0 is None:
            self.f0 = self.antennas.itervalues().next().f0

        if self.timestamp is None:
            self.timestamp = self.antennas.itervalues().next().timestamp

        # if not isinstance(wavelength, NP.ndarray) and not isinstance(wavelength, list) and not isinstance(wavelength, (int,float)):
        #     raise TypeError('wavelength must be a scalar, list or numpy array.')

        # wavelength = NP.asarray(wavelength)

        wavelength = FCNST.c / self.f
        min_lambda = NP.min(NP.abs(wavelength))

        # Change itervalues() to values() when porting to Python 3.x
        # May have to change *blc and *trc with zip(*blc) and zip(*trc) when using Python 3.x

        if (pol is None) or (pol == 'P1'):
            blc_P1 = [[self.antennas[label].blc_P1[0,0], self.antennas[label].blc_P1[0,1]] for label in self.antennas if not self.antennas[label].pol.flag_P1]
            trc_P1 = [[self.antennas[label].trc_P1[0,0], self.antennas[label].trc_P1[0,1]] for label in self.antennas if not self.antennas[label].pol.flag_P1]

            self.ants_blc_P1 = NP.asarray(map(min, *blc_P1))
            self.ants_trc_P1 = NP.asarray(map(max, *trc_P1))

            self.gridx_P1, self.gridy_P1 = GRD.grid_2d([(self.ants_blc_P1[0], self.ants_trc_P1[0]),(self.ants_blc_P1[1], self.ants_trc_P1[1])], pad=xypad, spacing=uvspacing*min_lambda, pow2=True)

            self.grid_blc_P1 = NP.asarray([NP.amin(self.gridx_P1[0,:]), NP.amin(self.gridy_P1[:,0])])
            self.grid_trc_P1 = NP.asarray([NP.amax(self.gridx_P1[0,:]), NP.amax(self.gridy_P1[:,0])])

        if (pol is None) or (pol == 'P2'):
            blc_P2 = [[self.antennas[label].blc_P2[0,0], self.antennas[label].blc_P2[0,1]] for label in self.antennas if not self.antennas[label].pol.flag_P2]
            trc_P2 = [[self.antennas[label].trc_P2[0,0], self.antennas[label].trc_P2[0,1]] for label in self.antennas if not self.antennas[label].pol.flag_P2]

            self.ants_blc_P2 = NP.asarray(map(min, *blc_P2))
            self.ants_trc_P2 = NP.asarray(map(max, *trc_P2))

            self.gridx_P2, self.gridy_P2 = GRD.grid_2d([(self.ants_blc_P2[0], self.ants_trc_P2[0]),(self.ants_blc_P2[1], self.ants_trc_P2[1])], pad=xypad, spacing=uvspacing*min_lambda, pow2=True)

            self.grid_blc_P2 = NP.asarray([NP.amin(self.gridx_P2[0,:]), NP.amin(self.gridy_P2[:,0])])
            self.grid_trc_P2 = NP.asarray([NP.amax(self.gridx_P2[0,:]), NP.amax(self.gridy_P2[:,0])])

    #################################################################################

    def grid_convolve(self, pol=None, ants=None, unconvolve_existing=False,
                      normalize=False, method='NN', distNN=NP.inf): 

        """
        ----------------------------------------------------------------------------
        Routine to project the electric field illumination pattern and the electric
        fields on the grid. It can operate on the entire antenna array or
        incrementally project the electric fields and illumination patterns from
        specific antennas on to an already existing grid.

        Inputs:

        pol         [String] The polarization to be gridded. Can be set to 'P1' or
                    'P2'. If set to None, gridding for both 'P1' and 'P2' is
                    performed. Default = None

        ants        [instance of class AntennaArray, single instance or list of
                    instances of class Antenna, or a dictionary holding instances of
                    of class Antenna] If a dictionary is provided, the keys
                    should be the antenna labels and the values should be instances 
                    of class Antenna. If a list is provided, it should be a list of 
                    valid instances of class Antenna. These instance(s) of class
                    Antenna will be merged to the existing grid contained in the
                    instance of AntennaArray class. If ants is not provided (set to
                    None), the gridding operations will be performed on the entire
                    set of antennas contained in the instance of class AntennaArray.
                    Default = None.

        unconvolve_existing
                   [Boolean] Default = False. If set to True, the effects of
                   gridding convolution contributed by the antenna(s) specified will
                   be undone before updating the antenna measurements on the grid,
                   if the antenna(s) is/are already found to in the set of antennas
                   held by the instance of AntennaArray. If False and if one or more
                   antenna instances specified are already found to be held in the
                   instance of class AntennaArray, the code will stop raising an
                   error indicating the gridding oepration cannot proceed. 

        normalize  [Boolean] Default = False. If set to True, the gridded weights
                   are divided by the sum of weights so that the gridded weights add
                   up to unity. 

        method     [string] The gridding method to be used in applying the antenna
                   weights on to the antenna array grid. Accepted values are 'NN'
                   (nearest neighbour - default), 'CS' (cubic spline), or 'BL'
                   (Bi-linear). In case of applying grid weights by 'NN' method, an
                   optional distance upper bound for the nearest neighbour can be 
                   provided in the parameter distNN to prune the search and make it
                   efficient

        distNN     [scalar] A positive value indicating the upper bound on distance
                   to the nearest neighbour in the gridding process. It has units of
                   distance, the same units as the antenna attribute location and 
                   antenna array attribute gridx_P1 and gridy_P1. Default is NP.inf
                   (infinite distance). It will be internally converted to have same
                   units as antenna attributes wtspos_P1 and wtspos_P2 (units in 
                   number of wavelengths)
        ----------------------------------------------------------------------------
        """

        eps = 1.0e-10

        if (pol is None) or (pol == 'P1'):

            if ants is not None:

                if isinstance(ants, Antenna):
                    ants = [ants]

                if isinstance(ants, (dict, AntennaArray)):
                    # Check if these antennas are new or old and compatible
                    for key in ants: 
                        if isinstance(ants[key], Antenna): # required if ants is a dictionary and not instance of AntennaArray
                            if key in self.antennas:
                                if unconvolve_existing: # Effects on the grid of antennas already existing must be removed 
                                    if self.antennas[key].gridinfo_P1: # if gridding info is not empty
                                        for i in range(len(self.f)):
                                            self.grid_unconvolve(ants[key].label)
                                else:
                                    raise KeyError('Antenna {0} already found to exist in the dictionary of antennas but cannot proceed grid_convolve() without unconvolving first.'.format(ants[key].label)) 
                            
                        else:
                            del ants[key] # remove the dictionary element since it is not an Antenna instance

                    for key in ants:
                        if not ants[key].pol.flag_P1:
                            for i in range(len(self.f)):
                                if method == 'NN':
                                    if ants[key].wtspos_P1_scale is None: 
                                        ibind, nnval = LKP.lookup(ants[key].wtspos_P1[i][:,0] + ants[key].location.x * (self.f[i]/FCNST.c),
                                                                  ants[key].wtspos_P1[i][:,1] + ants[key].location.y * (self.f[i]/FCNST.c),
                                                                  ants[key].wts_P1[i], self.gridx_P1*self.f[i]/FCNST.c,
                                                                  self.gridy_P1*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                  remove_oob=True)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridx_P1.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif ants[key].wtspos_P1_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            ibind, nnval = LKP.lookup(ants[key].wtspos_P1[0][:,0]+ants[key].location.x*(self.f[0]/FCNST.c),
                                                                      ants[key].wtspos_P1[0][:,1]+ants[key].location.y*(self.f[0]/FCNST.c),
                                                                      ants[key].wts_P1[0], self.gridx_P1*self.f[0]/FCNST.c,
                                                                      self.gridy_P1*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                      remove_oob=True)[:2]
                                            roi_ind = NP.unravel_index(ibind, self.gridx_P1.shape)
                                            if normalize:
                                                nnval /= NP.sum(nnval)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P1[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                    self.grid_Ef_P1[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += ants[key].pol.Ef_P1[i] * nnval
                                else:
                                    if ants[key].wtspos_P1_scale is None: 
                                        grid_illumination_P1 = GRD.conv_grid2d(ants[key].location.x * (self.f[i]/FCNST.c),
                                                                               ants[key].location.y * (self.f[i]/FCNST.c),
                                                                               ants[key].wtspos_P1[i][:,0],
                                                                               ants[key].wtspos_P1[i][:,1],
                                                                               ants[key].wts_P1[i],
                                                                               self.gridx_P1 * (self.f[i]/FCNST.c),
                                                                               self.gridy_P1 * (self.f[i]/FCNST.c),
                                                                               method=method)
                                        grid_illumination_P1 = grid_illumination_P1.reshape(self.gridx_P1.shape)
                                        if normalize:
                                            grid_illumination_P1 = grid_illumination_P1 / NP.sum(grid_illumination_P1)
                                        roi_ind = NP.where(NP.abs(grid_illumination_P1) >= eps)
                                    elif ants[key].wtspos_P1_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            grid_illumination_P1 = GRD.conv_grid2d(ants[key].location.x * (self.f[0]/FCNST.c),
                                                                                   ants[key].location.y * (self.f[0]/FCNST.c),
                                                                                   ants[key].wtspos_P1[0][:,0],
                                                                                   ants[key].wtspos_P1[0][:,1],
                                                                                   ants[key].wts_P1[0],
                                                                                   self.gridx_P1 * (self.f[0]/FCNST.c),
                                                                                   self.gridy_P1 * (self.f[0]/FCNST.c),
                                                                                   method=method)
                                            grid_illumination_P1 = grid_illumination_P1.reshape(self.gridx_P1.shape)
                                            if normalize:
                                                grid_illumination_P1 = grid_illumination_P1 / NP.sum(grid_illumination_P1)
                                            roi_ind = NP.where(NP.abs(grid_illumination_P1) >= eps)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P1[:,:,i] += grid_illumination_P1
                                    self.grid_Ef_P1[:,:,i] += ants[key].pol.Ef_P1[i] * grid_illumination_P1

                                if key in self.antennas:
                                    if i not in self.antennas[key].gridinfo_P1:
                                        self.antennas[key].gridinfo_P1 = {} # Create an empty dictionary for each channel to hold grid info
                                    self.antennas[key].gridinfo_P1[i]['f'] = self.f[i]
                                    self.antennas[key].gridinfo_P1[i]['flag'] = False
                                    self.antennas[key].gridinfo_P1[i]['gridxy_ind'] = zip(*roi_ind)
                                    self.antennas[key].wtspos_P1_scale = ants[key].wtspos_P1_scale
                                    if method == 'NN':
                                        self.antennas[key].gridinfo_P1[i]['illumination'] = nnval
                                        self.antennas[key].gridinfo_P1[i]['Ef'] = ants[key].pol.Ef_P1[i] * nnval
                                    else:
                                        self.antennas[key].gridinfo_P1[i]['illumination'] = grid_illumination_P1[roi_ind]
                                        self.antennas[key].gridinfo_P1[i]['Ef'] = ants[key].pol.Ef_P1[i] * grid_illumination_P1[roi_ind]

                elif isinstance(ants, list):
                    # Check if these antennas are new or old and compatible
                    for key in range(len(ants)): 
                        if isinstance(ants[key], Antenna): # required if ants is a dictionary and not instance of AntennaArray
                            if ants[key].label in self.antennas:
                                if unconvolve_existing: # Effects on the grid of antennas already existing must be removed 
                                    if self.antennas[ants[key].label].gridinfo_P1: # if gridding info is not empty
                                        for i in range(len(self.f)):
                                            self.grid_unconvolve(ants[key].label)
                                else:
                                    raise KeyError('Antenna {0} already found to exist in the dictionary of antennas but cannot proceed grid_convolve() without unconvolving first.'.format(ants[key].label))
                            
                        else:
                            del ants[key] # remove the dictionary element since it is not an Antenna instance

                    for key in range(len(ants)):
                        if not ants[key].pol.flag_P1:
                            for i in range(len(self.f)):
                                if method == 'NN':
                                    if ants[key].wtspos_P1_scale is None: 
                                        ibind, nnval = LKP.lookup(ants[key].wtspos_P1[i][:,0] + ants[key].location.x * (self.f[i]/FCNST.c),
                                                                  ants[key].wtspos_P1[i][:,1] + ants[key].location.y * (self.f[i]/FCNST.c),
                                                                  ants[key].wts_P1[i], self.gridx_P1*self.f[i]/FCNST.c,
                                                                  self.gridy_P1*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                  remove_oob=True)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridx_P1.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif ants[key].wtspos_P1_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            ibind, nnval = LKP.lookup(ants[key].wtspos_P1[0][:,0]+ants[key].location.x*(self.f[0]/FCNST.c),
                                                                      ants[key].wtspos_P1[0][:,1]+ants[key].location.y*(self.f[0]/FCNST.c),
                                                                      ants[key].wts_P1[0], self.gridx_P1*self.f[0]/FCNST.c,
                                                                      self.gridy_P1*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                      remove_oob=True)[:2]
                                            roi_ind = NP.unravel_index(ibind, self.gridx_P1.shape)
                                            if normalize:
                                                nnval /= NP.sum(nnval)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P1[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                    self.grid_Ef_P1[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += ants[key].pol.Ef_P1[i] * nnval
                                else:
                                    if ants[key].wtspos_P1_scale is None:
                                        grid_illumination_P1 = GRD.conv_grid2d(ants[key].location.x * (self.f[i]/FCNST.c),
                                                                               ants[key].location.y * (self.f[i]/FCNST.c),
                                                                               ants[key].wtspos_P1[i][:,0],
                                                                               ants[key].wtspos_P1[i][:,1],
                                                                               ants[key].wts_P1[i],
                                                                               self.gridx_P1 * (self.f[i]/FCNST.c),
                                                                               self.gridy_P1 * (self.f[i]/FCNST.c),
                                                                               method=method)
                                        grid_illumination_P1 = grid_illumination_P1.reshape(self.gridx_P1.shape)
                                        if normalize:
                                            grid_illumination_P1 = grid_illumination_P1 / NP.sum(grid_illumination_P1)
                                        roi_ind = NP.where(NP.abs(grid_illumination_P1) >= eps)
                                    elif ants[key].wtspos_P1_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            grid_illumination_P1 = GRD.conv_grid2d(ants[key].location.x * (self.f[0]/FCNST.c),
                                                                                   ants[key].location.y * (self.f[0]/FCNST.c),
                                                                                   ants[key].wtspos_P1[0][:,0],
                                                                                   ants[key].wtspos_P1[0][:,1],
                                                                                   ants[key].wts_P1[0],
                                                                                   self.gridx_P1 * (self.f[0]/FCNST.c),
                                                                                   self.gridy_P1 * (self.f[0]/FCNST.c),
                                                                                   method=method)
                                            grid_illumination_P1 = grid_illumination_P1.reshape(self.gridx_P1.shape)
                                            if normalize:
                                                grid_illumination_P1 = grid_illumination_P1 / NP.sum(grid_illumination_P1)
                                            roi_ind = NP.where(NP.abs(grid_illumination_P1) >= eps)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P1[:,:,i] += grid_illumination_P1
                                    self.grid_Ef_P1[:,:,i] += ants[key].pol.Ef_P1[i] * grid_illumination_P1

                                if ants[key].label in self.antennas:
                                    if i not in self.antennas[key].gridinfo_P1:
                                        self.antennas[key].gridinfo_P1 = {} # Create an empty dictionary for each channel to hold grid info
                                    self.antennas[ants[key].label].gridinfo_P1[i]['f'] = self.f[i]
                                    self.antennas[ants[key].label].gridinfo_P1[i]['flag'] = False
                                    self.antennas[ants[key].label].gridinfo_P1[i]['gridxy_ind'] = zip(*roi_ind)
                                    self.antennas[key].wtspos_P1_scale = ants[key].wtspos_P1_scale
                                    if method == 'NN':
                                        self.antennas[ants[key].label].gridinfo_P1[i]['illumination'] = nnval
                                        self.antennas[ants[key].label].gridinfo_P1[i]['Ef'] = ants[key].pol.Ef_P1[i] * nnval
                                    else:
                                        self.antennas[ants[key].label].gridinfo_P1[i]['illumination'] = grid_illumination_P1[roi_ind]
                                        self.antennas[ants[key].label].gridinfo_P1[i]['Ef'] = ants[key].pol.Ef_P1[i] * grid_illumination_P1[roi_ind] 
                else:
                    raise TypeError('ants must be an instance of AntennaArray, a dictionary of Antenna instances, a list of Antenna instances or an Antenna instance.')

            else:

                self.grid_illumination_P1 = NP.zeros((self.gridx_P1.shape[0],
                                                      self.gridx_P1.shape[1],
                                                      len(self.f)),
                                                     dtype=NP.complex_)
                self.grid_Ef_P1 = NP.zeros((self.gridx_P1.shape[0],
                                            self.gridx_P1.shape[1],
                                            len(self.f)), dtype=NP.complex_)

                for key in self.antennas:
                    if not self.antennas[key].pol.flag_P1:
                        for i in range(len(self.f)):
                            if method == 'NN':
                                if self.antennas[key].wtspos_P1_scale is None: 
                                    ibind, nnval = LKP.lookup(self.antennas[key].wtspos_P1[i][:,0]+self.antennas[key].location.x*(self.f[i]/FCNST.c),
                                                              self.antennas[key].wtspos_P1[i][:,1]+self.antennas[key].location.y*(self.f[i]/FCNST.c),
                                                              self.antennas[key].wts_P1[i], self.gridx_P1*self.f[i]/FCNST.c,
                                                              self.gridy_P1*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                              remove_oob=True)[:2]
                                    roi_ind = NP.unravel_index(ibind, self.gridx_P1.shape)
                                    if normalize:
                                        nnval /= NP.sum(nnval)
                                elif self.antennas[key].wtspos_P1_scale == 'scale':
                                    if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                        ibind, nnval = LKP.lookup(self.antennas[key].wtspos_P1[0][:,0]+self.antennas[key].location.x*(self.f[0]/FCNST.c),
                                                                  self.antennas[key].wtspos_P1[0][:,1]+self.antennas[key].location.y*(self.f[0]/FCNST.c),
                                                                  self.antennas[key].wts_P1[0], self.gridx_P1*self.f[0]/FCNST.c,
                                                                  self.gridy_P1*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                  remove_oob=True)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridx_P1.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                else:
                                    raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                self.grid_illumination_P1[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                self.grid_Ef_P1[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += self.antennas[key].pol.Ef_P1[i] * nnval
                            else:
                                if self.antennas[key].wtspos_P1_scale is None:
                                    grid_illumination_P1 = GRD.conv_grid2d(self.antennas[key].location.x * (self.f[i]/FCNST.c),
                                                                           self.antennas[key].location.y * (self.f[i]/FCNST.c),
                                                                           self.antennas[key].wtspos_P1[i][:,0],
                                                                           self.antennas[key].wtspos_P1[i][:,1],
                                                                           self.antennas[key].wts_P1[i],
                                                                           self.gridx_P1 * (self.f[i]/FCNST.c),
                                                                           self.gridy_P1 * (self.f[i]/FCNST.c),
                                                                           method=method)
                                    grid_illumination_P1 = grid_illumination_P1.reshape(self.gridx_P1.shape)
                                    if normalize:
                                        grid_illumination_P1 = grid_illumination_P1 / NP.sum(grid_illumination_P1)
                                    roi_ind = NP.where(NP.abs(grid_illumination_P1) >= eps)
                                elif self.antennas[key].wtspos_P1_scale == 'scale':
                                    if i == 0:
                                        grid_illumination_P1 = GRD.conv_grid2d(self.antennas[key].location.x * (self.f[0]/FCNST.c),
                                                                               self.antennas[key].location.y * (self.f[0]/FCNST.c),
                                                                               self.antennas[key].wtspos_P1[0][:,0],
                                                                               self.antennas[key].wtspos_P1[0][:,1],
                                                                               self.antennas[key].wts_P1[0],
                                                                               self.gridx_P1 * (self.f[0]/FCNST.c),
                                                                               self.gridy_P1 * (self.f[0]/FCNST.c),
                                                                               method=method)
                                        grid_illumination_P1 = grid_illumination_P1.reshape(self.gridx_P1.shape)
                                        if normalize:
                                            grid_illumination_P1 = grid_illumination_P1 / NP.sum(grid_illumination_P1)
                                        roi_ind = NP.where(NP.abs(grid_illumination_P1) >= eps)
                                else:
                                    raise ValueError('Invalid scale option specified. Aborting grid_convolve().')
                                
                                self.grid_illumination_P1[:,:,i] += grid_illumination_P1
                                self.grid_Ef_P1[:,:,i] += self.antennas[key].pol.Ef_P1[i] * grid_illumination_P1

                            self.antennas[key].gridinfo_P1[i] = {} # Create a nested dictionary to hold channel info
                            self.antennas[key].gridinfo_P1[i]['f'] = self.f[i]
                            self.antennas[key].gridinfo_P1[i]['flag'] = False
                            self.antennas[key].gridinfo_P1[i]['gridxy_ind'] = zip(*roi_ind)
                            if method == 'NN':
                                self.antennas[key].gridinfo_P1[i]['illumination'] = nnval
                                self.antennas[key].gridinfo_P1[i]['Ef'] = self.antennas[key].pol.Ef_P1[i] * nnval  
                            else:
                                self.antennas[key].gridinfo_P1[i]['illumination'] = grid_illumination_P1[roi_ind]
                                self.antennas[key].gridinfo_P1[i]['Ef'] = self.antennas[key].pol.Ef_P1[i] * grid_illumination_P1[roi_ind]

        if (pol is None) or (pol == 'P2'):

            if ants is not None:

                if isinstance(ants, Antenna):
                    ants = [ants]

                if isinstance(ants, (dict, AntennaArray)):
                    # Check if these antennas are new or old and compatible
                    for key in ants: 
                        if isinstance(ants[key], Antenna): # required if ants is a dictionary and not instance of AntennaArray
                            if key in self.antennas:
                                if unconvolve_existing: # Effects on the grid of antennas already existing must be removed 
                                    if self.antennas[key].gridinfo_P2: # if gridding info is not empty
                                        for i in range(len(self.f)):
                                            self.grid_unconvolve(ants[key].label)
                                else:
                                    raise KeyError('Antenna {0} already found to exist in the dictionary of antennas but cannot proceed grid_convolve() without unconvolving first.'.format(ants[key].label)) 
                            
                        else:
                            del ants[key] # remove the dictionary element since it is not an Antenna instance

                    for key in ants:
                        if not ants[key].pol.flag_P2:
                            for i in range(len(self.f)):
                                if method == 'NN':
                                    if ants[key].wtspos_P2_scale is None: 
                                        ibind, nnval = LKP.lookup(ants[key].wtspos_P2[i][:,0] + ants[key].location.x * (self.f[i]/FCNST.c),
                                                                  ants[key].wtspos_P2[i][:,1] + ants[key].location.y * (self.f[i]/FCNST.c),
                                                                  ants[key].wts_P2[i], self.gridx_P2*self.f[i]/FCNST.c,
                                                                  self.gridy_P2*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                  remove_oob=True)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridx_P2.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif ants[key].wtspos_P2_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            ibind, nnval = LKP.lookup(ants[key].wtspos_P2[0][:,0]+ants[key].location.x*(self.f[0]/FCNST.c),
                                                                      ants[key].wtspos_P2[0][:,1]+ants[key].location.y*(self.f[0]/FCNST.c),
                                                                      ants[key].wts_P2[0], self.gridx_P2*self.f[0]/FCNST.c,
                                                                      self.gridy_P2*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                      remove_oob=True)[:2]
                                            roi_ind = NP.unravel_index(ibind, self.gridx_P2.shape)
                                            if normalize:
                                                nnval /= NP.sum(nnval)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P2[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                    self.grid_Ef_P2[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += ants[key].pol.Ef_P2[i] * nnval
                                else:
                                    if ants[key].wtspos_P2_scale is None: 
                                        grid_illumination_P2 = GRD.conv_grid2d(ants[key].location.x * (self.f[i]/FCNST.c),
                                                                               ants[key].location.y * (self.f[i]/FCNST.c),
                                                                               ants[key].wtspos_P2[i][:,0],
                                                                               ants[key].wtspos_P2[i][:,1],
                                                                               ants[key].wts_P2[i],
                                                                               self.gridx_P2 * (self.f[i]/FCNST.c),
                                                                               self.gridy_P2 * (self.f[i]/FCNST.c),
                                                                               method=method)
                                        grid_illumination_P2 = grid_illumination_P2.reshape(self.gridx_P2.shape)
                                        if normalize:
                                            grid_illumination_P2 = grid_illumination_P2 / NP.sum(grid_illumination_P2)
                                        roi_ind = NP.where(NP.abs(grid_illumination_P2) >= eps)
                                    elif ants[key].wtspos_P2_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            grid_illumination_P2 = GRD.conv_grid2d(ants[key].location.x * (self.f[0]/FCNST.c),
                                                                                   ants[key].location.y * (self.f[0]/FCNST.c),
                                                                                   ants[key].wtspos_P2[0][:,0],
                                                                                   ants[key].wtspos_P2[0][:,1],
                                                                                   ants[key].wts_P2[0],
                                                                                   self.gridx_P2 * (self.f[0]/FCNST.c),
                                                                                   self.gridy_P2 * (self.f[0]/FCNST.c),
                                                                                   method=method)
                                            grid_illumination_P2 = grid_illumination_P2.reshape(self.gridx_P2.shape)
                                            if normalize:
                                                grid_illumination_P2 = grid_illumination_P2 / NP.sum(grid_illumination_P2)
                                            roi_ind = NP.where(NP.abs(grid_illumination_P2) >= eps)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P2[:,:,i] += grid_illumination_P2
                                    self.grid_Ef_P2[:,:,i] += ants[key].pol.Ef_P2[i] * grid_illumination_P2

                                if key in self.antennas:
                                    if i not in self.antennas[key].gridinfo_P2:
                                        self.antennas[key].gridinfo_P2 = {} # Create an empty dictionary for each channel to hold grid info
                                    self.antennas[key].gridinfo_P2[i]['f'] = self.f[i]
                                    self.antennas[key].gridinfo_P2[i]['flag'] = False
                                    self.antennas[key].gridinfo_P2[i]['gridxy_ind'] = zip(*roi_ind)
                                    self.antennas[key].wtspos_P2_scale = ants[key].wtspos_P2_scale
                                    if method == 'NN':
                                        self.antennas[key].gridinfo_P2[i]['illumination'] = nnval
                                        self.antennas[key].gridinfo_P2[i]['Ef'] = ants[key].pol.Ef_P2[i] * nnval
                                    else:
                                        self.antennas[key].gridinfo_P2[i]['illumination'] = grid_illumination_P2[roi_ind]
                                        self.antennas[key].gridinfo_P2[i]['Ef'] = ants[key].pol.Ef_P2[i] * grid_illumination_P2[roi_ind]

                elif isinstance(ants, list):
                    # Check if these antennas are new or old and compatible
                    for key in range(len(ants)): 
                        if isinstance(ants[key], Antenna): # required if ants is a dictionary and not instance of AntennaArray
                            if ants[key].label in self.antennas:
                                if unconvolve_existing: # Effects on the grid of antennas already existing must be removed 
                                    if self.antennas[ants[key].label].gridinfo_P2: # if gridding info is not empty
                                        for i in range(len(self.f)):
                                            self.grid_unconvolve(ants[key].label)
                                else:
                                    raise KeyError('Antenna {0} already found to exist in the dictionary of antennas but cannot proceed grid_convolve() without unconvolving first.'.format(ants[key].label))
                            
                        else:
                            del ants[key] # remove the dictionary element since it is not an Antenna instance

                    for key in range(len(ants)):
                        if not ants[key].pol.flag_P2:
                            for i in range(len(self.f)):
                                if method == 'NN':
                                    if ants[key].wtspos_P2_scale is None: 
                                        ibind, nnval = LKP.lookup(ants[key].wtspos_P2[i][:,0] + ants[key].location.x * (self.f[i]/FCNST.c),
                                                                  ants[key].wtspos_P2[i][:,1] + ants[key].location.y * (self.f[i]/FCNST.c),
                                                                  ants[key].wts_P2[i], self.gridx_P2*self.f[i]/FCNST.c,
                                                                  self.gridy_P2*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                  remove_oob=True)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridx_P2.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif ants[key].wtspos_P2_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            ibind, nnval = LKP.lookup(ants[key].wtspos_P2[0][:,0]+ants[key].location.x*(self.f[0]/FCNST.c),
                                                                      ants[key].wtspos_P2[0][:,1]+ants[key].location.y*(self.f[0]/FCNST.c),
                                                                      ants[key].wts_P2[0], self.gridx_P2*self.f[0]/FCNST.c,
                                                                      self.gridy_P2*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                      remove_oob=True)[:2]
                                            roi_ind = NP.unravel_index(ibind, self.gridx_P2.shape)
                                            if normalize:
                                                nnval /= NP.sum(nnval)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P2[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                    self.grid_Ef_P2[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += ants[key].pol.Ef_P2[i] * nnval
                                else:
                                    if ants[key].wtspos_P2_scale is None:
                                        grid_illumination_P2 = GRD.conv_grid2d(ants[key].location.x * (self.f[i]/FCNST.c),
                                                                               ants[key].location.y * (self.f[i]/FCNST.c),
                                                                               ants[key].wtspos_P2[i][:,0],
                                                                               ants[key].wtspos_P2[i][:,1],
                                                                               ants[key].wts_P2[i],
                                                                               self.gridx_P2 * (self.f[i]/FCNST.c),
                                                                               self.gridy_P2 * (self.f[i]/FCNST.c),
                                                                               method=method)
                                        grid_illumination_P2 = grid_illumination_P2.reshape(self.gridx_P2.shape)
                                        if normalize:
                                            grid_illumination_P2 = grid_illumination_P2 / NP.sum(grid_illumination_P2)
                                        roi_ind = NP.where(NP.abs(grid_illumination_P2) >= eps)
                                    elif ants[key].wtspos_P2_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            grid_illumination_P2 = GRD.conv_grid2d(ants[key].location.x * (self.f[0]/FCNST.c),
                                                                                   ants[key].location.y * (self.f[0]/FCNST.c),
                                                                                   ants[key].wtspos_P2[0][:,0],
                                                                                   ants[key].wtspos_P2[0][:,1],
                                                                                   ants[key].wts_P2[0],
                                                                                   self.gridx_P2 * (self.f[0]/FCNST.c),
                                                                                   self.gridy_P2 * (self.f[0]/FCNST.c),
                                                                                   method=method)
                                            grid_illumination_P2 = grid_illumination_P2.reshape(self.gridx_P2.shape)
                                            if normalize:
                                                grid_illumination_P2 = grid_illumination_P2 / NP.sum(grid_illumination_P2)
                                            roi_ind = NP.where(NP.abs(grid_illumination_P2) >= eps)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P2[:,:,i] += grid_illumination_P2
                                    self.grid_Ef_P2[:,:,i] += ants[key].pol.Ef_P2[i] * grid_illumination_P2

                                if ants[key].label in self.antennas:
                                    if i not in self.antennas[key].gridinfo_P2:
                                        self.antennas[key].gridinfo_P2 = {} # Create an empty dictionary for each channel to hold grid info
                                    self.antennas[ants[key].label].gridinfo_P2[i]['f'] = self.f[i]
                                    self.antennas[ants[key].label].gridinfo_P2[i]['flag'] = False
                                    self.antennas[ants[key].label].gridinfo_P2[i]['gridxy_ind'] = zip(*roi_ind)
                                    self.antennas[key].wtspos_P2_scale = ants[key].wtspos_P2_scale
                                    if method == 'NN':
                                        self.antennas[ants[key].label].gridinfo_P2[i]['illumination'] = nnval
                                        self.antennas[ants[key].label].gridinfo_P2[i]['Ef'] = ants[key].pol.Ef_P2[i] * nnval
                                    else:
                                        self.antennas[ants[key].label].gridinfo_P2[i]['illumination'] = grid_illumination_P2[roi_ind]
                                        self.antennas[ants[key].label].gridinfo_P2[i]['Ef'] = ants[key].pol.Ef_P2[i] * grid_illumination_P2[roi_ind] 
                else:
                    raise TypeError('ants must be an instance of AntennaArray, a dictionary of Antenna instances, a list of Antenna instances or an Antenna instance.')

            else:

                self.grid_illumination_P2 = NP.zeros((self.gridx_P2.shape[0],
                                                      self.gridx_P2.shape[1],
                                                      len(self.f)),
                                                     dtype=NP.complex_)
                self.grid_Ef_P2 = NP.zeros((self.gridx_P2.shape[0],
                                            self.gridx_P2.shape[1],
                                            len(self.f)), dtype=NP.complex_)

                for key in self.antennas:
                    if not self.antennas[key].pol.flag_P2:
                        for i in range(len(self.f)):
                            if method == 'NN':
                                if self.antennas[key].wtspos_P2_scale is None: 
                                    ibind, nnval = LKP.lookup(self.antennas[key].wtspos_P2[i][:,0]+self.antennas[key].location.x*(self.f[i]/FCNST.c),
                                                              self.antennas[key].wtspos_P2[i][:,1]+self.antennas[key].location.y*(self.f[i]/FCNST.c),
                                                              self.antennas[key].wts_P2[i], self.gridx_P2*self.f[i]/FCNST.c,
                                                              self.gridy_P2*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                              remove_oob=True)[:2]
                                    roi_ind = NP.unravel_index(ibind, self.gridx_P2.shape)
                                    if normalize:
                                        nnval /= NP.sum(nnval)
                                elif self.antennas[key].wtspos_P2_scale == 'scale':
                                    if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                        ibind, nnval = LKP.lookup(self.antennas[key].wtspos_P2[0][:,0]+self.antennas[key].location.x*(self.f[0]/FCNST.c),
                                                                  self.antennas[key].wtspos_P2[0][:,1]+self.antennas[key].location.y*(self.f[0]/FCNST.c),
                                                                  self.antennas[key].wts_P2[0], self.gridx_P2*self.f[0]/FCNST.c,
                                                                  self.gridy_P2*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                  remove_oob=True)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridx_P2.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                else:
                                    raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                self.grid_illumination_P2[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                self.grid_Ef_P2[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += self.antennas[key].pol.Ef_P2[i] * nnval
                            else:
                                if self.antennas[key].wtspos_P2_scale is None:
                                    grid_illumination_P2 = GRD.conv_grid2d(self.antennas[key].location.x * (self.f[i]/FCNST.c),
                                                                           self.antennas[key].location.y * (self.f[i]/FCNST.c),
                                                                           self.antennas[key].wtspos_P2[i][:,0],
                                                                           self.antennas[key].wtspos_P2[i][:,1],
                                                                           self.antennas[key].wts_P2[i],
                                                                           self.gridx_P2 * (self.f[i]/FCNST.c),
                                                                           self.gridy_P2 * (self.f[i]/FCNST.c),
                                                                           method=method)
                                    grid_illumination_P2 = grid_illumination_P2.reshape(self.gridx_P2.shape)
                                    if normalize:
                                        grid_illumination_P2 = grid_illumination_P2 / NP.sum(grid_illumination_P2)
                                    roi_ind = NP.where(NP.abs(grid_illumination_P2) >= eps)
                                elif self.antennas[key].wtspos_P2_scale == 'scale':
                                    if i == 0:
                                        grid_illumination_P2 = GRD.conv_grid2d(self.antennas[key].location.x * (self.f[0]/FCNST.c),
                                                                               self.antennas[key].location.y * (self.f[0]/FCNST.c),
                                                                               self.antennas[key].wtspos_P2[0][:,0],
                                                                               self.antennas[key].wtspos_P2[0][:,1],
                                                                               self.antennas[key].wts_P2[0],
                                                                               self.gridx_P2 * (self.f[0]/FCNST.c),
                                                                               self.gridy_P2 * (self.f[0]/FCNST.c),
                                                                               method=method)
                                        grid_illumination_P2 = grid_illumination_P2.reshape(self.gridx_P2.shape)
                                        if normalize:
                                            grid_illumination_P2 = grid_illumination_P2 / NP.sum(grid_illumination_P2)
                                        roi_ind = NP.where(NP.abs(grid_illumination_P2) >= eps)
                                else:
                                    raise ValueError('Invalid scale option specified. Aborting grid_convolve().')
                                
                                self.grid_illumination_P2[:,:,i] += grid_illumination_P2
                                self.grid_Ef_P2[:,:,i] += self.antennas[key].pol.Ef_P2[i] * grid_illumination_P2

                            self.antennas[key].gridinfo_P2[i] = {} # Create a nested dictionary to hold channel info
                            self.antennas[key].gridinfo_P2[i]['f'] = self.f[i]
                            self.antennas[key].gridinfo_P2[i]['flag'] = False
                            self.antennas[key].gridinfo_P2[i]['gridxy_ind'] = zip(*roi_ind)
                            if method == 'NN':
                                self.antennas[key].gridinfo_P2[i]['illumination'] = nnval
                                self.antennas[key].gridinfo_P2[i]['Ef'] = self.antennas[key].pol.Ef_P2[i] * nnval  
                            else:
                                self.antennas[key].gridinfo_P2[i]['illumination'] = grid_illumination_P2[roi_ind]
                                self.antennas[key].gridinfo_P2[i]['Ef'] = self.antennas[key].pol.Ef_P2[i] * grid_illumination_P2[roi_ind]

    ################################################################################

    def grid_unconvolve(self, ants, pol=None):

        """
        ----------------------------------------------------------------------------
        Routine to de-project the electric field illumination pattern and the
        electric fields on the grid. It can operate on the entire antenna array or
        incrementally de-project the electric fields and illumination patterns of
        specific antennas from an already existing grid.

        Inputs:

        ants        [instance of class AntennaArray, single instance or list of
                    instances of class Antenna, or a dictionary holding instances of
                    of class Antenna] If a dictionary is provided, the keys
                    should be the antenna labels and the values should be instances 
                    of class Antenna. If a list is provided, it should be a list of 
                    valid instances of class Antenna. These instance(s) of class
                    Antenna will be merged to the existing grid contained in the
                    instance of AntennaArray class. If any of the antennas are not
                    found to be in the already existing set of antennas, an
                    exception is raised accordingly and code execution stops.

        pol         [String] The polarization to be gridded. Can be set to 'P1' or
                    'P2'. If set to None, gridding for both 'P1' and 'P2' is
                    performed. Default = None

        ----------------------------------------------------------------------------
        """

        try:
            ants
        except NameError:
            raise NameError('No antenna(s) supplied.')

        if (pol is None) or (pol == 'P1'):

            if isinstance(ants, (Antenna, str)):
                ants = [ants]

            if isinstance(ants, (dict, AntennaArray)):
                # Check if these antennas are new or old and compatible
                for key in ants: 
                    if isinstance(ants[key], Antenna): # required if ants is a dictionary and not instance of AntennaArray
                        if key in self.antennas:
                            if self.antennas[key].gridinfo_P1: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.antennas[key].gridinfo_P1[i]['gridxy_ind'])
                                    self.grid_illumination_P1[xind, yind, i] -= self.antennas[key].gridinfo_P1[i]['illumination']
                                    self.grid_Ef_P1[xind, yind, i] -= self.antennas[key].gridinfo_P1[i]['Ef']
                                self.antennas[key].gridinfo_P1 = {}
                        else:
                            raise KeyError('Antenna {0} not found to exist in the dictionary of antennas.'.format(ants[key].label))
                                
            elif isinstance(ants, list):
                # Check if these antennas are new or old and compatible
                for key in range(len(ants)): 
                    if isinstance(ants[key], Antenna): # required if ants is a dictionary and not instance of AntennaArray
                        if ants[key].label in self.antennas:
                            if self.antennas[ants[key].label].gridinfo_P1: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.antennas[ants[key].label].gridinfo_P1[i]['gridxy_ind'])
                                    self.grid_illumination_P1[xind, yind, i] -= self.antennas[ants[key].label].gridinfo_P1[i]['illumination']
                                    self.grid_Ef_P1[xind, yind, i] -= self.antennas[ants[key].label].gridinfo_P1[i]['Ef']
                                self.antennas[ants[key].label].gridinfo_P1 = {}
                        else:
                            raise KeyError('Antenna {0} not found to exist in the dictionary of antennas.'.format(ants[key].label))
                    elif isinstance(ants[key], str):
                        if ants[key] in self.antennas:
                            if self.antennas[ants[key]].gridinfo_P1: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.antennas[ants[key]].gridinfo_P1[i]['gridxy_ind'])
                                    self.grid_illumination_P1[xind, yind, i] -= self.antennas[ants[key]].gridinfo_P1[i]['illumination']
                                    self.grid_Ef_P1[xind, yind, i] -= self.antennas[ants[key]].gridinfo_P1[i]['Ef']
                                self.antennas[ants[key]].gridinfo_P1 = {}
                        else:
                            raise KeyError('Antenna {0} not found to exist in the dictionary of antennas.'.format(ants[key]))
                    else:
                        raise TypeError('ants must be an instance of class AntennaArray, a list of instances of class Antenna, a dictionary of instances of class Antenna or a list of antenna labels.')
            else:
                raise TypeError('ants must be an instance of AntennaArray, a dictionary of Antenna instances, a list of Antenna instances, an Antenna instance, or a list of antenna labels.')

        if (pol is None) or (pol == 'P2'):

            if isinstance(ants, (Antenna, str)):
                ants = [ants]

            if isinstance(ants, (dict, AntennaArray)):
                # Check if these antennas are new or old and compatible
                for key in ants: 
                    if isinstance(ants[key], Antenna): # required if ants is a dictionary and not instance of AntennaArray
                        if key in self.antennas:
                            if self.antennas[key].gridinfo_P2: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.antennas[key].gridinfo_P2[i]['gridxy_ind'])
                                    self.grid_illumination_P2[xind, yind, i] -= self.antennas[key].gridinfo_P2[i]['illumination']
                                    self.grid_Ef_P2[xind, yind, i] -= self.antennas[key].gridinfo_P2[i]['Ef']
                                self.antennas[key].gridinfo_P2 = {}
                        else:
                            raise KeyError('Antenna {0} not found to exist in the dictionary of antennas.'.format(ants[key].label))
                                
            elif isinstance(ants, list):
                # Check if these antennas are new or old and compatible
                for key in range(len(ants)): 
                    if isinstance(ants[key], Antenna): # required if ants is a dictionary and not instance of AntennaArray
                        if ants[key].label in self.antennas:
                            if self.antennas[ants[key].label].gridinfo_P2: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.antennas[ants[key].label].gridinfo_P2[i]['gridxy_ind'])
                                    self.grid_illumination_P2[xind, yind, i] -= self.antennas[ants[key].label].gridinfo_P2[i]['illumination']
                                    self.grid_Ef_P2[xind, yind, i] -= self.antennas[ants[key].label].gridinfo_P2[i]['Ef']
                                self.antennas[ants[key].label].gridinfo_P2 = {}
                        else:
                            raise KeyError('Antenna {0} not found to exist in the dictionary of antennas.'.format(ants[key].label))
                    elif isinstance(ants[key], str):
                        if ants[key] in self.antennas:
                            if self.antennas[ants[key]].gridinfo_P2: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.antennas[ants[key]].gridinfo_P2[i]['gridxy_ind'])
                                    self.grid_illumination_P2[xind, yind, i] -= self.antennas[ants[key]].gridinfo_P2[i]['illumination']
                                    self.grid_Ef_P2[xind, yind, i] -= self.antennas[ants[key]].gridinfo_P2[i]['Ef']
                                self.antennas[ants[key]].gridinfo_P2 = {}
                        else:
                            raise KeyError('Antenna {0} not found to exist in the dictionary of antennas.'.format(ants[key]))
                    else:
                        raise TypeError('ants must be an instance of class AntennaArray, a list of instances of class Antenna, a dictionary of instances of class Antenna or a list of antenna labels.')
            else:
                raise TypeError('ants must be an instance of AntennaArray, a dictionary of Antenna instances, a list of Antenna instances, an Antenna instance, or a list of antenna labels.')

    ##################################################################################

    def update(self, updates=None, verbose=False):

        """
        ----------------------------------------------------------------------------
        Updates the antenna array instance with newer attribute values. Can also be
        used to add and/or remove antennas with/without affecting the existing grid.

        Inputs:

        updates     [Dictionary] Consists of information updates. One of the keys is
                    'label' which indicates an antenna label. If absent, the code
                    execution stops by throwing an exception. The other optional
                    keys and the information they hold are listed below:
                    'action'      [String scalar] Indicates the type of update
                                  operation. 'add' adds the Antenna instance
                                  to the AntennaArray instance. 'remove' removes the
                                  antenna from teh antenna array instance. 'modify' 
                                  modifies the antenna attributes in the antenna 
                                  array instance. This key has to be set. No default.
                    'grid_action' [Boolean] If set to True, will apply the grdding
                                  operations (grid(), grid_convolve(), and 
                                  grid_unconvolve()) appropriately according to the
                                  value of the 'action' key. If set to None or
                                  False, gridding effects will remain unchanged.
                                  Default=None=False.
                    'antenna'     [instance of class Antenna] Updated Antenna
                                  class instance. Can work for action key 'remove' 
                                  even if not set (=None) or set to an empty
                                  string '' as long as 'label' key is specified. 
                    'gridpol'     [Optional. String scalar] Initiates the specified
                                  action on polarization 'P1' or 'P2'. Can be set
                                  to 'P1' or 'P2'. If not provided (=None), then 
                                  the specified action applies to both polarizations.
                                  Default = None.
                    'Et_P1'       [Optional. Numpy array] Complex Electric field 
                                  time series in polarization P1. Is used only if 
                                  set and if 'action' key value is set to 'modify'.
                                  Default = None.
                    'Et_P2'       [Optional. Numpy array] Complex Electric field 
                                  time series in polarization P2. Is used only if 
                                  set and if 'action' key value is set to 'modify'.
                                  Default = None.
                    't'           [Optional. Numpy array] Time axis of the time 
                                  series. Is used only if set and if 'action' key
                                  value is set to 'modify'. Default = None.
                    'timestamp'   [Optional. Scalar] Unique identifier of the time 
                                  series. Is used only if set and if 'action' key
                                  value is set to 'modify'. Default = None.
                    'location'    [Optional. instance of GEOM.Point class] Antenna
                                  location in the local ENU coordinate system. Is
                                  used only if set and if 'action' key value is set
                                  to 'modify'. Default = None.
                    'wtsinfo_P1'  [Optional. List of dictionaries] See 
                                  description in Antenna class member function 
                                  update(). Is used only if set and if 'action' 
                                  key value is set to 'modify'. Default = None.
                    'wtsinfo_P2'  [Optional. List of dictionaries] See 
                                  description in Antenna class member function 
                                  update(). Is used only if set and if 'action' 
                                  key value is set to 'modify'. Default = None.
                    'flag_P1'     [Optional. Boolean] Flagging status update for
                                  polarization P1 of the antenna. If set to True, 
                                  polarization P1 measurements of the antenna will
                                  be flagged. If not set (=None), the previous or
                                  default flag status will continue to apply. If set 
                                  to False, the antenna status will be updated to 
                                  become unflagged. Default = None.
                    'flag_P2'     [Optional. Boolean] Flagging status update for
                                  polarization P2 of the antenna. If set to True, 
                                  polarization P2 measurements of the antenna will
                                  be flagged. If not set (=None), the previous or
                                  default flag status will continue to apply. If set 
                                  to False, the antenna status will be updated to 
                                  become unflagged. Default = None.
                    'gridfunc_freq'
                                  [Optional. String scalar] Read the description of
                                  inputs to Antenna class member function update().
                                  If set to None (not provided), this attribute is
                                  determined based on the size of wtspos_P1 and 
                                  wtspos_P2. It is applicable only when 'action'
                                  key is set to 'modify'. Default = None.
                    'ref_freq'    [Optional. Scalar] Positive value (in Hz) of
                                  reference frequency (used if gridfunc_freq is
                                  set to 'scale') at which wtspos_P1 and wtspos_P2
                                  in wtsinfo_P1 and wtsinfo_P2, respectively, are
                                  provided. If set to None, the referene frequency
                                  already set in antenna array instance remains
                                  unchanged. Default = None.
                    'pol_type'    [Optional. String scalar] 'Linear' or 'Circular'.
                                  Used only when action key is set to 'modify'. If 
                                  not provided, then the previous value remains in
                                  effect. Default = None.
                    'norm_wts'    [Optional. Boolean] Default = False. If set to
                                  True, the gridded weights are divided by the sum
                                  of weights so that the gridded weights add up to
                                  unity. This is used only when grid_action keyword
                                  is set when action keyword is set to 'add' or 
                                  'modify'
                    'gridmethod'  [Optional. String] Indicates gridding method. It
                                  accepts the following values 'NN' (nearest 
                                  neighbour), 'BL' (Bi-linear interpolation), and
                                  'CS' (Cubic Spline interpolation). Default = 'NN'
                    'distNN'      [Optional. Scalar] Indicates the upper bound on
                                  distance for a nearest neighbour search if the
                                  value of 'gridmethod' is set to 'NN'. The units 
                                  are of physical distance, the same as what is used
                                  for antenna locations. Default = NP.inf
        
        verbose     [Boolean] Default = False. If set to True, prints some 
                    diagnotic or progress messages.

        ----------------------------------------------------------------------------
        """

        if updates is not None:
            if not isinstance(updates, list):
                updates = [updates]
            for dictitem in updates:
                if not isinstance(dictitem, dict):
                    raise TypeError('Updates to {0} instance should be provided in the form of a list of dictionaries.'.format(self.__class__.__name__))
                elif 'label' not in dictitem:
                    raise KeyError('No antenna label specified in the dictionary item to be updated.')

                if 'action' not in dictitem:
                    raise KeyError('No action specified for update. Action key should be set to "add", "remove" or "modify".')
                elif dictitem['action'] == 'add':
                    if dictitem['label'] in self.antennas:
                        if verbose:
                            print 'Antenna {0} for adding already exists in current instance of {1}. Skipping over to the next item to be updated.'.format(dictitem['label'], self.__class__.__name__)
                    else:
                        if verbose:
                            print 'Adding antenna {0}...'.format(dictitem['label'])
                        self.add_antennas(dictitem['antenna'])
                        if 'grid_action' in dictitem:
                            self.grid_convolve(pol=dictitem['gridpol'], ants=dictitem['antenna'], unconvolve_existing=False)
                elif dictitem['action'] == 'remove':
                    if dictitem['label'] not in self.antennas:
                        if verbose:
                            print 'Antenna {0} for removal not found in current instance of {1}. Skipping over to the next item to be updated.'.format(dictitem['label'], self.__class__.__name__) 
                    else:
                        if verbose:
                            print 'Removing antenna {0}...'.format(dictitem['label'])
                        if 'grid_action' in dictitem:
                            self.grid_unconvolve(dictitem['label'], dictitem['gridpol'])
                        self.remove_antennas(dictitem['label'])
                elif dictitem['action'] == 'modify':
                    if dictitem['label'] not in self.antennas:
                        if verbose:
                            print 'Antenna {0} for modification not found in current instance of {1}. Skipping over to the next item to be updated.'.format(dictitem['label'], self.__class__.__name__)
                    else:
                        if verbose:
                            print 'Modifying antenna {0}...'.format(dictitem['label'])
                        if 'Et_P1' not in dictitem: dictitem['Et_P1']=None
                        if 'Et_P2' not in dictitem: dictitem['Et_P2']=None
                        if 't' not in dictitem: dictitem['t']=None
                        if 'timestamp' not in dictitem: dictitem['timestamp']=None
                        if 'location' not in dictitem: dictitem['location']=None
                        if 'wtsinfo_P1' not in dictitem: dictitem['wtsinfo_P1']=None
                        if 'wtsinfo_P2' not in dictitem: dictitem['wtsinfo_P2']=None
                        if 'flag_P1' not in dictitem: dictitem['flag_P1']=None
                        if 'flag_P2' not in dictitem: dictitem['flag_P2']=None
                        if 'gridfunc_freq' not in dictitem: dictitem['gridfunc_freq']=None
                        if 'ref_freq' not in dictitem: dictitem['ref_freq']=None
                        if 'pol_type' not in dictitem: dictitem['pol_type']=None
                        if 'norm_wts' not in dictitem: dictitem['norm_wts']=False
                        if 'gridmethod' not in dictitem: dictitem['gridmethod']='NN'
                        if 'distNN' not in dictitem: dictitem['distNN']=NP.inf
                        self.antennas[dictitem['label']].update(dictitem['label'], dictitem['Et_P1'], dictitem['Et_P2'], dictitem['t'], dictitem['timestamp'], dictitem['location'], dictitem['wtsinfo_P1'], dictitem['wtsinfo_P2'], dictitem['flag_P1'], dictitem['flag_P2'], dictitem['gridfunc_freq'], dictitem['ref_freq'], dictitem['pol_type'], verbose)
                        if 'gric_action' in dictitem:
                            self.grid_convolve(pol=dictitem['gridpol'], ants=dictitem['antenna'], unconvolve_existing=True, normalize=dictitem['norm_wts'], method=dictitem['gridmethod'], distNN=dictitem['distNN'])
                else:
                    raise ValueError('Update action should be set to "add", "remove" or "modify".')

    #############################################################################

    def save(self, gridfile, pol=None, tabtype='BinTableHDU', antenna_save=True, 
             antfile=None, overwrite=False, verbose=True):

        """
        -------------------------------------------------------------------------
        Saves the antenna array information to disk. 

        Input:

        gridfile     [string] grid filename with full path. Will be appended 
                     with '.fits' extension

        Keyword Input(s):

        pol          [string] indicates which polarization information to be 
                     saved. Allowed values are 'P1', 'P2' or None (default). If 
                     None, information on both polarizations are saved.
                     
        tabtype      [string] indicates table type for one of the extensions in 
                     the FITS file. Allowed values are 'BinTableHDU' and 
                     'TableHDU' for binary ascii tables respectively. Default is
                     'BinTableHDU'.

        antenna_save [boolean] indicates if information on individual antennas is
                     to be saved. If True (default), individual antenna
                     information is saved into filename given by antfile. If
                     False, only grid information is saved.

        antfile      [string] Filename to save the antenna information to. This 
                     is appended with the antenna label and '.fits' extension. 
                     If not provided, gridfile is used as the basename. antfile 
                     is used only if antenna_save is set to True.

        overwrite    [boolean] True indicates overwrite even if a file already 
                     exists. Default = False (does not overwrite)
                     
        verbose      [boolean] If True (default), prints diagnostic and progress
                     messages. If False, suppress printing such messages.
        -------------------------------------------------------------------------
        """

        try:
            gridfile
        except NameError:
            raise NameError('No filename provided. Aborting AntennaArray.save().')

        filename = gridfile + '.fits'

        if verbose:
            print '\nSaving antenna array information...'
            
        hdulist = []
        hdulist += [fits.PrimaryHDU()]
        hdulist[0].header['f0'] = (self.f0, 'Center frequency (Hz)')
        hdulist[0].header['tobs'] = (self.timestamp, 'Timestamp associated with observation.')
        hdulist[0].header['EXTNAME'] = 'PRIMARY'

        if verbose:
            print '\tCreated a primary HDU.'

        antpos_info = self.antenna_positions(sort=True)
        cols = []
        cols += [fits.Column(name='Antenna', format='8A', array=NP.asarray(antpos_info['antennas']))]
        cols += [fits.Column(name='Position', format='3D', array=antpos_info['positions'])]
        columns = fits.ColDefs(cols, tbtype=tabtype)
        tbhdu = fits.new_table(columns)
        tbhdu.header.set('EXTNAME', 'Antenna Positions')
        hdulist += [tbhdu]
        if verbose:
            print '\tCreated an extension in Binary table format for antenna positions.'

        hdulist += [fits.ImageHDU(self.f, name='FREQ')]
        if verbose:
            print '\t\tCreated an extension HDU of {0:0d} frequency channels'.format(len(self.f))

        if (pol is None) or (pol == 'P1'):
            if verbose:
                print '\tWorking on polarization P1...'
            if self.gridx_P1 is not None:
                hdulist += [fits.ImageHDU(self.gridx_P1, name='gridx_P1')]
                if verbose:
                    print '\t\tCreated an extension HDU of x-coordinates of grid of size: {0[0]}x{0[1]}'.format(self.gridx_P1.shape)
            if self.gridy_P1 is not None:
                hdulist += [fits.ImageHDU(self.gridy_P1, name='gridy_P1')]
                if verbose:
                    print '\t\tCreated an extension HDU of y-coordinates of grid of size: {0[0]}x{0[1]}'.format(self.gridy_P1.shape)
            if self.grid_illumination_P1 is not None:
                hdulist += [fits.ImageHDU(self.grid_illumination_P1.real, name='grid_illumination_P1_real')]
                hdulist += [fits.ImageHDU(self.grid_illumination_P1.imag, name='grid_illumination_P1_imag')]
                if verbose:
                    print "\t\tCreated separate extension HDUs of grid's illumination pattern \n\t\t\twith size {0[0]}x{0[1]}x{0[2]} for real and imaginary parts.".format(self.grid_illumination_P1.shape)
            if self.grid_Ef_P1 is not None:
                hdulist += [fits.ImageHDU(self.grid_Ef_P1.real, name='grid_Ef_P1_real')]
                hdulist += [fits.ImageHDU(self.grid_Ef_P1.imag, name='grid_Ef_P1_imag')]
                if verbose:
                    print "\t\tCreated separate extension HDUs of grid's Electric field spectra of \n\t\t\tsize {0[0]}x{0[1]}x{0[2]} for real and imaginary parts.".format(self.grid_Ef_P1.shape)

        if (pol is None) or (pol == 'P2'):
            if verbose:
                print '\tWorking on polarization P2...'
            if self.gridx_P2 is not None:
                hdulist += [fits.ImageHDU(self.gridx_P2, name='gridx_P2')]
                if verbose:
                    print '\t\tCreated an extension HDU of x-coordinates of grid of size: {0[0]}x{0[1]}'.format(self.gridx_P2.shape)
            if self.gridy_P2 is not None:
                hdulist += [fits.ImageHDU(self.gridy_P2, name='gridy_P2')]
                if verbose:
                    print '\t\tCreated an extension HDU of y-coordinates of grid of size: {0[0]}x{0[1]}'.format(self.gridy_P2.shape)
            if self.grid_illumination_P2 is not None:
                hdulist += [fits.ImageHDU(self.grid_illumination_P2.real, name='grid_illumination_P2_real')]
                hdulist += [fits.ImageHDU(self.grid_illumination_P2.imag, name='grid_illumination_P2_imag')]
                if verbose:
                    print "\t\tCreated separate extension HDUs of grid's illumination pattern \n\t\t\twith size {0[0]}x{0[1]}x{0[2]} for real and imaginary parts.".format(self.grid_illumination_P2.shape)
            if self.grid_Ef_P2 is not None:
                hdulist += [fits.ImageHDU(self.grid_Ef_P2.real, name='grid_Ef_P2_real')]
                hdulist += [fits.ImageHDU(self.grid_Ef_P2.imag, name='grid_Ef_P2_imag')]
                if verbose:
                    print "\t\tCreated separate extension HDUs of grid's Electric field spectra of \n\t\t\tsize {0[0]}x{0[1]}x{0[2]} for real and imaginary parts.".format(self.grid_Ef_P2.shape)

        if verbose:
            print '\tNow writing FITS file to disk:\n\t\t{0}'.format(filename)

        hdu = fits.HDUList(hdulist)
        hdu.writeto(filename, clobber=overwrite)

        if verbose:
            print '\tGridding data written successfully to FITS file on disk:\n\t\t{0}\n'.format(filename)

        if antenna_save:
            if antfile is None:
                antfile = gridfile
            for label in self.antennas:
                if verbose:
                    print 'Now calling save() method of antenna {0}...'.format(label)
                self.antennas[label].save(antfile, tabtype=tabtype,
                                          overwrite=overwrite, verbose=verbose)
            if verbose:
                print 'Successfully completed save() operation.'

#################################################################################
        
class Image:

    """
    -----------------------------------------------------------------------------
    Class to manage image information and processing pertaining to the class 
    holding antenna array information.

    Attributes:

    timestamp:  [Scalar] String or float representing the timestamp for the 
                current attributes

    f:          [vector] Frequency channels (in Hz)

    f0:         [Scalar] Positive value for the center frequency in Hz.

    gridx_P1     [Numpy array] x-locations of the grid lattice for P1 polarization

    gridy_P1     [Numpy array] y-locations of the grid lattice for P1 polarization

    gridx_P2     [Numpy array] x-locations of the grid lattice for P2 polarization

    gridy_P2     [Numpy array] y-locations of the grid lattice for P2 polarization

    grid_illuminaton_P1
                 [Numpy array] Electric field illumination for P1 polarization 
                 on the grid. Could be complex. Same size as the grid

    grid_illuminaton_P2
                 [Numpy array] Electric field illumination for P2 polarization 
                 on the grid. Could be complex. Same size as the grid

    grid_Ef_P1   [Numpy array] Complex Electric field of polarization P1 
                 projected on the grid. 

    grid_Ef_P2   [Numpy array] Complex Electric field of polarization P2 
                 projected on the grid. 
    
    holograph_PB_P1
                 [Numpy array] Complex holographic electric field pattern on sky
                 for polarization P1. Obtained by inverse fourier transforming 
                 grid_illumination_P1. It is 3-dimensional (third dimension is 
                 the frequency axis)

    holograph_P1 [Numpy array] Complex holographic image cube for polarization P1
                 obtained by inverse fourier transforming Ef_P1

    PB_P1        [Numpy array] Power pattern of the antenna obtained by squaring
                 the absolute value of holograph_PB_P1. It is 3-dimensional 
                 (third dimension is the frequency axis)

    lf_P1        [Numpy array] 3D grid of l-axis in the direction cosines 
                 coordinate system corresponding to polarization P1, the third 
                 axis being along frequency.

    mf_P1        [Numpy array] 3D grid of m-axis in the direction cosines 
                 coordinate system corresponding to polarization P1, the third 
                 axis being along frequency.

    img_P1       [Numpy array] 3D image cube obtained by squaring the absolute 
                 value of holograph_P1. The third dimension is along frequency.

    holograph_PB_P2
                 [Numpy array] Complex holographic electric field pattern on sky
                 for polarization P2. Obtained by inverse fourier transforming 
                 grid_illumination_P2. It is 3-dimensional (third dimension is 
                 the frequency axis)

    holograph_P2 [Numpy array] Complex holographic image cube for polarization P2
                 obtained by inverse fourier transforming Ef_P2

    PB_P2        [Numpy array] Power pattern of the antenna obtained by squaring
                 the absolute value of holograph_PB_P2. It is 3-dimensional 
                 (third dimension is the frequency axis)

    lf_P2        [Numpy array] 3D grid of l-axis in the direction cosines 
                 coordinate system corresponding to polarization P2, the third 
                 axis being along frequency.

    mf_P2        [Numpy array] 3D grid of m-axis in the direction cosines 
                 coordinate system corresponding to polarization P2, the third 
                 axis being along frequency.

    img_P2       [Numpy array] 3D image cube obtained by squaring the absolute 
                 value of holograph_P2. The third dimension is along frequency.

    Member Functions:

    __init__()   Initializes an instance of class Image which manages information
                 and processing of images from data obtained by an antenna array.
                 It can be initialized either by values in an instance of class 
                 AntennaArray, by values in a fits file containing information
                 about the antenna array, or to defaults.

    imagr()      Imaging engine that performs inverse fourier transforms of 
                 appropriate electric field quantities associated with the 
                 antenna array.

    save()       Saves the image information to disk

    Read the member function docstrings for more details
    -----------------------------------------------------------------------------
    """

    def __init__(self, f0=None, f=None, pol=None, antenna_array=None,
                 infile=None, timestamp=None, verbose=True):
        
        """
        -------------------------------------------------------------------------
        Initializes an instance of class Image which manages information and
        processing of images from data obtained by an antenna array. It can be
        initialized either by values in an instance of class AntennaArray, by
        values in a fits file containing information about the antenna array, or
        to defaults.

        Class attributes initialized are:
        timestamp, f, f0, gridx_P1, gridy_P1, grid_illumination_P1, grid_Ef_P1, 
        holograph_P1, holograph_PB_P1, img_P1, PB_P1, lf_P1, mf_P1, gridx_P1,
        gridy_P1, grid_illumination_P1, grid_Ef_P1, holograph_P1,
        holograph_PB_P1, img_P1, PB_P1, lf_P1, and mf_P1

        Read docstring of class Image for details on these attributes.
        -------------------------------------------------------------------------
        """

        if verbose:
            print '\nInitializing an instance of class Image...\n'
            print '\tVerifying for compatible arguments...'

        if timestamp is not None:
            self.timestamp = timestamp
            if verbose:
                print '\t\tInitialized time stamp.'

        if f0 is not None:
            self.f0 = f0
            if verbose:
                print '\t\tInitialized center frequency.'

        if f is not None:
            self.f = NP.asarray(f)
            if verbose:
                print '\t\tInitialized frequency channels.'

        if (infile is None) and (antenna_array is None):
            self.gridx_P1 = None
            self.gridy_P1 = None
            self.grid_illumination_P1 = None
            self.grid_Ef_P1 = None
            self.holograph_P1 = None
            self.holograph_PB_P1 = None
            self.img_P1 = None
            self.PB_P1 = None
            self.lf_P1 = None
            self.mf_P1 = None

            self.gridx_P2 = None
            self.gridy_P2 = None
            self.grid_illumination_P2 = None
            self.grid_Ef_P2 = None
            self.holograph_P2 = None
            self.holograph_PB_P2 = None
            self.img_P2 = None
            self.PB_P2 = None
            self.lf_P2 = None
            self.mf_P2 = None
        
            if verbose:
                print '\t\tInitialized gridx_P1, gridy_P1, grid_illumination_P1, and grid_Ef_P1'
                print '\t\tInitialized lf_P1, mf_P1, holograph_PB_P1, PB_P1, holograph_P1, and img_P1'
                print '\t\tInitialized gridx_P2, gridy_P2, grid_illumination_P2, and grid_Ef_P2'
                print '\t\tInitialized lf_P2, mf_P2, holograph_PB_P2, PB_P2, holograph_P2, and img_P2'

        if (infile is not None) and (antenna_array is not None):
            raise ValueError('Both gridded data file and antenna array informtion are specified. One and only one of these should be specified. Cannot initialize an instance of class Image.')     

        if verbose:
            print '\tArguments verified for initialization.'

        if infile is not None:
            if verbose:
                print '\tInitializing from input file...'

            try:
                hdulist = fits.open(infile)
            except IOError:
                raise IOError('File not found. Image instance not initialized.')
            except EOFError:
                raise EOFError('EOF encountered. File cannot be read. Image instance not initialized.')
            else:
                extnames = [hdu.header['EXTNAME'] for hdu in hdulist]
                if verbose:
                    print '\t\tFITS file opened successfully. The extensions have been read.'

                if 'FREQ' in extnames:
                    self.f = hdulist['FREQ'].data
                    if verbose:
                        print '\t\t\tInitialized frequency channels.'
                else:
                    raise KeyError('Frequency information unavailable in the input file.')

                if 'f0' in hdulist[0].header:
                    self.f0 = hdulist[0].header['f0']
                    if verbose:
                        print '\t\t\tInitialized center frequency to {0} Hz from FITS header.'.format(self.f0)
                else:
                    self.f0 = self.f[int(len(self.f)/2)]
                    if verbose:
                        print '\t\t\tNo center frequency found in FITS header. Setting it to \n\t\t\t\tthe center of frequency channels: {0} Hz'.format(self.f0)

                if 'tobs' in hdulist[0].header:
                    self.timestamp = hdulist[0].header['tobs']
                    if verbose:
                        print '\t\t\tInitialized time stamp.'

                if (pol is None) or (pol == 'P1'):
                    if verbose:
                        print '\n\t\t\tWorking on polarization P1...'

                    if ('GRIDX_P1' not in extnames) or ('GRIDY_P1' not in extnames) or ('GRID_ILLUMINATION_P1_REAL' not in extnames) or ('GRID_ILLUMINATION_P1_IMAG' not in extnames) or ('GRID_EF_P1_REAL' not in extnames) or ('GRID_EF_P1_IMAG' not in extnames):
                        raise KeyError('One or more pieces of gridding information is missing in the input file for polarization P1. Verify the file contains appropriate data.')

                    self.gridx_P1 = hdulist['GRIDX_P1'].data
                    self.gridy_P1 = hdulist['GRIDY_P1'].data
                    self.grid_illumination_P1 = hdulist['GRID_ILLUMINATION_P1_REAL'].data + 1j * hdulist['GRID_ILLUMINATION_P1_IMAG'].data
                    self.grid_Ef_P1 = hdulist['GRID_EF_P1_REAL'].data + 1j * hdulist['GRID_EF_P1_IMAG'].data
                    self.holograph_P1 = None
                    self.img_P1 = None
                    self.holograph_PB_P1 = None
                    self.PB_P1 = None
                    self.lf_P1 = None
                    self.mf_P1 = None
                    if verbose:
                        print '\t\t\tInitialized gridx_P1, gridy_P1, grid_illumination_P1, and grid_Ef_P1'
                        print '\t\t\tInitialized lf_P1, mf_P1, holograph_PB_P1, PB_P1, holograph_P1, and img_P1'

                if (pol is None) or (pol == 'P2'):
                    if verbose:
                        print '\n\t\t\tWorking on polarization P2...'

                    if ('GRIDX_P2' not in extnames) or ('GRIDY_P2' not in extnames) or ('GRID_ILLUMINATION_P2_REAL' not in extnames) or ('GRID_ILLUMINATION_P2_IMAG' not in extnames) or ('GRID_EF_P2_REAL' not in extnames) or ('GRID_EF_P2_IMAG' not in extnames):
                        raise KeyError('One or more pieces of gridding information is missing in the input file for polarization P2. Verify the file contains appropriate data.')

                    self.gridx_P2 = hdulist['GRIDX_P2'].data
                    self.gridy_P2 = hdulist['GRIDY_P2'].data
                    self.grid_illumination_P2 = hdulist['GRID_ILLUMINATION_P2_REAL'].data + 1j * hdulist['GRID_ILLUMINATION_P2_IMAG'].data
                    self.grid_Ef_P2 = hdulist['GRID_EF_P2_REAL'].data + 1j * hdulist['GRID_EF_P2_IMAG'].data
                    self.holograph_P2 = None
                    self.img_P2 = None
                    self.holograph_PB_P2 = None
                    self.PB_P2 = None
                    self.lf_P2 = None
                    self.mf_P2 = None
                    if verbose:
                        print '\t\t\tInitialized gridx_P2, gridy_P2, grid_illumination_P2, and grid_Ef_P2'
                        print '\t\t\tInitialized lf_P2, mf_P2, holograph_PB_P2, PB_P2, holograph_P2, and img_P2'

            hdulist.close()
            if verbose:
                print '\t\tClosed input FITS file.'

        if antenna_array is not None:
            if verbose:
                print '\tInitializing from an instance of class AntennaArray...'

            if isinstance(antenna_array, AntennaArray):
                self.f = antenna_array.f
                if verbose:
                    print '\t\tInitialized frequency channels.'

                self.f0 = antenna_array.f0
                if verbose:
                    print '\t\tInitialized center frequency to {0} Hz from antenna array info.'.format(self.f0)

                self.timestamp = antenna_array.timestamp
                if verbose:
                    print '\t\tInitialized time stamp to {0} from antenna array info.'.format(self.timestamp)
            
                if (pol is None) or (pol == 'P1'):
                    if verbose:
                        print '\n\t\tWorking on polarization P1...'
                    self.gridx_P1 = antenna_array.gridx_P1
                    self.gridy_P1 = antenna_array.gridy_P1
                    self.grid_illumination_P1 = antenna_array.grid_illumination_P1
                    self.grid_Ef_P1 = antenna_array.grid_Ef_P1
                    self.holograph_P1 = None
                    self.img_P1 = None
                    self.holograph_PB_P1 = None
                    self.PB_P1 = None
                    self.lf_P1 = None
                    self.mf_P1 = None
                    if verbose:
                        print '\t\tInitialized gridx_P1, gridy_P1, grid_illumination_P1, and grid_Ef_P1.'
                        print '\t\tInitialized lf_P1, mf_P1, holograph_PB_P1, PB_P1, holograph_P1, and img_P1'

                if (pol is None) or (pol == 'P2'):
                    if verbose:
                        print '\n\t\tWorking on polarization P2...'
                    self.gridx_P2 = antenna_array.gridx_P2
                    self.gridy_P2 = antenna_array.gridy_P2
                    self.grid_illumination_P2 = antenna_array.grid_illumination_P2
                    self.grid_Ef_P2 = antenna_array.grid_Ef_P2
                    self.holograph_P2 = None
                    self.img_P2 = None
                    self.holograph_PB_P2 = None
                    self.PB_P2 = None
                    self.lf_P2 = None
                    self.mf_P2 = None
                    if verbose:
                        print '\t\tInitialized gridx_P2, gridy_P2, grid_illumination_P2, and grid_Ef_P2.'
                        print '\t\tInitialized lf_P2, mf_P2, holograph_PB_P2, PB_P2, holograph_P2, and img_P2'

            else:
                raise TypeError('antenna_array is not an instance of class AntennaArray. Cannot initiate instance of class Image.')

        if verbose:
            print '\nSuccessfully initialized an instance of class Image\n'

    #############################################################################

    def imagr(self, pol=None, verbose=True):

        """
        -------------------------------------------------------------------------
        Imaging engine that performs inverse fourier transforms of appropriate
        electric field quantities associated with the antenna array.

        Keyword Inputs:

        pol       [string] indicates which polarization information to be 
                  imaged. Allowed values are 'P1', 'P2' or None (default). If 
                  None, both polarizations are imaged.

        verbose   [boolean] If True (default), prints diagnostic and progress
                  messages. If False, suppress printing such messages.
        -------------------------------------------------------------------------
        """

        if verbose:
            print '\nPreparing to image...\n'

        if self.f is None:
            raise ValueError('Frequency channels have not been initialized. Cannot proceed with imaging.')

        if (pol is None) or (pol == 'P1'):
            
            if verbose:
                print '\tWorking on polarization P1...'

            grid_shape = self.grid_Ef_P1.shape
            if verbose:
                print '\t\tPreparing to zero pad and Inverse Fourier Transform...'

            self.holograph_P1 = NP.fft.fftshift(NP.fft.fft2(NP.pad(self.grid_Ef_P1, ((0,grid_shape[0]), (0,grid_shape[1]), (0,0)), 'constant', constant_values=(0,)), axes=(0,1)))
            if verbose:
                print '\t\tComputed complex holographic voltage image from antenna array.'

            self.holograph_PB_P1 = NP.fft.fftshift(NP.fft.fft2(NP.pad(self.grid_illumination_P1, ((0,grid_shape[0]), (0,grid_shape[1]), (0,0)), 'constant', constant_values=(0,)), axes=(0,1)))
            if verbose:
                print '\t\tComputed complex holographic voltage pattern of antenna array.'

            dx = self.gridx_P1[0,1] - self.gridx_P1[0,0]
            dy = self.gridy_P1[1,0] - self.gridy_P1[0,0]
            self.lf_P1 = NP.outer(NP.fft.fftshift(NP.fft.fftfreq(2*grid_shape[1], dx)), FCNST.c/self.f)
            self.mf_P1 = NP.outer(NP.fft.fftshift(NP.fft.fftfreq(2*grid_shape[0], dy)), FCNST.c/self.f)
            if verbose:
                print '\t\tComputed the direction cosine coordinates for the image.'
            grid_lf_P1 = NP.repeat(NP.expand_dims(self.lf_P1, axis=0), self.mf_P1.shape[0], axis=0)
            grid_mf_P1 = NP.repeat(NP.expand_dims(self.mf_P1, axis=1), self.lf_P1.shape[0], axis=1)
            nan_ind = grid_lf_P1**2 + grid_mf_P1**2 > 1.0
            self.holograph_P1[nan_ind] = NP.nan
            self.holograph_PB_P1[nan_ind] = NP.nan
            if verbose:
                print '\t\tImage pixels corresponding to invalid direction cosine coordinates flagged as NAN.'

        if (pol is None) or (pol == 'P2'):

            if verbose:
                print '\tWorking on polarization P2...'

            grid_shape = self.grid_Ef_P2.shape
            if verbose:
                print '\t\tPreparing to zero pad and Inverse Fourier Transform...'

            self.holograph_P2 = NP.fft.fftshift(NP.fft.fft2(NP.pad(self.grid_Ef_P2, ((0,grid_shape[0]), (0,grid_shape[1]), (0,0)), 'constant', constant_values=(0,)), axes=(0,1)))
            if verbose:
                print '\t\tComputed complex holographic voltage image from antenna array.'

            self.holograph_PB_P2 = NP.fft.fftshift(NP.fft.fft2(NP.pad(self.grid_illumination_P2, ((0,grid_shape[0]), (0,grid_shape[1]), (0,0)), 'constant', constant_values=(0,)), axes=(0,1)))
            if verbose:
                print '\t\tComputed complex holographic voltage pattern of antenna array.'

            dx = self.gridx_P2[0,1] - self.gridx_P2[0,0]
            dy = self.gridy_P2[1,0] - self.gridy_P2[0,0]
            self.lf_P2 = NP.outer(NP.fft.fftshift(NP.fft.fftfreq(2*grid_shape[1], dx)), FCNST.c/self.f)
            self.mf_P2 = NP.outer(NP.fft.fftshift(NP.fft.fftfreq(2*grid_shape[0], dy)), FCNST.c/self.f)
            if verbose:
                print '\t\tComputed the direction cosine coordinates for the image.'
            grid_lf_P2 = NP.repeat(NP.expand_dims(self.lf_P2, axis=0), self.mf_P2.shape[0], axis=0)
            grid_mf_P2 = NP.repeat(NP.expand_dims(self.mf_P2, axis=1), self.lf_P2.shape[0], axis=1)
            nan_ind = grid_lf_P2**2 + grid_mf_P2**2 > 1.0
            self.holograph_P2[nan_ind] = NP.nan
            self.holograph_PB_P2[nan_ind] = NP.nan
            if verbose:
                print '\t\tImage pixels corresponding to invalid direction cosine coordinates (if any) \n\t\t\thave been flagged as NAN.'
                print '\nImaging completed successfully.\n'

    #############################################################################
        
    def save(self, imgfile, pol=None, overwrite=False, verbose=True):

        """
        -------------------------------------------------------------------------
        Saves the image information to disk.

        Input:

        imgfile     [string] Image filename with full path. Will be appended 
                     with '.fits' extension

        Keyword Input(s):

        pol          [string] indicates which polarization information to be 
                     saved. Allowed values are 'P1', 'P2' or None (default). If 
                     None, information on both polarizations are saved.
                     
        overwrite    [boolean] True indicates overwrite even if a file already 
                     exists. Default = False (does not overwrite)
                     
        verbose      [boolean] If True (default), prints diagnostic and progress
                     messages. If False, suppress printing such messages.
        -------------------------------------------------------------------------
        """

        try:
            imgfile
        except NameError:
            raise NameError('No filename provided. Aborting Image.save()')

        filename = imgfile + '.fits'

        if verbose:
            print '\nSaving image information...'
            
        hdulst = []
        hdulst += [fits.PrimaryHDU()]
        hdulst[0].header['f0'] = (self.f0, 'Center frequency (Hz)')
        hdulst[0].header['tobs'] = (self.timestamp, 'Timestamp associated with observation.')
        hdulst[0].header['EXTNAME'] = 'PRIMARY'

        if verbose:
            print '\tCreated a primary HDU.'

        hdulst += [fits.ImageHDU(self.f, name='FREQ')]
        if verbose:
            print '\t\tCreated an extension HDU of {0:0d} frequency channels'.format(len(self.f))

        if (pol is None) or (pol == 'P1'):
            if verbose:
                print '\tWorking on polarization P1...'

            if self.lf_P1 is not None:
                hdulst += [fits.ImageHDU(self.lf_P1, name='grid_lf_P1')]
                if verbose:
                    print '\t\tCreated an extension HDU of l-coordinates of grid of size: {0[0]} \n\t\t\tfor each of the {0[1]} frequency channels'.format(self.lf_P1.shape)
            if self.mf_P1 is not None:
                hdulst += [fits.ImageHDU(self.mf_P1, name='grid_mf_P1')]
                if verbose:
                    print '\t\tCreated an extension HDU of m-coordinates of grid of size: {0[0]} \n\t\t\tfor each of the {0[1]} frequency channels'.format(self.mf_P1.shape)

            if self.holograph_PB_P1 is not None:
                hdulst += [fits.ImageHDU(self.holograph_PB_P1.real, name='holograph_PB_P1_real')]
                hdulst += [fits.ImageHDU(self.holograph_PB_P1.imag, name='holograph_PB_P1_imag')]
                if verbose:
                    print "\t\tCreated separate extension HDUs of grid's voltage reception pattern spectra\n\t\t\twith size {0[0]}x{0[1]}x{0[2]} for real and imaginary parts.".format(self.holograph_PB_P1.shape)
            if self.holograph_P1 is not None:
                hdulst += [fits.ImageHDU(self.holograph_P1.real, name='holograph_P1_real')]
                hdulst += [fits.ImageHDU(self.holograph_P1.imag, name='holograph_P1_imag')]
                if verbose:
                    print "\t\tCreated separate extension HDUs of grid's voltage holograph spectra of \n\t\t\tsize {0[0]}x{0[1]}x{0[2]} for real and imaginary parts.".format(self.holograph_P1.shape)

        if (pol is None) or (pol == 'P2'):
            if verbose:
                print '\tWorking on polarization P2...'

            if self.lf_P2 is not None:
                hdulst += [fits.ImageHDU(self.lf_P2, name='grid_lf_P2')]
                if verbose:
                    print '\t\tCreated an extension HDU of l-coordinates of grid of size: {0[0]} \n\t\t\tfor each of the {0[1]} frequency channels'.format(self.lf_P2.shape)
            if self.mf_P2 is not None:
                hdulst += [fits.ImageHDU(self.mf_P2, name='grid_mf_P2')]
                if verbose:
                    print '\t\tCreated an extension HDU of m-coordinates of grid of size: {0[0]} \n\t\t\tfor each of the {0[1]} frequency channels'.format(self.mf_P2.shape)

            if self.holograph_PB_P2 is not None:
                hdulst += [fits.ImageHDU(self.holograph_PB_P2.real, name='holograph_PB_P2_real')]
                hdulst += [fits.ImageHDU(self.holograph_PB_P2.imag, name='holograph_PB_P2_imag')]
                if verbose:
                    print "\t\tCreated separate extension HDUs of grid's voltage reception pattern spectra\n\t\t\twith size {0[0]}x{0[1]}x{0[2]} for real and imaginary parts.".format(self.holograph_PB_P2.shape)
            if self.holograph_P2 is not None:
                hdulst += [fits.ImageHDU(self.holograph_P2.real, name='holograph_P2_real')]
                hdulst += [fits.ImageHDU(self.holograph_P2.imag, name='holograph_P2_imag')]
                if verbose:
                    print "\t\tCreated separate extension HDUs of grid's voltage holograph spectra of \n\t\t\tsize {0[0]}x{0[1]}x{0[2]} for real and imaginary parts.".format(self.holograph_P2.shape)

        if verbose:
            print '\tNow writing FITS file to disk:\n\t\t{0}'.format(filename)

        hdu = fits.HDUList(hdulst)
        hdu.writeto(filename, clobber=overwrite)

        if verbose:
            print '\tImage information written successfully to FITS file on disk:\n\t\t{0}\n'.format(filename)

    #############################################################################
