import numpy as NP
import multiprocessing as MP
import itertools as IT
import copy
import scipy.constants as FCNST
from astropy.io import fits
import matplotlib.pyplot as PLT
import progressbar as PGB
import my_DSP_modules as DSP
import geometry as GEOM
import my_gridding_modules as GRD
import my_operations as OPS
import lookup_operations as LKP

################### Routines essential for parallel processing ################

def unwrap_antenna_FT(arg, **kwarg):
    return Antenna.FT_new(*arg, **kwarg)

def unwrap_interferometer_FX(arg, **kwarg):
    return Interferometer.FX_new(*arg, **kwarg)

def unwrap_antenna_update(arg, **kwarg):
    return Antenna.update_new(*arg, **kwarg)

def unwrap_interferometer_update(arg, **kwarg):
    return Interferometer.update_new(*arg, **kwarg)

def antenna_grid_mapping(gridind_raveled, values, bins=None):
    if bins is None:
        raise ValueError('Input parameter bins must be specified')

    if NP.iscomplexobj(values):
        retval = OPS.binned_statistic(gridind_raveled, values.real, statistic='sum', bins=bins)[0]
        retval = retval.astype(NP.complex64)
        retval += 1j * OPS.binned_statistic(gridind_raveled, values.imag, statistic='sum', bins=bins)[0]
    else:
        retval = OPS.binned_statistic(gridind_raveled, values, statistic='sum', bins=bins)[0]

    print MP.current_process().name
    return retval

def antenna_grid_mapping_arg_splitter(args, **kwargs):
    return antenna_grid_mapping(*args, **kwargs)

def antenna_grid_mapper(gridind_raveled, values, bins, label, outq):
    if NP.iscomplexobj(values):
        retval = OPS.binned_statistic(gridind_raveled, values.real, statistic='sum', bins=bins)[0]
        retval = retval.astype(NP.complex64)
        retval += 1j * OPS.binned_statistic(gridind_raveled, values.imag, statistic='sum', bins=bins)[0]
    else:
        retval = OPS.binned_statistic(gridind_raveled, values, statistic='sum', bins=bins)[0]
    outdict = {}
    outdict[label] = retval
    print MP.current_process().name
    outq.put(outdict)

def baseline_grid_mapping(gridind_raveled, values, bins=None):
    if bins is None:
        raise ValueError('Input parameter bins must be specified')

    if NP.iscomplexobj(values):
        retval = OPS.binned_statistic(gridind_raveled, values.real, statistic='sum', bins=bins)[0]
        retval = retval.astype(NP.complex64)
        retval += 1j * OPS.binned_statistic(gridind_raveled, values.imag, statistic='sum', bins=bins)[0]
    else:
        retval = OPS.binned_statistic(gridind_raveled, values, statistic='sum', bins=bins)[0]

    print MP.current_process().name
    return retval

def baseline_grid_mapping_arg_splitter(args, **kwargs):
    return baseline_grid_mapping(*args, **kwargs)

def baseline_grid_mapper(gridind_raveled, values, bins, label, outq):
    if NP.iscomplexobj(values):
        retval = OPS.binned_statistic(gridind_raveled, values.real, statistic='sum', bins=bins)[0]
        retval = retval.astype(NP.complex64)
        retval += 1j * OPS.binned_statistic(gridind_raveled, values.imag, statistic='sum', bins=bins)[0]
    else:
        retval = OPS.binned_statistic(gridind_raveled, values, statistic='sum', bins=bins)[0]
    outdict = {}
    outdict[label] = retval
    print MP.current_process().name
    outq.put(outdict)

def find_1NN_arg_splitter(args, **kwargs):
    return LKP.find_1NN(*args, **kwargs)

#################################################################################  

class PolInfo_old:

    """
    ----------------------------------------------------------------------------
    Class to manage polarization information of an antenna. 

    Attributes:

    Et_P1:   A complex numpy vector representing a time series of electric field
             for polarization P1

    Et_P2:   A complex numpy vector representing a time series of electric field
             for polarization P2 which is orthogonal to P1
        
    flag_P1: [Boolean] True means P1 is to be flagged. Default = False

    flag_P2: [Boolean] True means P2 is to be flagged. Default = False

    pol_type: [string] Type of polarization. Accepted values are 'Linear' or
              'Circular' 

    Ef_P1:   A complex numpy vector representing the Fourier transform of Et_P1

    Ef_P2:   A complex numpy vector representing the Fourier transform of Et_P2

    Member functions:

    __init__():    Initializes an instance of class PolInfo

    __str__():     Prints a summary of current attributes.

    temporal_F():  Perform a Fourier transform of an Electric field time series

    update():      Routine to update the Electric field and flag information.
    
    delay_compensation():
                   Routine to apply delay compensation to Electric field spectra 
                   through additional phase.

    Read the member function docstrings for details. 
    ----------------------------------------------------------------------------
    """

    def __init__(self, nsamples=1):
        """
        ------------------------------------------------------------------------
        Initialize the PolInfo Class which manages polarization information of
        an antenna. 

        Class attributes initialized are:
        Et_P1, Et_P2, flag_P1, flag_P2, pol_type, Ef_P1, Ef_P2
     
        Read docstring of class PolInfo for details on these attributes.
        ------------------------------------------------------------------------
        """

        if not isinstance(nsamples, int):
            raise TypeError('nsamples must be an integer')
        elif nsamples <= 0:
            nsamples = 1

        self.Et_P1 = NP.empty(nsamples, dtype=NP.complex64)
        self.Et_P2 = NP.empty(nsamples, dtype=NP.complex64)
        self.Ef_P1 = NP.empty(2 * nsamples, dtype=NP.complex64)
        self.Ef_P2 = NP.empty(2 * nsamples, dtype=NP.complex64)
        self.Et_P1.fill(NP.nan)
        self.Et_P2.fill(NP.nan)
        self.Ef_P1.fill(NP.nan)
        self.Ef_P2.fill(NP.nan)
        self.flag_P1 = True
        self.flag_P2 = True
        self.pol_type = ''

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
            self.Ef_P1 = DSP.FT1D(Et_P1, ax=0, use_real=False, inverse=False, shift=True)
            self.Ef_P2 = DSP.FT1D(Et_P2, ax=0, use_real=False, inverse=False, shift=True)
        elif pol in ['P1','p1','P2','p2','x','X','y','Y']:
            if pol in ['P1','p1','x','X']:
                Et_P1 = NP.pad(self.Et_P1, (0,len(self.Et_P1)), 'constant',
                               constant_values=(0,0))
                self.Ef_P1 = DSP.FT1D(Et_P1, ax=0, use_real=False, inverse=False, shift=True)
            else:
                Et_P2 = NP.pad(self.Et_P2, (0,len(self.Et_P2)), 'constant',
                               constant_values=(0,0))
                self.Ef_P2 = DSP.FT1D(Et_P2, ax=0, use_real=False, inverse=False, shift=True)
        else:
            raise ValueError('Polarization string unrecognized. Verify inputs. Aborting PolInfo.temporal_F()')

    ############################################################################

    def delay_compensation(self, delaydict=None):
        
        """
        -------------------------------------------------------------------------
        Routine to apply delay compensation to Electric field spectra through
        additional phase.

        Keyword input(s):

        delaydict   [dictionary] contains the following keys:
                    'pol': string specifying the polarization for which delay 
                           compensation is to be applied. Accepted values are
                           'x', 'X', 'p1', 'P1', 'y', 'Y', 'p2', and 'P2'. No
                           default.
                    'frequencies': scalar, list or numpy vector specifying the 
                           frequencie(s) (in Hz) for which delays are specified. 
                           If a scalar is specified, the delays are assumed to be
                           frequency independent and the delays are assumed to be
                           valid for all frequencies. If a vector is specified, 
                           it must be of same size as the delays and as the 
                           number of samples in the electric field timeseries. 
                           These frequencies are assumed to match those of the 
                           electric field spectrum. No default.
                    'delays': list or numpy vector specifying the delays (in 
                           seconds) at the respective frequencies which are to be 
                           compensated through additional phase in the electric 
                           field spectrum. Must be of same size as frequencies 
                           and the size of the electric field timeseries. No
                           default.
                    'fftshifted': boolean scalar indicating if the frequencies 
                           provided have already been fft-shifted. If True 
                           (default), the frequencies are assumed to have been 
                           fft-shifted. Otherwise, they have to be fft-shifted
                           before applying the delay compensation to rightly 
                           align with the fft-shifted electric field spectrum
                           computed in member function temporal_F().

        -------------------------------------------------------------------------
        """

        if delaydict is None:
            raise NameError('Delay information must be supplied for delay correction in the dictionary delaydict.')
       
        if delaydict['pol'] is None:
            raise KeyError('Key "pol" indicating polarization not found in delaydict holding delay information.')

        if delaydict['pol'] not in ['x', 'X', 'p1', 'P1', 'y', 'Y', 'p2', 'P2']:
            raise ValueError('Invalid value for "pol" keywrod in delaydict.')

        if 'delays' in delaydict:
            if NP.asarray(delaydict['delays']).size == 1:
                if delaydict['pol'] in ['x', 'X', 'p1', 'P1']:
                    delays = delaydict['delays'] + NP.zeros(self.Et_P1.size)
                else:
                    delays = delaydict['delays'] + NP.zeros(self.Et_P2.size)
            else:
                if delaydict['pol'] in ['x', 'X', 'p1', 'P1']:
                    if (NP.asarray(delaydict['delays']).size != self.Et_P1.size):
                        raise IndexError('Size of delays in delaydict must be equal to 1 or match that of the timeseries.')
                    else:
                        delays = NP.asarray(delaydict['delays']).ravel()
                else:
                    if (NP.asarray(delaydict['delays']).size != self.Et_P2.size):
                        raise IndexError('Size of delays in delaydict must be equal to 1 or match that of the timeseries.')
                    else:
                        delays = NP.asarray(delaydict['delays']).ravel()
        else:
            if delaydict['pol'] in ['x', 'X', 'p1', 'P1']:
                delays = NP.zeros(self.Et_P1.size)
            else:
                delays = NP.zeros(self.Et_P2.size)
            
        if 'frequencies' not in delaydict:
            raise KeyError('Key "frequencies" not found in dictionary delaydict holding delay information.')
        else:
            frequencies = NP.asarray(delaydict['frequencies']).ravel()

        if delaydict['pol'] in ['x', 'X', 'p1', 'P1']:
            if frequencies.size != self.Et_P1.size:
                raise IndexError('Size of frequencies must match that of the Electric field time series.')
        else:
            if frequencies.size != self.Et_P2.size:
                raise IndexError('Size of frequencies must match that of the Electric field time series.')
        
        temp_phases = 2 * NP.pi * delays * frequencies

        # Convert phases to fft-shifted arrangement based on key "fftshifted" in delaydict
        if 'fftshifted' in delaydict:
            if delaydict['fftshifted'] is not None:
                if not delaydict['fftshifted']:
                    temp_phases = NP.fft.fftshift(temp_phases)  

        # Expand the size to account for the fact that the Fourier transform of the timeseries is obtained after zero padding
        phases = NP.empty(2*frequencies.size) 
        phases[0::2] = temp_phases
        phases[1::2] = temp_phases

        if delaydict['pol'] in ['x', 'X', 'p1', 'P1']:
            self.Ef_P1 *= NP.exp(1j * phases)
        else:
            self.Ef_P2 *= NP.exp(1j * phases)

        ## INSERT FEATURE: yet to modify the timeseries after application of delay compensation ##

    ############################################################################

    def update(self, Et_P1=None, Et_P2=None, flag_P1=False, flag_P2=False,
               delaydict_P1=None, delaydict_P2=None, pol_type='Linear'):

        """
        ------------------------------------------------------------------------
        Routine to update the Electric field and flag information.

        Keyword input(s):

        Et_P1:         [Complex vector] The new electric field time series in
                       polarization P1 that will replace the current attribute

        Et_P2:         [Complex vector] The new electric field time series in 
                       polarization P2 that will replace the current attribute

        flag_P1:       [boolean] flag update for polarization P1

        flag_P2:       [boolean] flag update for polarization P2
                        
        delaydict_P1:  Dictionary containing information on delay compensation
                       to be applied to the fourier transformed electric fields
                       for polarization P1. Default is None (no delay
                       compensation to be applied). Refer to the docstring of
                       member function delay_compensation() of class PolInfo
                       for more details.

        delaydict_P2:  Dictionary containing information on delay compensation
                       to be applied to the fourier transformed electric fields
                       for polarization P2. Default is None (no delay
                       compensation to be applied). Refer to the docstring of
                       member function delay_compensation() of class PolInfo
                       for more details.

        pol_type:      'Linear' or 'Circular' polarization
        ------------------------------------------------------------------------
        """
        
        if Et_P1 is not None:
            self.Et_P1 = NP.asarray(Et_P1)
            self.temporal_F(pol='X')
              
        if Et_P2 is not None:
            self.Et_P2 = NP.asarray(Et_P2)
            self.temporal_F(pol='Y')

        if delaydict_P1 is not None:
            if 'pol' not in delaydict_P1:
                delaydict_P1['pol'] = 'P1'
            self.delay_compensation(delaydict=delaydict_P1)

        if delaydict_P2 is not None:
            if 'pol' not in delaydict_P2:
                delaydict_P2['pol'] = 'P2'
            self.delay_compensation(delaydict=delaydict_P2)

        if flag_P1 is not None: self.flag_P1 = flag_P1
        if flag_P2 is not None: self.flag_P2 = flag_P2
        if pol_type is not None: self.pol_type = pol_type

#####################################################################################

class Antenna_old:

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

    def __init__(self, label, latitude, location, center_freq, nsamples=1):
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

        self.pol = PolInfo(nsamples=nsamples)
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
               flag_P2=None, gridfunc_freq=None, delaydict_P1=None,
               delaydict_P2=None, ref_freq=None, pol_type='Linear',
               verbose=False):
        """
        -------------------------------------------------------------------------
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

        delaydict_P1: Dictionary containing information on delay compensation
                   to be applied to the fourier transformed electric fields
                   for polarization P1. Default is None (no delay
                   compensation to be applied). Refer to the docstring of
                   member function delay_compensation() of class PolInfo
                   for more details.

        delaydict_P2: Dictionary containing information on delay compensation
                   to be applied to the fourier transformed electric fields
                   for polarization P2. Default is None (no delay
                   compensation to be applied). Refer to the docstring of
                   member function delay_compensation() of class PolInfo
                   for more details.

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
                   diagnostic or progress messages.

        ------------------------------------------------------------------------
        """

        if label is not None: self.label = label
        if location is not None: self.location = location
        if timestamp is not None: self.timestamp = timestamp

        if t is not None:
            self.t = t
            self.f = self.f0 + self.channels()           

        if (flag_P1 is not None) or (flag_P2 is not None) or (Et_P1 is not None) or (Et_P2 is not None) or (delaydict_P1 is not None) or (delaydict_P2 is not None):
            self.pol.update(Et_P1=Et_P1, Et_P2=Et_P2, flag_P1=flag_P1, flag_P2=flag_P2, delaydict_P1=delaydict_P1, delaydict_P2=delaydict_P2, pol_type=pol_type)

        if wtsinfo_P1 is not None:
            self.wtspos_P1 = []
            self.wts_P1 = []
            angles = []
            if len(wtsinfo_P1) == len(self.f):
                self.wtspos_P1_scale = None
                # self.wts_P1 += [wtsinfo[1] for wtsinfo in wtsinfo_P1]
                angles += [wtsinfo['orientation'] for wtsinfo in wtsinfo_P1]
                for i in xrange(len(self.f)):
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

class AntennaArray_old:

    """
    ----------------------------------------------------------------------------
    Class to manage collective information on a group of antennas.

    Attributes:

    antennas:    [Dictionary] Dictionary consisting of keys which hold instances
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

    grid_ready_P1 
                 [boolean] True if the grid has been created for P1 polarization,
                 False otherwise

    grid_ready_P2
                 [boolean] True if the grid has been created for P2 polarization,
                 False otherwise

    gridx_P1     [Numpy array] x-locations of the grid lattice for P1
                 polarization

    gridy_P1     [Numpy array] y-locations of the grid lattice for P1
                 polarization

    gridx_P2     [Numpy array] x-locations of the grid lattice for P2
                 polarization

    gridy_P2     [Numpy array] y-locations of the grid lattice for P2
                 polarization

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
                      
    add_antennas()    Routine to add antenna(s) to the antenna array instance. 
                      A wrapper for operator overloading __add__() and __radd__()
                      
    remove_antennas() Routine to remove antenna(s) from the antenna array 
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
        self.grid_ready_P1, self.grid_ready_P2 = False, False
        self.grid_illumination_P1 = None
        self.grid_illumination_P2 = None
        self.grid_Ef_P1 = None
        self.grid_Ef_P2 = None
        self.f = None
        self.f0 = None
        self.t = None
        self.timestamp = None
        
    ################################################################################# 

    def __str__(self):
        printstr = '\n-----------------------------------------------------------------'
        printstr += '\n Instance of class "{0}" in module "{1}".\n Holds the following "Antenna" class instances with labels:\n '.format(self.__class__.__name__, self.__module__)
        printstr += '  '.join(sorted(self.antennas.keys()))
        printstr += '\n Antenna array bounds: blc = [{0[0]}, {0[1]}],\n                       trc = [{1[0]}, {1[1]}]'.format(self.ants_blc_P1.ravel(), self.ants_trc_P1.ravel())
        printstr += '\n Grid bounds: blc = [{0[0]}, {0[1]}],\n              trc = [{1[0]}, {1[1]}]'.format(self.grid_blc_P1.ravel(), self.grid_trc_P1.ravel())
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
                print "For updating, use the update() method. Ignoring antenna {0}".format(others.label)
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
        
        """
        ----------------------------------------------------------------------------
        Routine to return the antenna label and position information (sorted by
        antenna label if specified)

        Keyword Inputs:

        sort     [boolean] If True, returned antenna information is sorted by
                 antenna labels. Default = False.

        Output:

        outdict  [dictionary] Output consists of a dictionary with the following 
                 keys and information:
                 'antennas': Contains a numpy array of strings of antenna labels
                 'positions': positions of antennas

                 If input parameter sort is set to True, the antenna labels and 
                 positions are sorted by antenna labels.
        ----------------------------------------------------------------------------
        """

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

            self.grid_ready_P1 = True

        if (pol is None) or (pol == 'P2'):
            blc_P2 = [[self.antennas[label].blc_P2[0,0], self.antennas[label].blc_P2[0,1]] for label in self.antennas if not self.antennas[label].pol.flag_P2]
            trc_P2 = [[self.antennas[label].trc_P2[0,0], self.antennas[label].trc_P2[0,1]] for label in self.antennas if not self.antennas[label].pol.flag_P2]

            self.ants_blc_P2 = NP.asarray(map(min, *blc_P2))
            self.ants_trc_P2 = NP.asarray(map(max, *trc_P2))

            self.gridx_P2, self.gridy_P2 = GRD.grid_2d([(self.ants_blc_P2[0], self.ants_trc_P2[0]),(self.ants_blc_P2[1], self.ants_trc_P2[1])], pad=xypad, spacing=uvspacing*min_lambda, pow2=True)

            self.grid_blc_P2 = NP.asarray([NP.amin(self.gridx_P2[0,:]), NP.amin(self.gridy_P2[:,0])])
            self.grid_trc_P2 = NP.asarray([NP.amax(self.gridx_P2[0,:]), NP.amax(self.gridy_P2[:,0])])

            self.grid_ready_P2 = True

    #################################################################################

    def grid_convolve(self, pol=None, ants=None, unconvolve_existing=False,
                      normalize=False, method='NN', distNN=NP.inf, tol=None,
                      maxmatch=None): 

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

        maxmatch   [scalar] A positive value indicating maximum number of input 
                   locations in the antenna grid to be assigned. Default = None. 
                   If set to None, all the antenna array grid elements specified 
                   are assigned values for each antenna. For instance, to have only 
                   one antenna array grid element to be populated per antenna, use
                   maxmatch=1. 

        tol        [scalar] If set, only lookup data with abs(val) > tol will be 
                   considered for nearest neighbour lookup. Default = None implies 
                   all lookup values will be considered for nearest neighbour 
                   determination. tol is to be interpreted as a minimum value 
                   considered as significant in the lookup table. 
        ----------------------------------------------------------------------------
        """

        eps = 1.0e-10

        if (pol is None) or (pol == 'P1'):

            if not self.grid_ready_P1: # Need to create a grid since it could have changed with updates
                self.grid(pol='P1') 

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
                                        reflocs = ants[key].wtspos_P1[i] + (self.f[i]/FCNST.c) * NP.asarray([ants[key].location.x, ants[key].location.y]).reshape(1,-1)
                                        inplocs = (self.f[i]/FCNST.c) * NP.hstack((self.gridx_P1.reshape(-1,1), self.gridy_P1.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN_old(reflocs, ants[key].wts_P1[i], inplocs,
                                                                      distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        # ibind, nnval = LKP.lookup(ants[key].wtspos_P1[i][:,0] + ants[key].location.x * (self.f[i]/FCNST.c),
                                        #                           ants[key].wtspos_P1[i][:,1] + ants[key].location.y * (self.f[i]/FCNST.c),
                                        #                           ants[key].wts_P1[i], self.gridx_P1*self.f[i]/FCNST.c,
                                        #                           self.gridy_P1*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                        #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridx_P1.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif ants[key].wtspos_P1_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            reflocs = ants[key].wtspos_P1[0] + (self.f[0]/FCNST.c) * NP.asarray([ants[key].location.x, ants[key].location.y]).reshape(1,-1)
                                            inplocs = (self.f[0]/FCNST.c) * NP.hstack((self.gridx_P1.reshape(-1,1), self.gridy_P1.reshape(-1,1)))
                                            ibind, nnval = LKP.lookup_1NN_old(reflocs, ants[key].wts_P1[0], inplocs,
                                                                          distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                          remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                            # ibind, nnval = LKP.lookup(ants[key].wtspos_P1[0][:,0]+ants[key].location.x*(self.f[0]/FCNST.c),
                                            #                           ants[key].wtspos_P1[0][:,1]+ants[key].location.y*(self.f[0]/FCNST.c),
                                            #                           ants[key].wts_P1[0], self.gridx_P1*self.f[0]/FCNST.c,
                                            #                           self.gridy_P1*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                            #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
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
                                        reflocs = ants[key].wtspos_P1[i] + (self.f[i]/FCNST.c) * NP.asarray([ants[key].location.x, ants[key].location.y]).reshape(1,-1)
                                        inplocs = (self.f[i]/FCNST.c) * NP.hstack((self.gridx_P1.reshape(-1,1), self.gridy_P1.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN_old(reflocs, ants[key].wts_P1[i], inplocs,
                                                                      distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]

                                        # ibind, nnval = LKP.lookup(ants[key].wtspos_P1[i][:,0] + ants[key].location.x * (self.f[i]/FCNST.c),
                                        #                           ants[key].wtspos_P1[i][:,1] + ants[key].location.y * (self.f[i]/FCNST.c),
                                        #                           ants[key].wts_P1[i], self.gridx_P1*self.f[i]/FCNST.c,
                                        #                           self.gridy_P1*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                        #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridx_P1.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif ants[key].wtspos_P1_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            reflocs = ants[key].wtspos_P1[0] + (self.f[0]/FCNST.c) * NP.asarray([ants[key].location.x, ants[key].location.y]).reshape(1,-1)
                                            inplocs = (self.f[0]/FCNST.c) * NP.hstack((self.gridx_P1.reshape(-1,1), self.gridy_P1.reshape(-1,1)))
                                            ibind, nnval = LKP.lookup_1NN_old(reflocs, ants[key].wts_P1[0], inplocs,
                                                                          distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                          remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]

                                            # ibind, nnval = LKP.lookup(ants[key].wtspos_P1[0][:,0]+ants[key].location.x*(self.f[0]/FCNST.c),
                                            #                           ants[key].wtspos_P1[0][:,1]+ants[key].location.y*(self.f[0]/FCNST.c),
                                            #                           ants[key].wts_P1[0], self.gridx_P1*self.f[0]/FCNST.c,
                                            #                           self.gridy_P1*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                            #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
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
                                    reflocs = self.antennas[key].wtspos_P1[i] + (self.f[i]/FCNST.c) * NP.asarray([self.antennas[key].location.x, self.antennas[key].location.y]).reshape(1,-1)
                                    inplocs = (self.f[i]/FCNST.c) * NP.hstack((self.gridx_P1.reshape(-1,1), self.gridy_P1.reshape(-1,1)))
                                    ibind, nnval = LKP.lookup_1NN_old(reflocs, self.antennas[key].wts_P1[i], inplocs,
                                                                  distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                  remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]

                                    # ibind, nnval = LKP.lookup(self.antennas[key].wtspos_P1[i][:,0]+self.antennas[key].location.x*(self.f[i]/FCNST.c),
                                    #                           self.antennas[key].wtspos_P1[i][:,1]+self.antennas[key].location.y*(self.f[i]/FCNST.c),
                                    #                           self.antennas[key].wts_P1[i], self.gridx_P1*self.f[i]/FCNST.c,
                                    #                           self.gridy_P1*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                    #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                    roi_ind = NP.unravel_index(ibind, self.gridx_P1.shape)
                                    if normalize:
                                        nnval /= NP.sum(nnval)
                                elif self.antennas[key].wtspos_P1_scale == 'scale':
                                    if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                        reflocs = self.antennas[key].wtspos_P1[0] + (self.f[0]/FCNST.c) * NP.asarray([self.antennas[key].location.x, self.antennas[key].location.y]).reshape(1,-1)
                                        inplocs = (self.f[0]/FCNST.c) * NP.hstack((self.gridx_P1.reshape(-1,1), self.gridy_P1.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN_old(reflocs, self.antennas[key].wts_P1[0], inplocs,
                                                                      distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        # ibind, nnval = LKP.lookup(self.antennas[key].wtspos_P1[0][:,0]+self.antennas[key].location.x*(self.f[0]/FCNST.c),
                                        #                           self.antennas[key].wtspos_P1[0][:,1]+self.antennas[key].location.y*(self.f[0]/FCNST.c),
                                        #                           self.antennas[key].wts_P1[0], self.gridx_P1*self.f[0]/FCNST.c,
                                        #                           self.gridy_P1*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                        #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
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

            if not self.grid_ready_P2: # Need to create a grid since it could have changed with updates
                self.grid(pol='P2') 

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
                                        reflocs = ants[key].wtspos_P2[i] + (self.f[i]/FCNST.c) * NP.asarray([ants[key].location.x, ants[key].location.y]).reshape(1,-1)
                                        inplocs = (self.f[i]/FCNST.c) * NP.hstack((self.gridx_P2.reshape(-1,1), self.gridy_P2.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN_old(reflocs, ants[key].wts_P2[i], inplocs,
                                                                      distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]

                                        # ibind, nnval = LKP.lookup(ants[key].wtspos_P2[i][:,0] + ants[key].location.x * (self.f[i]/FCNST.c),
                                        #                           ants[key].wtspos_P2[i][:,1] + ants[key].location.y * (self.f[i]/FCNST.c),
                                        #                           ants[key].wts_P2[i], self.gridx_P2*self.f[i]/FCNST.c,
                                        #                           self.gridy_P2*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                        #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridx_P2.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif ants[key].wtspos_P2_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            reflocs = ants[key].wtspos_P2[0] + (self.f[0]/FCNST.c) * NP.asarray([ants[key].location.x, ants[key].location.y]).reshape(1,-1)
                                            inplocs = (self.f[0]/FCNST.c) * NP.hstack((self.gridx_P2.reshape(-1,1), self.gridy_P2.reshape(-1,1)))
                                            ibind, nnval = LKP.lookup_1NN_old(reflocs, ants[key].wts_P2[0], inplocs,
                                                                          distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                          remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                            # ibind, nnval = LKP.lookup(ants[key].wtspos_P2[0][:,0]+ants[key].location.x*(self.f[0]/FCNST.c),
                                            #                           ants[key].wtspos_P2[0][:,1]+ants[key].location.y*(self.f[0]/FCNST.c),
                                            #                           ants[key].wts_P2[0], self.gridx_P2*self.f[0]/FCNST.c,
                                            #                           self.gridy_P2*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                            #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
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
                                        reflocs = ants[key].wtspos_P2[i] + (self.f[i]/FCNST.c) * NP.asarray([ants[key].location.x, ants[key].location.y]).reshape(1,-1)
                                        inplocs = (self.f[i]/FCNST.c) * NP.hstack((self.gridx_P2.reshape(-1,1), self.gridy_P2.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN_old(reflocs, ants[key].wts_P2[i], inplocs,
                                                                      distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        # ibind, nnval = LKP.lookup(ants[key].wtspos_P2[i][:,0] + ants[key].location.x * (self.f[i]/FCNST.c),
                                        #                           ants[key].wtspos_P2[i][:,1] + ants[key].location.y * (self.f[i]/FCNST.c),
                                        #                           ants[key].wts_P2[i], self.gridx_P2*self.f[i]/FCNST.c,
                                        #                           self.gridy_P2*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                        #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridx_P2.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif ants[key].wtspos_P2_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            reflocs = ants[key].wtspos_P2[0] + (self.f[0]/FCNST.c) * NP.asarray([ants[key].location.x, ants[key].location.y]).reshape(1,-1)
                                            inplocs = (self.f[0]/FCNST.c) * NP.hstack((self.gridx_P2.reshape(-1,1), self.gridy_P2.reshape(-1,1)))
                                            ibind, nnval = LKP.lookup_1NN_old(reflocs, ants[key].wts_P2[0], inplocs,
                                                                          distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                          remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                            # ibind, nnval = LKP.lookup(ants[key].wtspos_P2[0][:,0]+ants[key].location.x*(self.f[0]/FCNST.c),
                                            #                           ants[key].wtspos_P2[0][:,1]+ants[key].location.y*(self.f[0]/FCNST.c),
                                            #                           ants[key].wts_P2[0], self.gridx_P2*self.f[0]/FCNST.c,
                                            #                           self.gridy_P2*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                            #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
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
                                    reflocs = self.antennas[key].wtspos_P2[i] + (self.f[i]/FCNST.c) * NP.asarray([self.antennas[key].location.x, self.antennas[key].location.y]).reshape(1,-1)
                                    inplocs = (self.f[i]/FCNST.c) * NP.hstack((self.gridx_P2.reshape(-1,1), self.gridy_P2.reshape(-1,1)))
                                    ibind, nnval = LKP.lookup_1NN_old(reflocs, self.antennas[key].wts_P2[i], inplocs,
                                                                  distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                  remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                    # ibind, nnval = LKP.lookup(self.antennas[key].wtspos_P2[i][:,0]+self.antennas[key].location.x*(self.f[i]/FCNST.c),
                                    #                           self.antennas[key].wtspos_P2[i][:,1]+self.antennas[key].location.y*(self.f[i]/FCNST.c),
                                    #                           self.antennas[key].wts_P2[i], self.gridx_P2*self.f[i]/FCNST.c,
                                    #                           self.gridy_P2*self.f[i]/FCNST.c, distance_ULIM=distNN*self.f[i]/FCNST.c,
                                    #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                    roi_ind = NP.unravel_index(ibind, self.gridx_P2.shape)
                                    if normalize:
                                        nnval /= NP.sum(nnval)
                                elif self.antennas[key].wtspos_P2_scale == 'scale':
                                    if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                        reflocs = self.antennas[key].wtspos_P2[0] + (self.f[0]/FCNST.c) * NP.asarray([self.antennas[key].location.x, self.antennas[key].location.y]).reshape(1,-1)
                                        inplocs = (self.f[0]/FCNST.c) * NP.hstack((self.gridx_P2.reshape(-1,1), self.gridy_P2.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN_old(reflocs, self.antennas[key].wts_P2[0], inplocs,
                                                                      distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        # ibind, nnval = LKP.lookup(self.antennas[key].wtspos_P2[0][:,0]+self.antennas[key].location.x*(self.f[0]/FCNST.c),
                                        #                           self.antennas[key].wtspos_P2[0][:,1]+self.antennas[key].location.y*(self.f[0]/FCNST.c),
                                        #                           self.antennas[key].wts_P2[0], self.gridx_P2*self.f[0]/FCNST.c,
                                        #                           self.gridy_P2*self.f[0]/FCNST.c, distance_ULIM=distNN*self.f[0]/FCNST.c,
                                        #                           remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
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
        -------------------------------------------------------------------------
        Updates the antenna array instance with newer attribute values. Can also 
        be used to add and/or remove antennas with/without affecting the existing
        grid.

        Inputs:

        updates     [Dictionary] Consists of information updates under the
                    following principal keys:
                    'antenna_array': Consists of updates for the AntennaArray
                                instance. This is a dictionary which consists of
                                the following keys:
                                'timestamp'   Unique identifier of the time 
                                              series. It is optional to set this 
                                              to a scalar. If not given, no 
                                              change is made to the existing
                                              timestamp attribute
                                'do_grid'     [boolean] If set to True, create or
                                              recreate a grid. To be specified 
                                              when the antenna locations are
                                              updated.
                    'antennas': Holds a dictionary consisting of updates for 
                                individual antennas. One of the keys is 'label' 
                                which indicates an antenna label. If absent, the 
                                code execution stops by throwing an exception. 
                                The other optional keys and the information they
                                hold are listed below:
                                'action'      [String scalar] Indicates the type 
                                              of update operation. 'add' adds the 
                                              Antenna instance to the 
                                              AntennaArray instance. 'remove' 
                                              removes the antenna from the
                                              antenna array instance. 'modify'
                                              modifies the antenna attributes in 
                                              the antenna array instance. This 
                                              key has to be set. No default.
                                'grid_action' [Boolean] If set to True, will 
                                              apply the grdding operations 
                                              (grid(), grid_convolve(), and 
                                              grid_unconvolve()) appropriately 
                                              according to the value of the 
                                              'action' key. If set to None or 
                                              False, gridding effects will remain
                                              unchanged. Default=None(=False).
                                'antenna'     [instance of class Antenna] Updated 
                                              Antenna class instance. Can work 
                                              for action key 'remove' even if not 
                                              set (=None) or set to an empty 
                                              string '' as long as 'label' key is 
                                              specified. 
                                'gridpol'     [Optional. String scalar] Initiates 
                                              the specified action on 
                                              polarization 'P1' or 'P2'. Can be 
                                              set to 'P1' or 'P2'. If not 
                                              provided (=None), then the 
                                              specified action applies to both
                                              polarizations. Default = None.
                                'Et_P1'       [Optional. Numpy array] Complex 
                                              Electric field time series in 
                                              polarization P1. Is used only if 
                                              set and if 'action' key value is 
                                              set to 'modify'. Default = None.
                                'Et_P2'       [Optional. Numpy array] Complex 
                                              Electric field time series in 
                                              polarization P2. Is used only if 
                                              set and if 'action' key value is 
                                              set to 'modify'. Default = None.
                                't'           [Optional. Numpy array] Time axis 
                                              of the time series. Is used only 
                                              if set and if 'action' key value is
                                              set to 'modify'. Default = None.
                                'timestamp'   [Optional. Scalar] Unique 
                                              identifier of the time series. Is 
                                              used only if set and if 'action' 
                                              key value is set to 'modify'.
                                              Default = None.
                                'location'    [Optional. instance of GEOM.Point
                                              class] 
                                              Antenna location in the local ENU 
                                              coordinate system. Used only if 
                                              set and if 'action' key value is 
                                              set to 'modify'. Default = None.
                                'wtsinfo_P1'  [Optional. List of dictionaries] 
                                              See description in Antenna class 
                                              member function update(). Is used 
                                              only if set and if 'action' key 
                                              value is set to 'modify'.
                                              Default = None.
                                'wtsinfo_P2'  [Optional. List of dictionaries] 
                                              See description in Antenna class 
                                              member function update(). Is used 
                                              only if set and if 'action' key 
                                              value is set to 'modify'.
                                              Default = None.
                                'flag_P1'     [Optional. Boolean] Flagging status 
                                              update for polarization P1 of the 
                                              antenna. If True, polarization P1 
                                              measurements of the antenna will be
                                              flagged. If not set (=None), the 
                                              previous or default flag status 
                                              will continue to apply. If set to 
                                              False, the antenna status will be
                                              updated to become unflagged.
                                              Default = None.
                                'flag_P2'     [Optional. Boolean] Flagging status 
                                              update for polarization P2 of the 
                                              antenna. If True, polarization P2 
                                              measurements of the antenna will be
                                              flagged. If not set (=None), the 
                                              previous or default flag status 
                                              will continue to apply. If set to 
                                              False, the antenna status will be
                                              updated to become unflagged.
                                              Default = None.
                                'gridfunc_freq'
                                              [Optional. String scalar] Read the 
                                              description of inputs to Antenna 
                                              class member function update(). If 
                                              set to None (not provided), this
                                              attribute is determined based on 
                                              the size of wtspos_P1 and wtspos_P2. 
                                              It is applicable only when 'action' 
                                              key is set to 'modify'. 
                                              Default = None.
                                'delaydict_P1'
                                              Dictionary containing information 
                                              on delay compensation to be applied 
                                              to the fourier transformed electric 
                                              fields for polarization P1. Default
                                              is None (no delay compensation to 
                                              be applied). Refer to the docstring 
                                              of member function
                                              delay_compensation() of class 
                                              PolInfo for more details.
                                'delaydict_P2'
                                              Dictionary containing information 
                                              on delay compensation to be applied 
                                              to the fourier transformed electric 
                                              fields for polarization P2. Default
                                              is None (no delay compensation to 
                                              be applied). Refer to the docstring 
                                              of member function
                                              delay_compensation() of class 
                                              PolInfo for more details.
                                'ref_freq'    [Optional. Scalar] Positive value 
                                              (in Hz) of reference frequency 
                                              (used if gridfunc_freq is set to
                                              'scale') at which wtspos_P1 and 
                                              wtspos_P2 in wtsinfo_P1 and 
                                              wtsinfo_P2, respectively, are 
                                              provided. If set to None, the 
                                              reference frequency already set in
                                              antenna array instance remains
                                              unchanged. Default = None.
                                'pol_type'    [Optional. String scalar] 'Linear' 
                                              or 'Circular'. Used only when 
                                              action key is set to 'modify'. If 
                                              not provided, then the previous
                                              value remains in effect.
                                              Default = None.
                                'norm_wts'    [Optional. Boolean] Default=False. 
                                              If set to True, the gridded weights 
                                              are divided by the sum of weights 
                                              so that the gridded weights add up 
                                              to unity. This is used only when
                                              grid_action keyword is set when
                                              action keyword is set to 'add' or
                                              'modify'
                                'gridmethod'  [Optional. String] Indicates 
                                              gridding method. It accepts the 
                                              following values 'NN' (nearest 
                                              neighbour), 'BL' (Bi-linear
                                              interpolation), and'CS' (Cubic
                                              Spline interpolation). Default='NN'
                                'distNN'      [Optional. Scalar] Indicates the 
                                              upper bound on distance for a 
                                              nearest neighbour search if the 
                                              value of 'gridmethod' is set to
                                              'NN'. The units are of physical
                                              distance, the same as what is 
                                              used for antenna locations.
                                              Default = NP.inf
                                'maxmatch'    [scalar] A positive value 
                                              indicating maximum number of input
                                              locations in the antenna grid to 
                                              be assigned. Default = None. If 
                                              set to None, all the antenna array 
                                              grid elements specified are 
                                              assigned values for each antenna.
                                              For instance, to have only one
                                              antenna array grid element to be
                                              populated per antenna, use
                                              maxmatch=1. 
                                'tol'         [scalar] If set, only lookup data 
                                              with abs(val) > tol will be
                                              considered for nearest neighbour 
                                              lookup. Default = None implies 
                                              all lookup values will be 
                                              considered for nearest neighbour
                                              determination. tol is to be
                                              interpreted as a minimum value
                                              considered as significant in the
                                              lookup table. 

        verbose     [Boolean] Default = False. If set to True, prints some 
                    diagnotic or progress messages.

        -------------------------------------------------------------------------
        """

        if updates is not None:
            if not isinstance(updates, dict):
                raise TypeError('Input parameter updates must be a dictionary')

            if 'antennas' in updates: # contains updates at level of individual antennas
                if not isinstance(updates['antennas'], list):
                    updates['antennas'] = [updates['antennas']]
                for dictitem in updates['antennas']:
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
                            if 'maxmatch' not in dictitem: dictitem['maxmatch']=None
                            if 'tol' not in dictitem: dictitem['tol']=None
                            if 'delaydict_P1' not in dictitem: dictitem['delaydict_P1']=None
                            if 'delaydict_P2' not in dictitem: dictitem['delaydict_P2']=None
                            self.antennas[dictitem['label']].update(dictitem['label'], dictitem['Et_P1'], dictitem['Et_P2'], dictitem['t'], dictitem['timestamp'], dictitem['location'], dictitem['wtsinfo_P1'], dictitem['wtsinfo_P2'], dictitem['flag_P1'], dictitem['flag_P2'], dictitem['gridfunc_freq'], dictitem['delaydict_P1'], dictitem['delaydict_P2'], dictitem['ref_freq'], dictitem['pol_type'], verbose)
                            if 'gric_action' in dictitem:
                                self.grid_convolve(pol=dictitem['gridpol'], ants=dictitem['antenna'], unconvolve_existing=True, normalize=dictitem['norm_wts'], method=dictitem['gridmethod'], distNN=dictitem['distNN'], tol=dictitem['tol'], maxmatch=dictitem['maxmatch'])
                    else:
                        raise ValueError('Update action should be set to "add", "remove" or "modify".')

            if 'antenna_array' in updates: # contains updates at 'antenna array' level
                if not isinstance(updates['antenna_array'], dict):
                    raise TypeError('Input parameter in updates for antenna array must be a dictionary with key "antenna_array"')
                
                if 'timestamp' in updates['antenna_array']:
                    self.timestamp = updates['antenna_array']['timestamp']

                if 'do_grid' in updates['antenna_array']:
                    if isinstance(updates['antenna_array']['do_grid'], boolean):
                        self.grid()
                    else:
                        raise TypeError('Value in key "do_grid" inside key "antenna_array" of input dictionary updates must be boolean.')

        self.t = self.antennas.itervalues().next().t # Update time axis
        self.f = self.antennas.itervalues().next().f # Update frequency axis

    #############################################################################

    def save(self, gridfile, pol=None, tabtype='BinTableHDU', antenna_save=True, 
             antfile=None, overwrite=False, verbose=True):

        """
        -------------------------------------------------------------------------
        Saves the antenna array information to disk. 

        Inputs:

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

################################################################################

class CrossPolInfo:

    """
    ----------------------------------------------------------------------------
    Class to manage cross polarization products of an interferometer.

    Attributes:

    Vt       [dictionary] holds cross-correlation time series under 4 cross-
             polarizations which are stored under keys 'P11', 'P12', 'P21', and
             'P22'

    Vf       [dictionary] holds cross-correlation spectra under 4 cross-
             polarizations which are stored under keys 'P11', 'P12', 'P21', and
             'P22'

    flag     [dictionary] holds boolean flags for each of the 4 cross-
             polarizations which are stored under keys 'P11', 'P12', 'P21', and
             'P22'. Default=True means it is flagged.

    Member functions:

    __init__()     Initializes an instance of class CrossPolInfo

    __str__()      Prints a summary of current attributes.

    update_flags() Updates the flags based on current inputs

    update()       Updates the visibility time series and spectra for different
                   cross-polarizations

    Read the member function docstrings for details. 
    ----------------------------------------------------------------------------
    """

    def __init__(self, nsamples=1):

        """
        ------------------------------------------------------------------------
        Initialize the CrossPolInfo Class which manages polarization information 
        of an interferometer. 

        Class attributes initialized are:
        Vt, Vf, and flags
     
        Read docstring of class PolInfo for details on these attributes.
        ------------------------------------------------------------------------
        """

        self.Vt = {}
        self.Vf = {}
        self.flag = {}

        if not isinstance(nsamples, int):
            raise TypeError('nsamples must be an integer')
        elif nsamples <= 0:
            nsamples = 1

        for pol in ['P11', 'P12', 'P21', 'P22']:
            self.Vt[pol] = NP.empty(nsamples, dtype=NP.complex64)
            self.Vf[pol] = NP.empty(nsamples, dtype=NP.complex64)
            self.Vt[pol].fill(NP.nan)
            self.Vf[pol].fill(NP.nan)
            
            self.flag[pol] = True

    ############################################################################ 

    def __str__(self):
        return ' Instance of class "{0}" in module "{1}" \n flag (P11): {2} \n flag (P12): {3} \n flag (P21): {4} \n flag (P22): {5} '.format(self.__class__.__name__, self.__module__, self.flag['P11'], self.flag['P12'], self.flag['P21'], self.flag['P22'])

    ############################################################################ 

    def update_flags(self, flags=None):

        """
        ------------------------------------------------------------------------
        Updates the flags based on current inputs
    
        Inputs:
    
        flags    [dictionary] holds boolean flags for each of the 4 cross-
                 polarizations which are stored under keys 'P11', 'P12', 'P21', 
                 and 'P22'. Default=None means no new flagging to be applied. If 
                 the value under the cross-polarization key is True, it is to be 
                 flagged and if False, it is to be unflagged.
        ------------------------------------------------------------------------
        """

        if flags is not None:
            if not isinstance(flags, dict):
                raise TypeError('Input parameter flags must be a dictionary')
            for pol in ['P11', 'P12', 'P21', 'P22']:
                if pol in flags:
                    if isinstance(flags[pol], bool):
                        self.flag[pol] = flags[pol]
                    else:
                        raise TypeError('flag values must be boolean')

    ############################################################################ 

    def update(self, Vt=None, Vf=None, flags=None):
        
        """
        ------------------------------------------------------------------------
        Updates the visibility time series and spectra for different
        cross-polarizations

        Inputs:
        
        Vt     [dictionary] holds cross-correlation time series under 4 cross-
               polarizations which are stored under keys 'P11', 'P12', 'P21', 
               and 'P22'. Default=None implies no updates for Vt.

        Vf     [dictionary] holds cross-correlation spectra under 4 cross-
               polarizations which are stored under keys 'P11', 'P12', 'P21', 
               and 'P22'. Default=None implies no updates for Vt.

        flag   [dictionary] holds boolean flags for each of the 4 cross-
               polarizations which are stored under keys 'P11', 'P12', 'P21', 
               and 'P22'. Default=None means no updates for flags.
        ------------------------------------------------------------------------
        """

        if flags is not None:
            self.update_flags(flags)

        if Vt is not None:
            if isinstance(Vt, dict):
                for pol in ['P11', 'P12', 'P21', 'P22']:
                    if pol in Vt:
                        self.Vt[pol] = Vt[pol]
                        if NP.any(NP.isnan(Vt[pol])):
                            # self.Vt[pol] = NP.nan
                            self.flag[pol] = True
            else:
                raise TypeError('Input parameter Vt must be a dictionary')

        if Vf is not None:
            if isinstance(Vf, dict):
                for pol in ['P11', 'P12', 'P21', 'P22']:
                    if pol in Vf:
                        self.Vf[pol] = Vf[pol]
                        if NP.any(NP.isnan(Vf[pol])):
                            # self.Vf[pol] = NP.nan
                            self.flag[pol] = True
            else:
                raise TypeError('Input parameter Vf must be a dictionary')

#################################################################################

class Interferometer:

    """
    ----------------------------------------------------------------------------
    Class to manage individual 2-element interferometer information.

    Attributes:

    A1          [instance of class Antenna] First antenna

    A2          [instance of class Antenna] Second antenna

    corr_type   [string] Correlator type. Accepted values are 'FX' (default) and
                'XF'

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

    f0:         [Scalar] Center frequency in Hz.

    crosspol:   [Instance of class CrossPolInfo] polarization information for the 
                interferometer. Read docstring of class CrossPolInfo for details

    wts:        [dictionary] The gridding weights for interferometer. Different 
                cross-polarizations 'P11', 'P12', 'P21' and 'P22' form the keys 
                of this dictionary. These values are in general complex. Under 
                each key, the values are maintained as a list of numpy vectors, 
                where each vector corresponds to a frequency channel. See 
                wtspos_scale for more requirements.

    wtspos      [dictionary] two-dimensional locations of the gridding weights in
                wts for each cross-polarization under keys 'P11', 'P12', 'P21',
                and 'P22'. The locations are in ENU coordinate system as a list of
                2-column numpy arrays. Each 2-column array in the list is the 
                position of the gridding weights for a corresponding frequency 
                channel. The size of the list must be the same as wts and the 
                number of channels. Units are in number of wavelengths. See 
                wtspos_scale for more requirements.

    wtspos_scale [dictionary] The scaling of weights is specified for each cross-
                 polarization under one of the keys 'P11', 'P12', 'P21' or 'P22'. 
                 The values under these keys can be either None (default) or 
                 'scale'. If None, numpy vectors in wts and wtspos under
                 corresponding keys are provided for each frequency channel. If 
                 set to 'scale' wts and wtspos contain a list of only one 
                 numpy array corresponding to a reference frequency. This is
                 scaled internally to correspond to the first channel.
                 The gridding positions are correspondingly scaled to all the 
                 frequency channels.

    blc          [2-element numpy array] Bottom Left corner where the
                 interferometer contributes non-zero weight to the grid. Same 
                 for all cross-polarizations

    trc          [2-element numpy array] Top right corner where the
                 interferometer contributes non-zero weight to the grid. Same 
                 for all cross-polarizations

    Member Functions:

    __init__():  Initializes an instance of class Interferometer

    __str__():   Prints a summary of current attributes

    channels():  Computes the frequency channels from a temporal Fourier 
                 Transform

    FX()         Computes the visibility spectrum using an FX operation, 
                 i.e., Fourier transform (F) followed by multiplication (X)
                 using antenna information in attributes A1 and A2. All four 
                 cross polarizations are computed.

    XF()         Computes the visibility spectrum using an XF operation, 
                 i.e., corss-correlation (X) followed by Fourier transform 
                 (F) using antenna information in attributes A1 and A2. All 
                 four cross polarizations are computed.

    f2t()        Computes the visibility time-series from the spectra for each 
                 cross-polarization
    
    t2f()        Computes the visibility spectra from the time-series for each 
                 cross-polarization

    flip_antenna_pair()
                 Flip the antenna pair in the interferometer. This inverts the
                 baseline vector and conjugates the visibility spectra

    update_flags()
                 Updates flags for cross-polarizations from component antenna
                 polarization flags and also overrides with flags if provided 
                 as input parameters

    update():    Updates the interferometer instance with newer attribute values
                 Updates the visibility spectrum and timeseries and applies FX
                 or XF operation.

    save():      Saves the interferometer information to disk. Needs serious 
                 development. 

    Read the member function docstrings for details.
    ----------------------------------------------------------------------------
    """

    def __init__(self, antenna1, antenna2, corr_type=None):

        """
        ------------------------------------------------------------------------
        Initialize the Interferometer Class which manages an interferometer's
        information 

        Class attributes initialized are:
        label, latitude, location, pol, t, timestamp, f0, f, wts, wtspos, 
        wtspos_scale, gridinfo, blc, trc
     
        Read docstring of class Antenna for details on these attributes.
        ------------------------------------------------------------------------
        """
        
        try:
            antenna1, antenna2
        except NameError:
            raise NameError('Two individual antenna instances must be provided.')

        if not isinstance(antenna1, Antenna):
            raise TypeError('antenna1 not an instance of class Antenna')

        if not isinstance(antenna2, Antenna):
            raise TypeError('antenna2 not an instance of class Antenna')

        self.A1 = antenna1
        self.A2 = antenna2

        if (corr_type is None) or (corr_type == 'FX'):
            self.corr_type = 'FX'
        elif corr_type == 'XF':
            self.corr_type = corr_type
        else:
            raise ValueError('Invalid correlator type')

        self.corr_type = corr_type
        
        self.latitude = 0.5 * (self.A1.latitude + self.A2.latitude) # mean latitude of two antennas
        self.location = self.A1.location - self.A2.location # Baseline vector
        if self.A1.f0 != self.A2.f0:
            raise ValueError('The center frequencies of the two antennas must be identical')
        self.f0 = self.A1.f0
        self.f = self.A1.f

        self.label = (self.A1.label, self.A2.label)

        self.t = 0.0
        self.timestamp = 0.0
        
        self.crosspol = CrossPolInfo(self.f.size)

        self.wtspos = {}
        self.wts = {}
        self.wtspos_scale = {}
        self._gridinfo = {}
        for pol in ['P11', 'P12', 'P21', 'P22']:
            self.wtspos[pol] = []
            self.wts[pol] = []
            self.wtspos_scale[pol] = None
            self._gridinfo[pol] = {}

        self.blc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1)
        self.trc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1)

    #################################################################################

    def __str__(self):
        return ' Instance of class "{0}" in module "{1}" \n label: ({2[0]}, {2[1]}) \n location: {3}'.format(self.__class__.__name__, self.__module__, self.label, self.location.__str__())

    #################################################################################

    def channels(self):

        """
        ------------------------------------------------------------------------
        Computes the frequency channels from a temporal Fourier Transform 

        Output(s):

        Frequencies corresponding to channels obtained by a Fourier Transform
        of the time series.
        ------------------------------------------------------------------------
        """

        return DSP.spectax(self.A1.t.size + self.A2.t.size, resolution=self.A1.t[1]-self.A1.t[0], shift=True)

    #############################################################################

    def FX(self):

        """
        -------------------------------------------------------------------------
        Computes the visibility spectrum using an FX operation, i.e., Fourier 
        transform (F) followed by multiplication (X). All four cross
        polarizations are computed.
        -------------------------------------------------------------------------
        """

        self.t = NP.hstack((self.A1.t.ravel(), self.A1.t.max()+self.A2.t.ravel()))
        self.f = self.f0 + self.channels()

        self.update_flags()

        self.crosspol.Vf['P11'] = self.A1.antpol.Ef['P1'] * self.A2.antpol.Ef['P1'].conjugate()
        self.crosspol.Vf['P12'] = self.A1.antpol.Ef['P1'] * self.A2.antpol.Ef['P2'].conjugate()
        self.crosspol.Vf['P21'] = self.A1.antpol.Ef['P2'] * self.A2.antpol.Ef['P1'].conjugate()
        self.crosspol.Vf['P22'] = self.A1.antpol.Ef['P2'] * self.A2.antpol.Ef['P2'].conjugate()

        self.f2t()

    #############################################################################

    def FX_new(self):

        """
        -------------------------------------------------------------------------
        Computes the visibility spectrum using an FX operation, i.e., Fourier 
        transform (F) followed by multiplication (X). All four cross
        polarizations are computed.
        -------------------------------------------------------------------------
        """

        self.t = NP.hstack((self.A1.t.ravel(), self.A1.t.max()+self.A2.t.ravel()))
        self.f = self.f0 + self.channels()

        self.update_flags()

        self.crosspol.Vf['P11'] = self.A1.antpol.Ef['P1'] * self.A2.antpol.Ef['P1'].conjugate()
        self.crosspol.Vf['P12'] = self.A1.antpol.Ef['P1'] * self.A2.antpol.Ef['P2'].conjugate()
        self.crosspol.Vf['P21'] = self.A1.antpol.Ef['P2'] * self.A2.antpol.Ef['P1'].conjugate()
        self.crosspol.Vf['P22'] = self.A1.antpol.Ef['P2'] * self.A2.antpol.Ef['P2'].conjugate()

        self.f2t()

        return self

    #############################################################################

    def XF(self):

        """
        -------------------------------------------------------------------------
        Computes the visibility spectrum using an XF operation, i.e., Correlation 
        (X) followed by Fourier transform (X). All four cross polarizations are
        computed.
        -------------------------------------------------------------------------
        """

        self.t = NP.hstack((self.A1.t.ravel(), self.A1.t.max()+self.A2.t.ravel()))
        self.f = self.f0 + self.channels()

        self.update_flags()

        self.crosspol.Vt['P11'] = DSP.XC(self.A1.antpol.Et['P1'], self.A2.antpol.Et['P1'], shift=False)
        self.crosspol.Vt['P12'] = DSP.XC(self.A1.antpol.Et['P1'], self.A2.antpol.Et['P2'], shift=False)
        self.crosspol.Vt['P21'] = DSP.XC(self.A1.antpol.Et['P2'], self.A2.antpol.Et['P1'], shift=False)
        self.crosspol.Vt['P22'] = DSP.XC(self.A1.antpol.Et['P2'], self.A2.antpol.Et['P2'], shift=False)

        self.t2f()

    #############################################################################

    def f2t(self):

        """
        -------------------------------------------------------------------------
        Computes the visibility time-series from the spectra for each cross-
        polarization
        -------------------------------------------------------------------------
        """
        
        for pol in ['P11', 'P12', 'P21', 'P22']:

            self.crosspol.Vt[pol] = DSP.FT1D(NP.fft.fftshift(self.crosspol.Vf[pol]), inverse=True, shift=True, verbose=False)

    #############################################################################

    def t2f(self):
        
        """
        -------------------------------------------------------------------------
        Computes the visibility spectra from the time-series for each cross-
        polarization
        -------------------------------------------------------------------------
        """

        for pol in ['P11', 'P12', 'P21', 'P22']:

            self.crosspol.Vf[pol] = DSP.FT1D(NP.fft.ifftshift(self.crosspol.Vt[pol]), shift=True, verbose=False)

    ############################################################################ 

    def flip_antenna_pair(self):
        
        """
        -------------------------------------------------------------------------
        Flip the antenna pair in the interferometer. This inverts the baseline
        vector and conjugates the visibility spectra
        -------------------------------------------------------------------------
        """

        self.A1, self.A2 = self.A2, self.A1 # Flip antenna instances
        self.location = -1 * self.location # Multiply baseline vector by -1
        self.blc *= -1
        self.trc *= -1

        self.crosspol.flag['P12'], self.crosspol.flag['P21'] = self.crosspol.flag['P21'], self.crosspol.flag['P12']

        self.crosspol.Vf['P11'] = self.crosspol.Vf['P11'].conjugate()
        self.crosspol.Vf['P22'] = self.crosspol.Vf['P22'].conjugate()
        self.crosspol.Vf['P12'], self.crosspol.Vf['P21'] = self.crosspol.Vf['P21'].conjugate(), self.crosspol.Vf['P12'].conjugate()

        self.f2t()
       
    ############################################################################

    def update_flags(self, flags=None):

        """
        ------------------------------------------------------------------------
        Updates flags for cross-polarizations from component antenna
        polarization flags and also overrides with flags if provided as input
        parameters

        Inputs:

        flags  [dictionary] boolean flags for each of the 4 cross-polarizations 
               of the interferometer which are stored under keys 'P11', 'P12',
               'P21', and 'P22'. Default=None means no updates for flags.
        ------------------------------------------------------------------------
        """

        for cpol in ['P11', 'P12', 'P21', 'P22']:
            self.crosspol.flag[cpol] = False

        # Flags determined from antenna level

        if self.A1.antpol.flag['P1'] or self.A2.antpol.flag['P1']:
            self.crosspol.flag['P11'] = True
        if self.A1.antpol.flag['P2'] or self.A2.antpol.flag['P1']:
            self.crosspol.flag['P21'] = True
        if self.A1.antpol.flag['P1'] or self.A2.antpol.flag['P2']:
            self.crosspol.flag['P12'] = True
        if self.A1.antpol.flag['P2'] or self.A2.antpol.flag['P2']:
            self.crosspol.flag['P22'] = True

        # Flags determined from interferometer level

        if flags is not None:
            self.crosspol.update_flags(flags)

    ############################################################################

    def update(self, label=None, Vt=None, t=None, timestamp=None,
               location=None, wtsinfo=None, flags=None, gridfunc_freq=None,
               ref_freq=None, do_correlate=None, verbose=False):

        """
        -------------------------------------------------------------------------
        Updates the interferometer instance with newer attribute values. Updates 
        the visibility spectrum and timeseries and applies FX or XF operation.

        Inputs:

        label      [Scalar] A unique identifier (preferably a string) for the 
                   antenna. Default=None means no update to apply

        latitude   [Scalar] Latitude of the antenna's location. Default=None 
                   means no update to apply

        location   [Instance of GEOM.Point class] The location of the antenna in 
                   local East, North, Up (ENU) coordinate system. Default=None 
                   means no update to apply

        timestamp  [Scalar] String or float representing the timestamp for the 
                   current attributes. Default=None means no update to apply

        t          [vector] The time axis for the visibility time series. 
                   Default=None means no update to apply

        flags      [dictionary] holds boolean flags for each of the 4 cross-
                   polarizations which are stored under keys 'P11', 'P12', 'P21', 
                   and 'P22'. Default=None means no updates for flags.

        Vt         [dictionary] holds cross-correlation time series under 4 
                   cross-polarizations which are stored under keys 'P11', 'P12', 
                   'P21', and 'P22'. Default=None implies no updates for Vt.

        wtsinfo    [dictionary] consists of weights information for each of the
                   four cross-polarizations under keys 'P11', 'P12', 'P21', and 
                   'P22'. Each of the values under the keys is a list of 
                   dictionaries. Length of list is equal to the number
                   of frequency channels or one (equivalent to setting
                   wtspos_scale to 'scale'.). The list is indexed by 
                   the frequency channel number. Each element in the list
                   consists of a dictionary corresponding to that frequency
                   channel. Each dictionary consists of these items with the
                   following keys:
                   wtspos      [2-column Numpy array, optional] u- and v- 
                               positions for the gridding weights. Units
                               are in number of wavelengths.
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

        gridfunc_freq
                   [String scalar] If set to None (not provided) or to 'scale'
                   assumes that wtspos in wtsinfo are given for a
                   reference frequency which need to be scaled for the frequency
                   channels. Will be ignored if the list of dictionaries under 
                   the cross-polarization keys in wtsinfo have number of elements 
                   equal to the number of frequency channels.

        ref_freq   [Scalar] Positive value (in Hz) of reference frequency (used
                   if gridfunc_freq is set to None or 'scale') at which
                   wtspos is provided. If set to None, ref_freq is assumed to be 
                   equal to the center frequency in the class Interferometer's 
                   attribute. 

        do_correlate
                   [string] Indicates whether correlation operation is to be
                   performed after updates. Accepted values are 'FX' (for FX
                   operation) and 'XF' (for XF operation). Default=None means
                   no correlating operation is to be performed after updates.

        verbose    [boolean] If True, prints diagnostic and progress messages. 
                   If False (default), suppress printing such messages.
        -------------------------------------------------------------------------
        """

        if label is not None: self.label = label
        if location is not None: self.location = location
        if timestamp is not None: self.timestamp = timestamp
        # if latitude is not None: self.latitude = latitude

        if t is not None:
            self.t = t
            self.f = self.f0 + self.channels()     

        if flags is not None:        # Flags determined from interferometer level
            self.update_flags(flags) 

        if Vt is not None:
            self.crosspol.update(Vt=Vt)
        
        if do_correlate is not None:
            if do_correlate == 'FX':
                self.FX()
            elif do_correlate == 'XF':
                self.XF()
            else:
                raise ValueError('Invalid specification for input parameter do_correlate.')

        blc_orig = NP.copy(self.blc)
        trc_orig = NP.copy(self.trc)
        eps = 1e-6

        if wtsinfo is not None:
            if not isinstance(wtsinfo, dict):
                raise TypeError('Input parameter wtsinfo must be a dictionary.')

            self.wtspos = {}
            self.wts = {}
            self.wtspos_scale = {}
            angles = []
            
            max_wtspos = []
            for pol in ['P11', 'P12', 'P21', 'P22']:
                self.wts[pol] = []
                self.wtspos[pol] = []
                self.wtspos_scale[pol] = None
                if pol in wtsinfo:
                    if len(wtsinfo[pol]) == len(self.f):
                        angles += [elem['orientation'] for elem in wtsinfo[pol]]
                        for i in xrange(len(self.f)):
                            rotation_matrix = NP.asarray([[NP.cos(-angles[i]),  NP.sin(-angles[i])],
                                                          [-NP.sin(-angles[i]), NP.cos(-angles[i])]])
                            if ('lookup' not in wtsinfo[pol][i]) or (wtsinfo[pol][i]['lookup'] is None):
                                self.wts[pol] += [wtsinfo[pol][i]['wts']]
                                wtspos = wtsinfo[pol][i]['wtspos']
                            else:
                                lookupdata = LKP.read_lookup(wtsinfo[pol][i]['lookup'])
                                wtspos = NP.hstack((lookupdata[0].reshape(-1,1),lookupdata[1].reshape(-1,1))) * (self.f[i]/FCNST.c)
                                self.wts[pol] += [lookupdata[2]]
                            self.wtspos[pol] += [ NP.dot(NP.asarray(wtspos), rotation_matrix.T) ]
                            max_wtspos += [NP.amax(NP.abs(self.wtspos[pol][-1]), axis=0)]
                    elif len(wtsinfo[pol]) == 1:
                        if (gridfunc_freq is None) or (gridfunc_freq == 'scale'):
                            self.wtspos_scale[pol] = 'scale'
                            if ref_freq is None:
                                ref_freq = self.f0
                            angles = wtsinfo[pol][0]['orientation']
                            rotation_matrix = NP.asarray([[NP.cos(-angles),  NP.sin(-angles)],
                                                          [-NP.sin(-angles), NP.cos(-angles)]])
                            if ('lookup' not in wtsinfo[pol][0]) or (wtsinfo[pol][0]['lookup'] is None):
                                self.wts[pol] += [ wtsinfo[pol][0]['wts'] ]
                                wtspos = wtsinfo[pol][0]['wtspos']
                            else:
                                lookupdata = LKP.read_lookup(wtsinfo[pol][0]['lookup'])
                                wtspos = NP.hstack((lookupdata[0].reshape(-1,1),lookupdata[1].reshape(-1,1))) * (ref_freq/FCNST.c)
                                self.wts[pol] += [lookupdata[2]]
                            self.wtspos[pol] += [ (self.f[0]/ref_freq) * NP.dot(NP.asarray(wtspos), rotation_matrix.T) ]     
                            max_wtspos += [NP.amax(NP.abs(self.wtspos[pol][-1]), axis=0)]
                        else:
                            raise ValueError('gridfunc_freq must be set to None, "scale" or "noscale".')
    
                        self.blc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) - FCNST.c/self.f.min() * NP.amin(NP.abs(self.wtspos[pol][0]), 0)
                        self.trc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) + FCNST.c/self.f.min() * NP.amax(NP.abs(self.wtspos[pol][0]), 0)
    
                    else:
                        raise ValueError('Number of elements in wtsinfo for {0} is incompatible with the number of channels.'.format(pol))
               
            max_wtspos = NP.amax(NP.asarray(max_wtspos).reshape(-1,blc_orig.size), axis=0)
            self.blc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) - FCNST.c/self.f.min() * max_wtspos
            self.trc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) + FCNST.c/self.f.min() * max_wtspos

        if (NP.abs(NP.linalg.norm(blc_orig)-NP.linalg.norm(self.blc)) > eps) or (NP.abs(NP.linalg.norm(trc_orig)-NP.linalg.norm(self.trc)) > eps):
            if verbose:
                print 'Grid corner(s) of interferometer {0} have changed. Should re-grid the interferometer array.'.format(self.label)

    #############################################################################

    def update_new(self, update_dict=None, verbose=True):

        """
        -------------------------------------------------------------------------
        Updates the interferometer instance with newer attribute values. Updates 
        the visibility spectrum and timeseries and applies FX or XF operation.

        Inputs:

        label      [Scalar] A unique identifier (preferably a string) for the 
                   interferometer. Default=None means no update to apply

        latitude   [Scalar] Latitude of the interferometer's location. 
                   Default=None means no update to apply

        location   [Instance of GEOM.Point class] The location of the 
                   interferometer in local East, North, Up (ENU) coordinate 
                   system. Default=None means no update to apply

        timestamp  [Scalar] String or float representing the timestamp for the 
                   current attributes. Default=None means no update to apply

        t          [vector] The time axis for the visibility time series. 
                   Default=None means no update to apply

        flags      [dictionary] holds boolean flags for each of the 4 cross-
                   polarizations which are stored under keys 'P11', 'P12', 'P21', 
                   and 'P22'. Default=None means no updates for flags.

        Vt         [dictionary] holds cross-correlation time series under 4 
                   cross-polarizations which are stored under keys 'P11', 'P12', 
                   'P21', and 'P22'. Default=None implies no updates for Vt.

        wtsinfo    [dictionary] consists of weights information for each of the
                   four cross-polarizations under keys 'P11', 'P12', 'P21', and 
                   'P22'. Each of the values under the keys is a list of 
                   dictionaries. Length of list is equal to the number
                   of frequency channels or one (equivalent to setting
                   wtspos_scale to 'scale'.). The list is indexed by 
                   the frequency channel number. Each element in the list
                   consists of a dictionary corresponding to that frequency
                   channel. Each dictionary consists of these items with the
                   following keys:
                   wtspos      [2-column Numpy array, optional] u- and v- 
                               positions for the gridding weights. Units
                               are in number of wavelengths.
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

        gridfunc_freq
                   [String scalar] If set to None (not provided) or to 'scale'
                   assumes that wtspos in wtsinfo are given for a
                   reference frequency which need to be scaled for the frequency
                   channels. Will be ignored if the list of dictionaries under 
                   the cross-polarization keys in wtsinfo have number of elements 
                   equal to the number of frequency channels.

        ref_freq   [Scalar] Positive value (in Hz) of reference frequency (used
                   if gridfunc_freq is set to None or 'scale') at which
                   wtspos is provided. If set to None, ref_freq is assumed to be 
                   equal to the center frequency in the class Interferometer's 
                   attribute. 

        do_correlate
                   [string] Indicates whether correlation operation is to be
                   performed after updates. Accepted values are 'FX' (for FX
                   operation) and 'XF' (for XF operation). Default=None means
                   no correlating operation is to be performed after updates.

        verbose    [boolean] If True, prints diagnostic and progress messages. 
                   If False (default), suppress printing such messages.
        -------------------------------------------------------------------------
        """

        label = None
        location = None
        timestamp = None
        t = None
        flags = None
        Vt = None
        do_correlate = None
        wtsinfo = None
        gridfunc_freq = None
        ref_freq = None
            
        if update_dict is not None:
            if not isinstance(update_dict, dict):
                raise TypeError('Input parameter containing updates must be a dictionary')

            if 'label' in update_dict: label = update_dict['label']
            if 'location' in update_dict: location = update_dict['location']
            if 'timestamp' in update_dict: timestamp = update_dict['timestamp']
            if 't' in update_dict: t = update_dict['t']
            if 'Vt' in update_dict: Vt = update_dict['Vt']
            if 'flags' in update_dict: flags = update_dict['flags']
            if 'do_correlate' in update_dict: do_correlate = update_dict['do_correlate']
            if 'wtsinfo' in update_dict: wtsinfo = update_dict['wtsinfo']
            if 'gridfunc_freq' in update_dict: gridfunc_freq = update_dict['gridfunc_freq']
            if 'ref_freq' in update_dict: ref_freq = update_dict['ref_freq']

        if label is not None: self.label = label
        if location is not None: self.location = location
        if timestamp is not None: self.timestamp = timestamp

        if t is not None:
            self.t = t
            self.f = self.f0 + self.channels()     

        if flags is not None:        # Flags determined from interferometer level
            self.update_flags(flags) 

        if Vt is not None:
            self.crosspol.update(Vt=Vt)
        
        if do_correlate is not None:
            if do_correlate == 'FX':
                self.FX()
            elif do_correlate == 'XF':
                self.XF()
            else:
                raise ValueError('Invalid specification for input parameter do_correlate.')

        blc_orig = NP.copy(self.blc)
        trc_orig = NP.copy(self.trc)
        eps = 1e-6

        if wtsinfo is not None:
            if not isinstance(wtsinfo, dict):
                raise TypeError('Input parameter wtsinfo must be a dictionary.')

            self.wtspos = {}
            self.wts = {}
            self.wtspos_scale = {}
            angles = []
            
            max_wtspos = []
            for pol in ['P11', 'P12', 'P21', 'P22']:
                self.wts[pol] = []
                self.wtspos[pol] = []
                self.wtspos_scale[pol] = None
                if pol in wtsinfo:
                    if len(wtsinfo[pol]) == len(self.f):
                        angles += [elem['orientation'] for elem in wtsinfo[pol]]
                        for i in xrange(len(self.f)):
                            rotation_matrix = NP.asarray([[NP.cos(-angles[i]),  NP.sin(-angles[i])],
                                                          [-NP.sin(-angles[i]), NP.cos(-angles[i])]])
                            if ('lookup' not in wtsinfo[pol][i]) or (wtsinfo[pol][i]['lookup'] is None):
                                self.wts[pol] += [wtsinfo[pol][i]['wts']]
                                wtspos = wtsinfo[pol][i]['wtspos']
                            else:
                                lookupdata = LKP.read_lookup(wtsinfo[pol][i]['lookup'])
                                wtspos = NP.hstack((lookupdata[0].reshape(-1,1),lookupdata[1].reshape(-1,1))) * (self.f[i]/FCNST.c)
                                self.wts[pol] += [lookupdata[2]]
                            self.wtspos[pol] += [ NP.dot(NP.asarray(wtspos), rotation_matrix.T) ]
                            max_wtspos += [NP.amax(NP.abs(self.wtspos[pol][-1]), axis=0)]
                    elif len(wtsinfo[pol]) == 1:
                        if (gridfunc_freq is None) or (gridfunc_freq == 'scale'):
                            self.wtspos_scale[pol] = 'scale'
                            if ref_freq is None:
                                ref_freq = self.f0
                            angles = wtsinfo[pol][0]['orientation']
                            rotation_matrix = NP.asarray([[NP.cos(-angles),  NP.sin(-angles)],
                                                          [-NP.sin(-angles), NP.cos(-angles)]])
                            if ('lookup' not in wtsinfo[pol][0]) or (wtsinfo[pol][0]['lookup'] is None):
                                self.wts[pol] += [ wtsinfo[pol][0]['wts'] ]
                                wtspos = wtsinfo[pol][0]['wtspos']
                            else:
                                lookupdata = LKP.read_lookup(wtsinfo[pol][0]['lookup'])
                                wtspos = NP.hstack((lookupdata[0].reshape(-1,1),lookupdata[1].reshape(-1,1))) * (ref_freq/FCNST.c)
                                self.wts[pol] += [lookupdata[2]]
                            self.wtspos[pol] += [ (self.f[0]/ref_freq) * NP.dot(NP.asarray(wtspos), rotation_matrix.T) ]     
                            max_wtspos += [NP.amax(NP.abs(self.wtspos[pol][-1]), axis=0)]
                        else:
                            raise ValueError('gridfunc_freq must be set to None, "scale" or "noscale".')
    
                        self.blc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) - FCNST.c/self.f.min() * NP.amin(NP.abs(self.wtspos[pol][0]), 0)
                        self.trc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) + FCNST.c/self.f.min() * NP.amax(NP.abs(self.wtspos[pol][0]), 0)
    
                    else:
                        raise ValueError('Number of elements in wtsinfo for {0} is incompatible with the number of channels.'.format(pol))
               
            max_wtspos = NP.amax(NP.asarray(max_wtspos).reshape(-1,blc_orig.size), axis=0)
            self.blc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) - FCNST.c/self.f.min() * max_wtspos
            self.trc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) + FCNST.c/self.f.min() * max_wtspos

        if (NP.abs(NP.linalg.norm(blc_orig)-NP.linalg.norm(self.blc)) > eps) or (NP.abs(NP.linalg.norm(trc_orig)-NP.linalg.norm(self.trc)) > eps):
            if verbose:
                print 'Grid corner(s) of interferometer {0} have changed. Should re-grid the interferometer array.'.format(self.label)

        return self

    #############################################################################

class InterferometerArray:

    """
    ----------------------------------------------------------------------------
    Class to manage interferometer array information.

    Attributes:

    antenna_array [instance of class AntennaArray] consists of the antenna array
                  information that determines all the interferometer pairs 

    interferometers
                  [dictionary] keys hold instances of class Interferometer. The
                  keys themselves are identical to the label attributes of the 
                  interferometer instances they hold.

    timestamp     [Scalar] String or float representing the timestamp for the 
                  current attributes


    t             [vector] The time axis for the time series of electric fields

    f             [vector] Frequency axis obtained by a Fourier Transform of
                  the electric field time series. Same length as attribute t 

    f0            [Scalar] Center frequency in Hz.

    blc           [numpy array] 2-element numpy array specifying bottom left 
                  corner of the grid coincident with bottom left interferometer 
                  location in ENU coordinate system

    trc           [numpy array] 2-element numpy array specifying top right 
                  corner of the grid coincident with top right interferometer 
                  location in ENU coordinate system

    grid_blc      [numpy array] 2-element numpy array specifying bottom left 
                  corner of the grid in ENU coordinate system including any 
                  padding used

    grid_trc      [numpy array] 2-element numpy array specifying top right 
                  corner of the grid in ENU coordinate system including any 
                  padding used

    gridx         [numpy array] two-dimensional numpy meshgrid array specifying
                  grid x-locations in units of physical distance (in metres) in 
                  the ENU coordinate system whose corners are specified by 
                  attributes grid_blc and grid_trc

    gridy         [numpy array] two-dimensional numpy meshgrid array specifying
                  grid y-locations in units of physical distance (in metres) in 
                  the ENU coordinate system whose corners are specified by 
                  attributes grid_blc and grid_trc

    grid_ready    [boolean] set to True if the gridding has been performed,
                  False if grid is not available yet. Set to False in case 
                  blc, trc, grid_blc or grid_trc is updated indicating gridding
                  is to be perfomed again

    grid_illumination
                  [dictionary] Gridded illumination cube for each 
                  cross-polarization is under one of the four keys 'P11', 'P12', 
                  'P21' or 'P22'. Under each of these keys the grid illumination 
                  is a three-dimensional complex numpy array of shape 
                  n_u x n_v x nchan, where, n_u, n_v and nchan are the grid size 
                  along u-axis, v-axis and frequency axis respectively.

    grid_Vf       [dictionary] Gridded visibility cube for each
                  cross-polarization is under one of the four keys 'P11', 'P12', 
                  'P21' or 'P22'. Under each of these keys the grid illumination 
                  is a three-dimensional complex numpy array of shape 
                  n_u x n_v x nchan, where, n_u, n_v and nchan are the grid size 
                  along u-axis, v-axis and frequency axis respectively.

    ordered_labels
                  [list] list of interferometer labels sorted by the first 
                  antenna label

    grid_mapper   [dictionary] baseline-to-grid mapping information for each of
                  four cross-polarizations under keys 'P11', 'P12', 'P21', and
                  'P22'. Under each cross-polarization, it is a dictionary with
                  values under the following keys:
                  'refind'    [list] each element in the list corresponds to a
                              sequential frequency channel and is another list 
                              with indices to the lookup locations that map to
                              the grid locations (indices in 'gridind') for this 
                              frequency channel. These indices index the array 
                              in 'refwts'
                  'gridind'   [list] each element in the list corresponds to a
                              sequential frequency channel and is another list 
                              with indices to the grid locations that map to
                              the lookup locations (indices in 'refind') for 
                              this frequency channel.
                  'refwts'    [numpy array] interferometer weights of size 
                              n_bl x n_wts flattened to be a vector. Indices in
                              'refind' index to this array. Currently only valid
                              when lookup weights scale with frequency.
                  'labels'    [dictionary] contains mapping information from 
                              interferometer (specified by key which is the 
                              interferometer label). The value under each label 
                              key is another dictionary with the following keys 
                              and information:
                              'flag'         [boolean] if True, this cross-
                                             polarization for this interferometer
                                             will not be mapped to the grid
                              'gridind'      [numpy vector] one-dimensional index 
                                             into the three-dimensional grid 
                                             locations where the interferometer
                                             contributes illumination and 
                                             visibilities. The one-dimensional 
                                             indices are obtained using numpy's
                                             multi_ravel_index() using the grid 
                                             shape, n_u x n_v x nchan
                              'illumination' [numpy vector] complex grid 
                                             illumination contributed by the 
                                             interferometer to different grid
                                             locations in 'gridind'. It is 
                                             mapped to the
                                             grid as specified by indices in 
                                             key 'gridind'
                              'Vf'           [numpy vector] complex grid 
                                             visibilities contributed by the 
                                             interferometer. It is mapped to the
                                             grid as specified by indices in 
                                             key 'gridind'
                  'bl'        [dictionary] dictionary with information on 
                              contribution of all baseline lookup weights. This
                              contains another dictionary with the following 
                              keys:
                              'ind_freq'     [list] each element in the list is
                                             for a frequency channel and 
                                             consists of a numpy vector which 
                                             consists of indices of the 
                                             contributing interferometers
                              'ind_all'      [numpy vector] consists of numpy 
                                             vector which consists of indices 
                                             of the contributing interferometers
                                             for all frequencies appended 
                                             together. Effectively, this is just
                                             values in 'ind_freq' of all 
                                             frequencies appended together.
                              'uniq_ind_all' [numpy vector] consists of numpy
                                             vector which consists of unique 
                                             indices of contributing baselines
                                             for all frequencies.
                              'rev_ind_all'  [numpy vector] reverse indices of 
                                             'ind_all' with reference to bins of
                                             'uniq_ind_all'
                              'illumination' [numpy vector] complex grid
                                             illumination weights contributed by
                                             each baseline (including associated
                                             kernel weight locations) and has a
                                             size equal to that in 'ind_all'
                  'grid'      [dictionary] contains information about populated
                              portions of the grid. It consists of values in the
                              following keys:
                              'ind_all'      [numpy vector] indices of all grid
                                             locations raveled to one dimension
                                             from three dimensions of size 
                                             n_u x n_v x nchan
                                             
    Member Functions:

    __init__()      Initializes an instance of class InterferometerArray

    __str__()       Prints summary of an instance of this class

    __add__()       Operator overloading for adding interferometer(s)

    __radd__()      Operator overloading for adding interferometer(s)

    __sub__()       Operator overloading for removing interferometer(s)

    add_interferometers()
                    Routine to add interferometer(s) to the interferometer 
                    array instance. A wrapper for operator overloading 
                    __add__() and __radd__()

    remove_interferometers()
                    Routine to remove interferometer(s) from the interferometer 
                    array instance. A wrapper for operator overloading __sub__()

    interferometers_containing_antenna()
                    Find interferometer pairs which contain the specified 
                    antenna labels

    baseline_vectors()
                    Routine to return the interferometer label and baseline 
                    vectors (sorted by interferometer label if specified)

    FX()            Computes the Fourier transform of the cross-correlated time 
                    series of the interferometer pairs in the interferometer 
                    array to compute the visibility spectra

    XF()            Computes the visibility spectra by cross-multiplying the 
                    electric field spectra for all the interferometer pairs in 
                    the interferometer array

    get_visibilities()
                    Routine to return the interferometer label and visibilities 
                    (sorted by interferometer label if specified)

    grid()          Routine to produce a grid based on the interferometer array 

    grid_convolve() Routine to project the complex illumination power pattern 
                    and the visibilities on the grid. It can operate on the 
                    entire interferometer array or incrementally project the 
                    visibilities and complex illumination power patterns from
                    specific interferometers on to an already existing grid. 
                    (The latter is not implemented yet)

    grid_convolve_old()
                    Routine to project the visibility illumination pattern and 
                    the visibilities on the grid. It can operate on the entire 
                    antenna array or incrementally project the visibilities and 
                    illumination patterns from specific antenna pairs on to an
                    already existing grid.

    make_grid_cube()
                    Constructs the grid of complex power illumination and 
                    visibilities using the gridding information determined for 
                    every baseline. Flags are taken into account while 
                    constructing this grid.

    grid_unconvolve()
                    [Needs to be re-written] Routine to de-project the 
                    visibility illumination pattern and the visibilities on the 
                    grid. It can operate on the entire interferometer array or 
                    incrementally de-project the visibilities and illumination 
                    patterns of specific antenna pairs from an already existing 
                    grid.

    update_flags()  Updates all flags in the interferometer array followed by 
                    any flags that need overriding through inputs of specific 
                    flag information

    update()        Updates the interferometer array instance with newer 
                    attribute values. Can also be used to add and/or remove 
                    interferometers with/without affecting the existing grid.
    ----------------------------------------------------------------------------
    """

    def __init__(self, antenna_pairs=None, antenna_array=None):

        """
        ------------------------------------------------------------------------
        Initializes an instance of class InterferometerArray

        Class attributes initialized are:
        antenna_array, interferometers, timestamp, t, f, f0, blc, trc, grid_blc,
        grid_trc, gridx, gridy, grid_ready, grid_illumination, grid_Vf, 
        ordered_labels, grid_mapper
        ------------------------------------------------------------------------
        """

        self.antenna_array = AntennaArray()
        self.interferometers = {}
        self.blc = NP.zeros(2)
        self.trc = NP.zeros(2)
        self.grid_blc = NP.zeros(2)
        self.grid_trc = NP.zeros(2)
        self.gridx, self.gridy = None, None
        self.gridu, self.gridv = None, None
        self.grid_ready = False
        self.grid_illumination = {}
        self.grid_Vf = {}

        self._bl_contribution = {}

        self.ordered_labels = [] # Usually output from member function baseline_vectors() or get_visibilities()
        self.grid_mapper = {}

        for pol in ['P11', 'P12', 'P21', 'P22']:
            self.grid_mapper[pol] = {}
            self.grid_mapper[pol]['labels'] = {}
            self.grid_mapper[pol]['refind'] = []
            # self.grid_mapper[pol]['bl_ind'] = []
            self.grid_mapper[pol]['gridind'] = []
            self.grid_mapper[pol]['refwts'] = None
            self.grid_mapper[pol]['bl'] = {}
            self.grid_mapper[pol]['bl']['ind_freq'] = []
            self.grid_mapper[pol]['bl']['ind_all'] = None
            self.grid_mapper[pol]['bl']['uniq_ind_all'] = None
            self.grid_mapper[pol]['bl']['rev_ind_all'] = None
            self.grid_mapper[pol]['bl']['illumination'] = None
            self.grid_mapper[pol]['grid'] = {}
            self.grid_mapper[pol]['grid']['ind_all'] = None

            self.grid_illumination[pol] = None
            self.grid_Vf[pol] = None
            self._bl_contribution[pol] = {}

        if (antenna_array is not None) and (antenna_pairs is not None):
            raise ValueError('InterferometerArray instance cannot be initialized with both inputs antenna_array and antenna_pairs.')

        if antenna_array is not None:
            if isinstance(antenna_array, AntennaArray):
                self.antenna_array = antenna_array
            else: # if antenna_array is just a list of antennas (Check this piece of code again)
                self.antenna_array = self.antenna_array + antenna_array

            ant_labels = self.antenna_array.antennas.keys()
            for i in xrange(len(ant_labels)-1):
                for j in xrange(i+1,len(ant_labels)):
                    ant_pair = Interferometer(self.antenna_array.antennas[ant_labels[i]], self.antenna_array.antennas[ant_labels[j]])
                    self.interferometers[ant_pair.label] = ant_pair

        if antenna_pairs is not None:
            if isinstance(antenna_pairs, Interferometer):
                self.interferometers[antenna_pairs.label] = antenna_pairs
            elif isinstance(antenna_pairs, dict):
                for key,value in antenna_pairs.items():
                    if isinstance(key, tuple):
                        if len(key) == 2:
                            if isinstance(value, Interferometer):
                                self.interferometers[key] = value
                            else:
                                print 'An item found not to be an instance of class Interferometer. Discarding and proceeding ahead.'
                        else:
                            print 'Invalid interferometer label found. Discarding and proceeding ahead.'
                    else:
                        print 'Invalid interferometer label found. Discarding and proceeding ahead.'
            elif isinstance(antenna_pairs, list):
                for value in antenna_pairs:
                    if isinstance(value, Interferometer):
                        self.interferometers[value.label] = value
                    else:
                        print 'An item found not to be an instance of class Interferometer. Discarding and proceeding ahead.'
            else:
                raise TypeError('Input parameter antenna_pairs found to be of compatible type, namely, instance of class Interferometer, list of instances of class Interferometer or dictionary of interferometers.')

            for label, interferometer in self.interferometers.items():
                if label[0] not in self.antenna_array.antennas:
                    self.antenna_array = self.antenna_array + interferometer.A1
                    # self.antenna_array.add_antennas(interferometer.A1)
                if label[1] not in self.antenna_array.antennas:
                    self.antenna_array = self.antenna_array + interferometer.A2
                    # self.antenna_array.add_antennas(interferometer.A2)

        self.f = self.antenna_array.f
        self.f0 = self.antenna_array.f0
        self.t = None
        self.timestamp = self.antenna_array.timestamp

    ################################################################################# 

    def __str__(self):
        printstr = '\n-----------------------------------------------------------------'
        printstr += '\n Instance of class "{0}" in module "{1}".\n Holds the following "Interferometer" class instances with labels:\n '.format(self.__class__.__name__, self.__module__)
        printstr += str(self.interferometers.keys()).strip('[]')
        # printstr += '  '.join(sorted(self.interferometers.keys()))
        printstr += '\n Interferometer array bounds: blc = [{0[0]}, {0[1]}],\n\ttrc = [{1[0]}, {1[1]}]'.format(self.blc, self.trc)
        printstr += '\n Grid bounds: blc = [{0[0]}, {0[1]}],\n\ttrc = [{1[0]}, {1[1]}]'.format(self.grid_blc, self.grid_trc)
        printstr += '\n-----------------------------------------------------------------'
        return printstr

    ################################################################################# 

    def __add__(self, others):

        """
        ----------------------------------------------------------------------------
        Operator overloading for adding interferometer(s)
    
        Inputs:
    
        others     [Instance of class InterferometerArray, dictionary holding
                   instance(s) of class Interferometer, list of instances of class 
                   Interferometer, or a single instance of class Interferometer] If 
                   a dictionary is provided, the keys should be the antenna labels 
                   and the values should be instances  of class Interferometer. If 
                   a list is provided, it should be a list of valid instances of 
                   class Interferometer. These instance(s) of class Interferometer 
                   will be added to the existing instance of InterferometerArray
                   class.
        ----------------------------------------------------------------------------
        """

        retval = self
        if isinstance(others, InterferometerArray):
            # for k,v in others.interferometers.items():
            for k,v in others.interferometers.iteritems():
                if k in retval.interferometers:
                    print "Interferometer {0} already included in the list of interferometers.".format(k)
                    print "For updating, use the update() method. Ignoring interferometer {0}".format(k)
                else:
                    retval.interferometers[k] = v
                    print 'Interferometer "{0}" added to the list of interferometers.'.format(k)
        elif isinstance(others, dict):
            # for item in others.values():
            for item in others.itervalues():
                if isinstance(item, Interferometer):
                    if item.label in retval.interferometers:
                        print "Interferometer {0} already included in the list of interferometers.".format(item.label)
                        print "For updating, use the update() method. Ignoring interferometer {0}".format(item.label)
                    else:
                        retval.interferometers[item.label] = item
                        print 'Interferometer "{0}" added to the list of interferometers.'.format(item.label)
        elif isinstance(others, list):
            for i in range(len(others)):
                if isinstance(others[i], Interferometer):
                    if others[i].label in retval.interferometers:
                        print "Interferometer {0} already included in the list of interferometers.".format(others[i].label)
                        print "For updating, use the update() method. Ignoring interferometer {0}".format(others[i].label)
                    else:
                        retval.interferometers[others[i].label] = others[i]
                        print 'Interferometer "{0}" added to the list of interferometers.'.format(others[i].label)
                else:
                    print 'Element \# {0} is not an instance of class Interferometer.'.format(i)
        elif isinstance(others, Interferometer):
            if others.label in retval.interferometers:
                print "Interferometer {0} already included in the list of interferometers.".format(others.label)
                print "For updating, use the update() method. Ignoring interferometer {0}".format(others[i].label)
            else:
                retval.interferometers[others.label] = others
                print 'Interferometer "{0}" added to the list of interferometers.'.format(others.label)
        else:
            print 'Input(s) is/are not instance(s) of class Interferometer.'

        return retval

    ################################################################################# 

    def __radd__(self, others):

        """
        ----------------------------------------------------------------------------
        Operator overloading for adding interferometer(s)
    
        Inputs:
    
        others     [Instance of class InterferometerArray, dictionary holding 
                   instance(s) of class Interferometer, list of instances of class 
                   Interferometer, or a single instance of class Interferometer] 
                   If a dictionary is provided, the keys should be the 
                   interferometer labels and the values should be instances of 
                   class Interferometer. If a list is provided, it should be a list 
                   of valid instances of class Interferometer. These instance(s) 
                   of class Interferometer will be added to the existing instance 
                   of InterferometerArray class.
        ----------------------------------------------------------------------------
        """

        return self.__add__(others)

    ################################################################################# 

    def __sub__(self, others):

        """
        ----------------------------------------------------------------------------
        Operator overloading for removing interferometer(s)
    
        Inputs:
    
        others     [Instance of class InterferometerArray, dictionary holding 
                   instance(s) of class Interferometer, list of instances of class 
                   Interferometer, list of strings containing interferometer labels 
                   or a single instance of class Interferometer] If a dictionary is 
                   provided, the keys should be the interferometer labels and the 
                   values should be instances of class Interferometer. If a list 
                   is provided, it should be a list of valid instances of class 
                   Interferometer. These instance(s) of class Interferometer will 
                   be removed from the existing instance of InterferometerArray 
                   class.
        ----------------------------------------------------------------------------
        """

        retval = self
        if isinstance(others, dict):
            for item in others.values():
                if isinstance(item, Interferometer):
                    if item.label not in retval.interferometers:
                        print "Interferometer {0} does not exist in the list of interferometers.".format(item.label)
                    else:
                        del retval.interferometers[item.label]
                        print 'Interferometer "{0}" removed from the list of interferometers.'.format(item.label)
        elif isinstance(others, list):
            for i in range(0,len(others)):
                if isinstance(others[i], str):
                    if others[i] in retval.interferometers:
                        del retval.interferometers[others[i]]
                        print 'Interferometer {0} removed from the list of interferometers.'.format(others[i])
                elif isinstance(others[i], Interferometer):
                    if others[i].label in retval.interferometers:
                        del retval.interferometers[others[i].label]
                        print 'Interferometer {0} removed from the list of interferometers.'.format(others[i].label)
                    else:
                        print "Interferometer {0} does not exist in the list of interferometers.".format(others[i].label)
                else:
                    print 'Element \# {0} has no matches in the list of interferometers.'.format(i)                        
        elif others in retval.interferometers:
            del retval.interferometers[others]
            print 'Interferometer "{0}" removed from the list of interferometers.'.format(others)
        elif isinstance(others, Interferometer):
            if others.label in retval.interferometers:
                del retval.interferometers[others.label]
                print 'Interferometer "{0}" removed from the list of interferometers.'.format(others.label)
            else:
                print "Interferometer {0} does not exist in the list of interferometers.".format(others.label)
        else:
            print 'No matches found in existing list of interferometers.'

        return retval

    ################################################################################# 

    def add_interferometers(self, A=None):

        """
        ----------------------------------------------------------------------------
        Routine to add interferometer(s) to the interferometer array instance. 
        A wrapper for operator overloading __add__() and __radd__()
    
        Inputs:
    
        A          [Instance of class InterferometerArray, dictionary holding 
                   instance(s) of class Interferometer, list of instances of class 
                   Interferometer, or a single instance of class Interferometer] If 
                   a dictionary is provided, the keys should be the interferometer 
                   labels and the values should be instances of class 
                   Interferometer. If a list is provided, it should be a list of 
                   valid instances of class Interferometer. These instance(s) of 
                   class Interferometer will be added to the existing instance of 
                   InterferometerArray class.
        ----------------------------------------------------------------------------
        """

        if A is None:
            print 'No interferometer(s) supplied.'
        elif isinstance(A, (list, Interferometer)):
            self = self.__add__(A)
        else:
            print 'Input(s) is/are not instance(s) of class Interferometer.'

    ################################################################################# 

    def remove_interferometers(self, A=None):

        """
        ----------------------------------------------------------------------------
        Routine to remove interferometer(s) from the interferometer array instance. 
        A wrapper for operator overloading __sub__()
    
        Inputs:
    
        A          [Instance of class InterferometerArray, dictionary holding 
                   instance(s) of class Interferometer, list of instances of class 
                   Interferometer, or a single instance of class Interferometer] If 
                   a dictionary is provided, the keys should be the interferometer 
                   labels and the values should be instances of class 
                   Interferometer. If a list is provided, it should be a list of 
                   valid instances of class Interferometer. These instance(s) of 
                   class Interferometer will be removed from the existing instance 
                   of InterferometerArray class.
        ----------------------------------------------------------------------------
        """

        if A is None:
            print 'No interferometer specified for removal.'
        else:
            self = self.__sub__(A)

    ################################################################################# 

    def interferometers_containing_antenna(self, antenna_label):

        """
        ----------------------------------------------------------------------------
        Find interferometer pairs which contain the specified antenna labels

        Inputs:

        antenna_label [list] List of antenna labels which will be searched for in
                      the interferometer pairs in the interferometer array.

        Outputs:

        ant_pair_labels
                      [list] List of interferometer pair labels containing one of
                      more of the specified antenna labels

        ant_order     [list] List of antenna order of antenna labels found in the 
                      interferometer pairs of the interferometer array. If the 
                      antenna label appears as the first antenna in the antenna 
                      pair, ant_order is assigned to 1 and if it is the second 
                      antenna in the pair, it is assigned to 2.
        ----------------------------------------------------------------------------
        """

        ant_pair_labels = [ant_pair_label for ant_pair_label in self.interferometers if antenna_label in ant_pair_label]
        ant_order = [1 if ant_pair_label[0] == antenna_label else 2 for ant_pair_label in ant_pair_labels]

        return (ant_pair_labels, ant_order)

    ################################################################################# 

    def baseline_vectors(self, pol=None, flag=False, sort=True):
        
        """
        ----------------------------------------------------------------------------
        Routine to return the interferometer label and baseline vectors (sorted by
        interferometer label if specified)

        Keyword Inputs:

        pol      [string] select baselines of this polarization that are either 
                 flagged or unflagged as specified by input parameter flag. 
                 Allowed values are 'P11', 'P12', 'P21', and 'P22'. Default=None. 
                 This means all baselines are returned irrespective of the flags

        flag     [boolean] If False, return unflagged baselines, otherwise return
                 flagged ones. Default=None means return all baselines
                 independent of flagging or polarization

        sort     [boolean] If True, returned interferometer information is sorted 
                 by interferometer's first antenna label. Default = True.

        Output:

        outdict  [dictionary] Output consists of a dictionary with the following 
                 keys and information:
                 'labels':    Contains a numpy array of strings of interferometer 
                              labels
                 'baselines': baseline vectors of interferometers (3-column 
                              array)
        ----------------------------------------------------------------------------
        """

        if not isinstance(sort, bool):
            raise TypeError('sort keyword has to be a Boolean value.')

        if flag is not None:
            if not isinstance(flag, bool):
                raise TypeError('flag keyword has to be a Boolean value.')

        if pol is None:
            if sort: # sort by first antenna label
                xyz = NP.asarray([[self.interferometers[label].location.x, self.interferometers[label].location.y, self.interferometers[label].location.z] for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0])])
                labels = sorted(self.interferometers.keys(), key=lambda tup: tup[0])
            else:
                xyz = NP.asarray([[self.interferometers[label].location.x, self.interferometers[label].location.y, self.interferometers[label].location.z] for label in self.interferometers.keys()])
                labels = self.interferometers.keys()
        else:
            if not isinstance(pol, str):
                raise TypeError('Input parameter must be a string')
            
            if not pol in ['P11', 'P12', 'P21', 'P22']:
                raise ValueError('Invalid specification for input parameter pol')

            if sort:                   # sort by first antenna label
                if flag is None:       # get all baselines
                    xyz = NP.asarray([[self.interferometers[label].location.x, self.interferometers[label].location.y, self.interferometers[label].location.z] for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0])])
                    labels = [label for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0])]
                else:
                    if flag:           # get flagged baselines
                        xyz = NP.asarray([[self.interferometers[label].location.x, self.interferometers[label].location.y, self.interferometers[label].location.z] for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0]) if self.interferometers[label].crosspol.flag[pol]])
                        labels = [label for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0]) if self.interferometers[label].crosspol.flag[pol]]                    
                    else:              # get unflagged baselines
                        xyz = NP.asarray([[self.interferometers[label].location.x, self.interferometers[label].location.y, self.interferometers[label].location.z] for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0]) if not self.interferometers[label].crosspol.flag[pol]])
                        labels = [label for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0]) if not self.interferometers[label].crosspol.flag[pol]]
            else:                      # no sorting
                if flag is None:       # get all baselines
                    xyz = NP.asarray([[self.interferometers[label].location.x, self.interferometers[label].location.y, self.interferometers[label].location.z] for label in self.interferometers.keys()])
                    labels = [label for label in self.interferometers.keys()]
                else:
                    if flag:           # get flagged baselines
                        xyz = NP.asarray([[self.interferometers[label].location.x, self.interferometers[label].location.y, self.interferometers[label].location.z] for label in self.interferometers.keys() if self.interferometers[label].crosspol.flag[pol]])
                        labels = [label for label in self.interferometers.keys() if self.interferometers[label].crosspol.flag[pol]]
                    else:              # get unflagged baselines
                        xyz = NP.asarray([[self.interferometers[label].location.x, self.interferometers[label].location.y, self.interferometers[label].location.z] for label in self.interferometers.keys() if not self.interferometers[label].crosspol.flag[pol]])
                        labels = [label for label in self.interferometers.keys() if not self.interferometers[label].crosspol.flag[pol]]

        outdict = {}
        outdict['labels'] = labels
        outdict['baselines'] = xyz

        return outdict

    ################################################################################# 

    def FX(self, parallel=False, nproc=None):

        """
        ----------------------------------------------------------------------------
        Computes the Fourier transform of the cross-correlated time series of the
        interferometer pairs in the interferometer array to compute the visibility
        spectra
        ----------------------------------------------------------------------------
        """
        
        if self.t is None:
            self.t = self.interferometers.itervalues().next().t

        if self.f is None:
            self.f = self.interferometers.itervalues().next().f

        if self.f0 is None:
            self.f0 = self.interferometers.itervalues().next().f0

        # for label in self.interferometers: # Start processes in parallel
        #     self.interferometers[label].start()

        if not parallel:
            for label in self.interferometers:
                self.interferometers[label].FX()
        elif parallel or (nproc is not None):
            if nproc is None:
                nproc = max(MP.cpu_count()-1, 1) 
            else:
                nproc = min(nproc, max(MP.cpu_count()-1, 1))
            pool = MP.Pool(processes=nproc)
            updated_interferometers = pool.map(unwrap_interferometer_FX, IT.izip(self.interferometers.values()))
            pool.close()
            pool.join()

            for interferometer in updated_interferometers: 
                self.interferometers[interferometer.label] = interferometer
            del updated_interferometers

    ################################################################################# 

    def XF(self):

        """
        ----------------------------------------------------------------------------
        Computes the visibility spectra by cross-multiplying the electric field
        spectra for all the interferometer pairs in the interferometer array
        ----------------------------------------------------------------------------
        """
        
        if self.t is None:
            self.t = self.interferometers.itervalues().next().t

        if self.f is None:
            self.f = self.interferometers.itervalues().next().f

        if self.f0 is None:
            self.f0 = self.interferometers.itervalues().next().f0

        for label in self.interferometers:
            self.interferometers[label].XF()

        
    ################################################################################# 

    def get_visibilities(self, pol, flag=False, sort=True):

        """
        ----------------------------------------------------------------------------
        Routine to return the interferometer label and visibilities (sorted by
        interferometer label if specified)

        Keyword Inputs:

        pol      [string] select baselines of this polarization that are either 
                 flagged or unflagged as specified by input parameter flag. 
                 Allowed values are 'P11', 'P12', 'P21', and 'P22'. Only one of
                 these values must be specified.

        flag     [boolean] If False, return visibilities of unflagged baselines,
                 otherwise return flagged ones. Default=None means all visibilities
                 independent of flagging are returned.

        sort     [boolean] If True, returned interferometer information is sorted 
                 by interferometer's first antenna label. Default = True.

        Output:

        outdict  [dictionary] Output consists of a dictionary with the following 
                 keys and information:
                 'labels':    Contains a numpy array of strings of interferometer 
                              labels
                 'visibilities': interferometer visibilities (n_bl x nchan array)
        ----------------------------------------------------------------------------
        """

        try: 
            pol 
        except NameError:
            raise NameError('Input parameter pol must be specified.')

        if not isinstance(pol, str):
            raise TypeError('Input parameter must be a string')
        
        if not pol in ['P11', 'P12', 'P21', 'P22']:
            raise ValueError('Invalid specification for input parameter pol')

        if not isinstance(sort, bool):
            raise TypeError('sort keyword has to be a Boolean value.')

        if flag is not None:
            if not isinstance(flag, bool):
                raise TypeError('flag keyword has to be a Boolean value.')

        if sort:                   # sort by first antenna label
            if flag is None:       # get all baselines
                    vis = NP.asarray([self.interferometers[label].crosspol.Vf[pol] for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0])])
                    labels = [label for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0])]   
            else:
                if flag:           # get flagged baselines
                    vis = NP.asarray([self.interferometers[label].crosspol.Vf[pol] for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0]) if self.interferometers[label].crosspol.flag[pol]])
                    labels = [label for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0]) if self.interferometers[label].crosspol.flag[pol]]                    
                else:              # get unflagged baselines
                    vis = NP.asarray([self.interferometers[label].crosspol.Vf[pol] for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0]) if not self.interferometers[label].crosspol.flag[pol]])
                    labels = [label for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0]) if not self.interferometers[label].crosspol.flag[pol]]
        else:                      # no sorting
            if flag is None:
                vis = NP.asarray([self.interferometers[label].crosspol.Vf[pol] for label in self.interferometers.keys()])
                labels = [label for label in self.interferometers.keys()]
            else:
                if flag:               # get flagged baselines
                    vis = NP.asarray([self.interferometers[label].crosspol.Vf[pol] for label in self.interferometers.keys() if self.interferometers[label].crosspol.flag[pol]])
                    labels = [label for label in self.interferometers.keys() if self.interferometers[label].crosspol.flag[pol]]                    
                else:                  # get unflagged baselines
                    vis = NP.asarray([self.interferometers[label].crosspol.Vf[pol] for label in self.interferometers.keys() if not self.interferometers[label].crosspol.flag[pol]])
                    labels = [label for label in sorted(self.interferometers.keys(), key=lambda tup: tup[0]) if not self.interferometers[label].crosspol.flag[pol]]

        outdict = {}
        outdict['labels'] = labels
        outdict['visibilities'] = vis

        return outdict

    ################################################################################# 

    def grid(self, uvspacing=0.5, uvpad=None, pow2=True, pol=None):
        
        """
        ----------------------------------------------------------------------------
        Routine to produce a grid based on the interferometer array 

        Inputs:

        uvspacing   [Scalar] Positive value indicating the maximum uv-spacing
                    desirable at the lowest wavelength (max frequency). 
                    Default = 0.5

        xypad       [List] Padding to be applied around the antenna locations 
                    before forming a grid. List elements should be positive. If it 
                    is a one-element list, the element is applicable to both x and 
                    y axes. If list contains three or more elements, only the 
                    first two elements are considered one for each axis. 
                    Default = None.

        pow2        [Boolean] If set to True, the grid is forced to have a size a 
                    next power of 2 relative to the actual sie required. If False,
                    gridding is done with the appropriate size as determined by
                    uvspacing. Default = True.

        pol         [String] The polarization to be gridded. Can be set to 'P11', 
                    'P12', 'P21', or 'P22'. If set to None, gridding for all the
                    polarizations is performed. 
        ----------------------------------------------------------------------------
        """

        if self.f is None:
            self.f = self.interferometers.itervalues().next().f

        if self.f0 is None:
            self.f0 = self.interferometers.itervalues().next().f0

        wavelength = FCNST.c / self.f
        min_lambda = NP.abs(wavelength).min()

        # Change itervalues() to values() when porting to Python 3.x
        # May have to change *blc and *trc with zip(*blc) and zip(*trc) when using Python 3.x

        blc = [[self.interferometers[label].blc[0,0], self.interferometers[label].blc[0,1]] for label in self.interferometers]
        trc = [[self.interferometers[label].trc[0,0], self.interferometers[label].trc[0,1]] for label in self.interferometers]

        self.trc = NP.amax(NP.abs(NP.vstack((NP.asarray(blc), NP.asarray(trc)))), axis=0).ravel() / min_lambda
        self.blc = -1 * self.trc

        self.gridu, self.gridv = GRD.grid_2d([(self.blc[0], self.trc[0]), (self.blc[1], self.trc[1])], pad=uvpad, spacing=uvspacing, pow2=True)

        self.grid_blc = NP.asarray([self.gridu.min(), self.gridv.min()])
        self.grid_trc = NP.asarray([self.gridu.max(), self.gridv.max()])

        self.grid_ready = True

    ################################################################################# 

    def grid_convolve(self, pol=None, antpairs=None, unconvolve_existing=False,
                      normalize=False, method='NN', distNN=NP.inf, tol=None,
                      maxmatch=None, identical_interferometers=True,
                      gridfunc_freq=None, mapping='weighted', wts_change=False,
                      parallel=False, nproc=None, pp_method='pool', verbose=True): 

        """
        ----------------------------------------------------------------------------
        Routine to project the complex illumination power pattern and the 
        visibilities on the grid. It can operate on the entire interferometer array 
        or incrementally project the visibilities and complex illumination power 
        patterns from specific interferometers on to an already existing grid. (The
        latter is not implemented yet)

        Inputs:

        pol         [String] The polarization to be gridded. Can be set to 'P11', 
                    'P12', 'P21' or 'P22'. If set to None, gridding for all the
                    polarizations is performed. Default = None

        antpairs    [instance of class InterferometerArray, single instance or list 
                    of instances of class Interferometer, or a dictionary holding 
                    instances of class Interferometer] If a dictionary is provided, 
                    the keys should be the interferometer labels and the values 
                    should be instances of class Interferometer. If a list is 
                    provided, it should be a list of valid instances of class 
                    Interferometer. These instance(s) of class Interferometer will 
                    be merged to the existing grid contained in the instance of 
                    InterferometerArray class. If ants is not provided (set to 
                    None), the gridding operations will be performed on the entire
                    set of interferometers contained in the instance of class 
                    InterferometerArray. Default = None.

        unconvolve_existing
                   [Boolean] Default = False. If set to True, the effects of
                   gridding convolution contributed by the interferometer(s) 
                   specified will be undone before updating the interferometer 
                   measurements on the grid, if the interferometer(s) is/are 
                   already found to in the set of interferometers held by the 
                   instance of InterferometerArray. If False and if one or more 
                   interferometer instances specified are already found to be held 
                   in the instance of class InterferometerArray, the code will stop
                   raising an error indicating the gridding oepration cannot
                   proceed. 

        normalize  [Boolean] Default = False. If set to True, the gridded weights
                   are divided by the sum of weights so that the gridded weights 
                   add up to unity. (Need to work on normaliation)

        method     [string] The gridding method to be used in applying the 
                   interferometer weights on to the interferometer array grid. 
                   Accepted values are 'NN' (nearest neighbour - default), 'CS' 
                   (cubic spline), or 'BL' (Bi-linear). In case of applying grid 
                   weights by 'NN' method, an optional distance upper bound for 
                   the nearest neighbour can be provided in the parameter distNN 
                   to prune the search and make it efficient. Currently, only the
                   nearest neighbour method is operational.

        distNN     [scalar] A positive value indicating the upper bound on 
                   distance to the nearest neighbour in the gridding process. It 
                   has units of distance, the same units as the interferometer 
                   attribute location and interferometer array attribute gridx 
                   and gridy. Default is NP.inf (infinite distance). It will be 
                   internally converted to have same units as interferometer 
                   attributes wtspos (units in number of wavelengths)

        maxmatch   [scalar] A positive value indicating maximum number of input 
                   locations in the interferometer grid to be assigned.
                   Default = None. If set to None, all the interferometer array 
                   grid elements specified are assigned values for each 
                   interferometer. For instance, to have only one interferometer 
                   array grid element to be populated per interferometer, use 
                   maxmatch=1. 

        tol        [scalar] If set, only lookup data with abs(val) > tol will be 
                   considered for nearest neighbour lookup. Default = None implies 
                   all lookup values will be considered for nearest neighbour 
                   determination. tol is to be interpreted as a minimum value 
                   considered as significant in the lookup table. 

        identical_interferometers
                   [boolean] indicates if all interferometer elements are to be
                   treated as identical. If True (default), they are identical
                   and their gridding kernels are identical. If False, they are
                   not identical and each one has its own gridding kernel.

        gridfunc_freq
                   [String scalar] If set to None (not provided) or to 'scale'
                   assumes that attribute wtspos is given for a
                   reference frequency which need to be scaled for the frequency
                   channels. Will be ignored if the number of elements of list 
                   in this attribute under the specific polarization are the same
                   as the number of frequency channels.

        mapping    [string] indicates the type of mapping between baseline locations
                   and the grid locations. Allowed values are 'sampled' and 
                   'weighted' (default). 'sampled' means only the baseline measurement 
                   closest ot a grid location contributes to that grid location, 
                   whereas, 'weighted' means that all the baselines contribute in
                   a weighted fashion to their nearest grid location. The former 
                   is faster but possibly discards baseline data whereas the latter
                   is slower but includes all data along with their weights.

        wts_change [boolean] indicates if weights and/or their lcoations have 
                   changed from the previous intergration or snapshot. 
                   Default=False means they have not changed. In such a case the 
                   baseline-to-grid mapping and grid illumination pattern do not 
                   have to be determined, and mapping and values from the previous 
                   snapshot can be used. If True, a new mapping has to be 
                   determined.

        parallel   [boolean] specifies if parallelization is to be invoked. 
                   False (default) means only serial processing

        nproc      [integer] specifies number of independent processes to spawn.
                   Default = None, means automatically determines the number of 
                   process cores in the system and use one less than that to 
                   avoid locking the system for other processes. Applies only 
                   if input parameter 'parallel' (see above) is set to True. 
                   If nproc is set to a value more than the number of process
                   cores in the system, it will be reset to number of process 
                   cores in the system minus one to avoid locking the system out 
                   for other processes

        pp_method  [string] specifies if the parallelization method is handled
                   automatically using multirocessing pool or managed manually
                   by individual processes and collecting results in a queue.
                   The former is specified by 'pool' (default) and the latter
                   by 'queue'. These are the two allowed values. The pool method 
                   has easier bookkeeping and can be fast if the computations 
                   not expected to be memory bound. The queue method is more
                   suited for memory bound processes but can be slower or 
                   inefficient in terms of CPU management.

        verbose    [boolean] If True, prints diagnostic and progress messages. 
                   If False (default), suppress printing such messages.
        ----------------------------------------------------------------------------
        """

        eps = 1.0e-10
        if pol is None:
            pol = ['P11', 'P12', 'P21', 'P22']
        elif not isinstance(pol, list):
            pol = [pol]

        if not self.grid_ready:
            self.grid()

        crosspol = ['P11', 'P12', 'P21', 'P22']

        for cpol in crosspol:
            if cpol in pol:

                if antpairs is not None:
    
                    if isinstance(antpairs, Interferometer):
                        antpairs = [antpairs]
    
                    if isinstance(antpairs, (dict, InterferometerArray)):
                        # Check if these interferometers are new or old and compatible
                        for key in antpairs: 
                            if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                                if key in self.interferometers:
                                    if unconvolve_existing: # Effects on the grid of interferometers already existing must be removed 
                                        if self.interferometers[key]._gridinfo[cpol]: # if gridding info is not empty
                                            for i in range(len(self.f)):
                                                self.grid_unconvolve(antpairs[key].label)
                                    else:
                                        raise KeyError('Interferometer {0} already found to exist in the dictionary of interferometers but cannot proceed grid_convolve() without unconvolving first.'.format(antpairs[key].label)) 
                                
                            else:
                                del antpairs[key] # remove the dictionary element since it is not an Interferometer instance
                
                    if identical_interferometers and (gridfunc_freq == 'scale'):
                        bl_dict = self.baseline_vectors(pol=cpol, flag=False, sort=True)
                        bl_xy = bl_dict['baselines'][:,:2]
                        self.ordered_labels = bl_dict['labels']
                        n_bl = bl_xy.shape[0]

                        vis_dict = self.get_visibilities(cpol, flag=False, sort=True)
                        vis = vis_dict['visibilities'].astype(NP.complex64)

                        # Since antenna pairs are identical, read from first antenna pair, since wtspos are scaled with frequency, read from first frequency channel
                        wtspos_xy = antpairs[0].wtspos[cpol][0] * FCNST.c/self.f[0] 
                        wts = antpairs[0].wts[cpol][0]
                        n_wts = wts.size

                        reflocs_xy = bl_xy[:,NP.newaxis,:] + wtspos_xy[NP.newaxis,:,:]
                        refwts_xy = wts.reshape(1,-1) * NP.ones((n_bl,1))

                        reflocs_xy = reflocs_xy.reshape(-1,bl_xy.shape[1])
                        refwts_xy = refwts_xy.reshape(-1,1).astype(NP.complex64)
                        reflocs_uv = reflocs_xy[:,NP.newaxis,:] * self.f.reshape(1,-1,1) / FCNST.c
                        refwts_uv = refwts_xy * NP.ones((1,self.f.size))
                        reflocs_uv = reflocs_uv.reshape(-1,bl_xy.shape[1])
                        refwts_uv = refwts_uv.reshape(-1,1).ravel()

                        inplocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                        ibind, nnval = LKP.lookup_1NN(reflocs_uv, refwts_uv, inplocs,
                                                      distance_ULIM=distNN*self.f.max()/FCNST.c,
                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                        
                else:  

                    bl_dict = self.baseline_vectors(pol=cpol, flag=None, sort=True)
                    self.ordered_labels = bl_dict['labels']
                    bl_xy = bl_dict['baselines'][:,:2] # n_bl x 2
                    n_bl = bl_xy.shape[0]

                    Vf_dict = self.get_visibilities(cpol, flag=None, sort=True)
                    Vf = Vf_dict['visibilities'].astype(NP.complex64)  # n_bl x nchan
                    if Vf.shape[0] != n_bl:
                        raise ValueError('Encountered unexpected behavior. Need to debug.')
                    if verbose:
                        print 'Gathered baseline data for gridding convolution for timestamp {0}'.format(self.timestamp)

                    if wts_change or (not self.grid_mapper[cpol]['labels']):
    
                        if gridfunc_freq == 'scale':
                            if identical_interferometers:
    
                                wts_tol = 1e-6
    
                                # Since antenna pairs are identical, read from first antenna pair, since wtspos are scaled with frequency, read from first frequency channel
    
                                wtspos_xy = self.interferometers.itervalues().next().wtspos[cpol][0] * FCNST.c/self.f[0] 
                                wts = self.interferometers.itervalues().next().wts[cpol][0].astype(NP.complex64)
                                wtspos_xy = wtspos_xy[NP.abs(wts) >= wts_tol, :]
                                wts = wts[NP.abs(wts) >= wts_tol]
                                n_wts = wts.size
        
                                reflocs_xy = bl_xy[:,NP.newaxis,:] + wtspos_xy[NP.newaxis,:,:] # n_bl x n_wts x 2 
                                refwts = wts.reshape(1,-1) * NP.ones((n_bl,1))  # n_bl x n_wts
                            else:
                                for i,label in enumerate(self.ordered_labels):
                                    bl_wtspos = self.interferometers[label].wtspos[cpol][0]
                                    bl_wts = self.interferometers[label].wts[cpol][0].astype(NP.complex64)
                                    if i == 0:
                                        wtspos = bl_wtspos[NP.newaxis,:,:] # 1 x n_wts x 2
                                        refwts = bl_wts.reshape(1,-1) # 1 x n_wts
                                    else:
                                        wtspos = NP.vstack((wtspos, bl_wtspos[NP.newaxis,:,:])) # n_bl x n_wts x 2
                                        refwts = NP.vstack((refwts, bl_wts.reshape(1,-1))) # n_bl x n_wts
                                    reflocs_xy = bl_xy[:,NP.newaxis,:] + wtspos * FCNST.c/self.f[0] # n_bl x n_wts x 2
                                    
                            reflocs_xy = reflocs_xy.reshape(-1,bl_xy.shape[1])  # (n_bl x n_wts) x 2
                            refwts = refwts.ravel()
                            self.grid_mapper[cpol]['refwts'] = NP.copy(refwts.ravel()) # (n_bl x n_wts)
                            
                        else: # Weights do not scale with frequency (needs serious development)
                            pass
                            
                        gridlocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                        contributed_bl_grid_Vf = None

                        if parallel:    # Use parallelization over frequency to determine gridding convolution
                            if nproc is None:
                                nproc = max(MP.cpu_count()-1, 1) 
                            else:
                                nproc = min(nproc, max(MP.cpu_count()-1, 1))
                            
                            if pp_method == 'queue':  ## Use MP.Queue(): useful for memory intensive parallelizing but can be slow
                                job_chunk_begin = range(0,self.f.size,nproc)
                                if verbose:
                                    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} job chunks '.format(len(job_chunk_begin)), PGB.ETA()], maxval=len(job_chunk_begin)).start()

                                for ijob, job_start in enumerate(job_chunk_begin):
                                    pjobs = []
                                    out_q = MP.Queue()
                                    for job_ind in xrange(job_start, min(job_start+nproc, self.f.size)):    # Start the processes and store outputs in the queue
                                        if mapping == 'weighted':
                                            pjob = MP.Process(target=LKP.find_1NN_pp, args=(gridlocs, reflocs_xy * self.f[job_ind]/FCNST.c, job_ind, out_q, distNN*self.f.max()/FCNST.c, True), name='process-{0:0d}-channel-{1:0d}'.format(job_ind-job_start, job_ind))
                                        else:
                                            pjob = MP.Process(target=LKP.find_1NN_pp, args=(reflocs_xy * self.f[job_ind]/FCNST.c, gridlocs, job_ind, out_q, distNN*self.f.max()/FCNST.c, True), name='process-{0:0d}-channel-{1:0d}'.format(job_ind-job_start, job_ind))             
                                        pjob.start()
                                        pjobs.append(pjob)
                                   
                                    for p in xrange(len(pjobs)):   # Unpack the queue output
                                        outdict = out_q.get()
                                        chan = outdict.keys()[0]
                                        if mapping == 'weighted':
                                            refind, gridind = outdict[chan]['inpind'], outdict[chan]['refind']
                                        else:
                                            gridind, refind = outdict[chan]['inpind'], outdict[chan]['refind']                                            
                                        self.grid_mapper[cpol]['refind'] += [refind]
                                        self.grid_mapper[cpol]['gridind'] += [gridind]

                                        bl_ind, lkp_ind = NP.unravel_index(refind, (n_bl, n_wts))
                                        self.grid_mapper[cpol]['bl']['ind_freq'] += [bl_ind]
                                        gridind_unraveled = NP.unravel_index(gridind, self.gridu.shape) + (chan+NP.zeros(gridind.size,dtype=int),)
                                        gridind_raveled = NP.ravel_multi_index(gridind_unraveled, self.gridu.shape+(self.f.size,))

                                        if self.grid_mapper[cpol]['bl']['ind_all'] is None:
                                            self.grid_mapper[cpol]['bl']['ind_all'] = NP.copy(bl_ind)
                                            self.grid_mapper[cpol]['bl']['illumination'] = refwts[refind]
                                            contributed_bl_grid_Vf = refwts[refind] * Vf[bl_ind,chan]
                                            self.grid_mapper[cpol]['grid']['ind_all'] = NP.copy(gridind_raveled)
                                        else:
                                            self.grid_mapper[cpol]['bl']['ind_all'] = NP.append(self.grid_mapper[cpol]['bl']['ind_all'], bl_ind)
                                            self.grid_mapper[cpol]['bl']['illumination'] = NP.append(self.grid_mapper[cpol]['bl']['illumination'], refwts[refind])
                                            contributed_bl_grid_Vf = NP.append(contributed_bl_grid_Vf, refwts[refind] * Vf[bl_ind,chan])
                                            self.grid_mapper[cpol]['grid']['ind_all'] = NP.append(self.grid_mapper[cpol]['grid']['ind_all'], gridind_raveled)
    
                                    for pjob in pjobs:
                                        pjob.join()

                                    del out_q

                                    if verbose:
                                        progress.update(ijob+1)
                                if verbose:
                                    progress.finish()

                            elif pp_method == 'pool':   ## Using MP.Pool.map(): Can be faster if parallelizing is not memory intensive
                                list_of_gridlocs = [gridlocs] * self.f.size
                                list_of_reflocs = [reflocs_xy * f/FCNST.c for f in self.f]
                                list_of_dist_NN = [distNN*self.f.max()/FCNST.c] * self.f.size
                                list_of_remove_oob = [True] * self.f.size

                                pool = MP.Pool(processes=nproc)
                                if mapping == 'weighted':
                                    list_of_NNout = pool.map(find_1NN_arg_splitter, IT.izip(list_of_gridlocs, list_of_reflocs, list_of_dist_NN, list_of_remove_oob))
                                else:
                                    list_of_NNout = pool.map(find_1NN_arg_splitter, IT.izip(list_of_reflocs, list_of_gridlocs, list_of_dist_NN, list_of_remove_oob))

                                pool.close()
                                pool.join()

                                for chan, NNout in enumerate(list_of_NNout):    # Unpack the pool output
                                    if mapping == 'weighted':
                                        refind, gridind = NNout[0], NNout[1]
                                    else:
                                        gridind, refind = NNout[0], NNout[1]

                                    self.grid_mapper[cpol]['refind'] += [refind]
                                    self.grid_mapper[cpol]['gridind'] += [gridind]

                                    bl_ind, lkp_ind = NP.unravel_index(refind, (n_bl, n_wts))
                                    self.grid_mapper[cpol]['bl']['ind_freq'] += [bl_ind]
                                    gridind_unraveled = NP.unravel_index(gridind, self.gridu.shape) + (chan+NP.zeros(gridind.size,dtype=int),)
                                    gridind_raveled = NP.ravel_multi_index(gridind_unraveled, self.gridu.shape+(self.f.size,))

                                    if chan == 0:
                                        self.grid_mapper[cpol]['bl']['ind_all'] = NP.copy(bl_ind)
                                        self.grid_mapper[cpol]['bl']['illumination'] = refwts[refind]
                                        contributed_bl_grid_Vf = refwts[refind] * Vf[bl_ind,chan]
                                        self.grid_mapper[cpol]['grid']['ind_all'] = NP.copy(gridind_raveled)
                                    else:
                                        self.grid_mapper[cpol]['bl']['ind_all'] = NP.append(self.grid_mapper[cpol]['bl']['ind_all'], bl_ind)
                                        self.grid_mapper[cpol]['bl']['illumination'] = NP.append(self.grid_mapper[cpol]['bl']['illumination'], refwts[refind])
                                        contributed_bl_grid_Vf = NP.append(contributed_bl_grid_Vf, refwts[refind] * Vf[bl_ind,chan])
                                        self.grid_mapper[cpol]['grid']['ind_all'] = NP.append(self.grid_mapper[cpol]['grid']['ind_all'], gridind_raveled)

                            else:
                                raise ValueError('Parallel processing method specified by input parameter ppmethod has to be "pool" or "queue"')
                            
                        else:    # Use serial processing over frequency to determine gridding convolution

                            if verbose:
                                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Frequency channels '.format(self.f.size), PGB.ETA()], maxval=self.f.size).start()
    
                            for i in xrange(self.f.size):
                                if mapping == 'weighted':
                                    refind, gridind = LKP.find_1NN(gridlocs, reflocs_xy * self.f[i]/FCNST.c, 
                                                                  distance_ULIM=distNN*self.f.max()/FCNST.c,
                                                                  remove_oob=True)[:2]
                                else:
                                    gridind, refind = LKP.find_1NN(reflocs_xy * self.f[i]/FCNST.c, gridlocs,
                                                                  distance_ULIM=distNN*self.f.max()/FCNST.c,
                                                                  remove_oob=True)[:2]
                                
                                self.grid_mapper[cpol]['refind'] += [refind]
                                self.grid_mapper[cpol]['gridind'] += [gridind]
    
                                bl_ind, lkp_ind = NP.unravel_index(refind, (n_bl, n_wts))
                                self.grid_mapper[cpol]['bl']['ind_freq'] += [bl_ind]
                                gridind_unraveled = NP.unravel_index(gridind, self.gridu.shape) + (i+NP.zeros(gridind.size,dtype=int),)
                                gridind_raveled = NP.ravel_multi_index(gridind_unraveled, self.gridu.shape+(self.f.size,))
                                if i == 0:
                                    self.grid_mapper[cpol]['bl']['ind_all'] = NP.copy(bl_ind)
                                    self.grid_mapper[cpol]['bl']['illumination'] = refwts[refind]
                                    contributed_bl_grid_Vf = refwts[refind] * Vf[bl_ind,i]
                                    self.grid_mapper[cpol]['grid']['ind_all'] = NP.copy(gridind_raveled)
                                else:
                                    self.grid_mapper[cpol]['bl']['ind_all'] = NP.append(self.grid_mapper[cpol]['bl']['ind_all'], bl_ind)
                                    self.grid_mapper[cpol]['bl']['illumination'] = NP.append(self.grid_mapper[cpol]['bl']['illumination'], refwts[refind])
                                    contributed_bl_grid_Vf = NP.append(contributed_bl_grid_Vf, refwts[refind] * Vf[bl_ind,i])
                                    self.grid_mapper[cpol]['grid']['ind_all'] = NP.append(self.grid_mapper[cpol]['grid']['ind_all'], gridind_raveled)
    
                                if verbose:
                                    progress.update(i+1)
                            if verbose:
                                progress.finish()
                                
                        self.grid_mapper[cpol]['bl']['uniq_ind_all'] = NP.unique(self.grid_mapper[cpol]['bl']['ind_all'])
                        self.grid_mapper[cpol]['bl']['rev_ind_all'] = OPS.binned_statistic(self.grid_mapper[cpol]['bl']['ind_all'], statistic='count', bins=NP.append(self.grid_mapper[cpol]['bl']['uniq_ind_all'], self.grid_mapper[cpol]['bl']['uniq_ind_all'].max()+1))[3]

                        if parallel and (mapping == 'weighted'):    # Use parallel processing over baselines to determine baseline-grid mapping of gridded aperture illumination and visibilities

                            if nproc is None:
                                nproc = max(MP.cpu_count()-1, 1) 
                            else:
                                nproc = min(nproc, max(MP.cpu_count()-1, 1))

                            if pp_method == 'queue':  ## Use MP.Queue(): useful for memory intensive parallelizing but can be slow

                                num_bl = self.grid_mapper[cpol]['bl']['uniq_ind_all'].size
                                job_chunk_begin = range(0,num_bl,nproc)
                                if verbose:
                                    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} job chunks '.format(len(job_chunk_begin)), PGB.ETA()], maxval=len(job_chunk_begin)).start()

                                for ijob, job_start in enumerate(job_chunk_begin):
                                    pjobs1 = []
                                    pjobs2 = []
                                    out_q1 = MP.Queue()
                                    out_q2 = MP.Queue()
    
                                    for job_ind in xrange(job_start, min(job_start+nproc, num_bl)):   # Start the parallel processes and store the output in the queue
                                        label = self.ordered_labels[self.grid_mapper[cpol]['bl']['uniq_ind_all'][job_ind]]
    
                                        if self.grid_mapper[cpol]['bl']['rev_ind_all'][job_ind] < self.grid_mapper[cpol]['bl']['rev_ind_all'][job_ind+1]:
        
                                            self.grid_mapper[cpol]['labels'][label] = {}
                                            self.grid_mapper[cpol]['labels'][label]['flag'] = self.interferometers[label].crosspol.flag[cpol]
        
                                            select_bl_ind = self.grid_mapper[cpol]['bl']['rev_ind_all'][self.grid_mapper[cpol]['bl']['rev_ind_all'][job_ind]:self.grid_mapper[cpol]['bl']['rev_ind_all'][job_ind+1]]
                                            gridind_raveled_around_bl = self.grid_mapper[cpol]['grid']['ind_all'][select_bl_ind]
                                            uniq_gridind_raveled_around_bl = NP.unique(gridind_raveled_around_bl)
                                            self.grid_mapper[cpol]['labels'][label]['gridind'] = uniq_gridind_raveled_around_bl
                                            pjob1 = MP.Process(target=baseline_grid_mapper, args=(gridind_raveled_around_bl, contributed_bl_grid_Vf[select_bl_ind], NP.append(uniq_gridind_raveled_around_bl, uniq_gridind_raveled_around_bl.max()+1), label, out_q1), name='process-{0:0d}-{1}-visibility'.format(job_ind, label))
                                            pjob2 = MP.Process(target=baseline_grid_mapper, args=(gridind_raveled_around_bl, self.grid_mapper[cpol]['bl']['illumination'][select_bl_ind], NP.append(uniq_gridind_raveled_around_bl, uniq_gridind_raveled_around_bl.max()+1), label, out_q2), name='process-{0:0d}-{1}-illumination'.format(job_ind, label))
                                            pjob1.start()
                                            pjob2.start()
                                            pjobs1.append(pjob1)
                                            pjobs2.append(pjob2)
    
                                    for p in xrange(len(pjobs1)):    # Unpack the gridded visibility and aperture illumination information from the pool output
                                        outdict = out_q1.get()
                                        label = outdict.keys()[0]
                                        self.grid_mapper[cpol]['labels'][label]['Vf'] = outdict[label]
                                        outdict = out_q2.get()
                                        label = outdict.keys()[0]
                                        self.grid_mapper[cpol]['labels'][label]['illumination'] = outdict[label]
    
                                    for pjob in pjobs1:
                                        pjob1.join()
                                    for pjob in pjobs2:
                                        pjob2.join()
        
                                    del out_q1, out_q2
                                    
                                    if verbose:
                                        progress.update(ijob+1)
                                if verbose:
                                    progress.finish()
                                    
                            elif pp_method == 'pool':    ## Using MP.Pool.map(): Can be faster if parallelizing is not memory intensive

                                list_of_gridind_raveled_around_bl = []
                                list_of_bl_grid_values = []
                                list_of_bl_Vf_contribution = []
                                list_of_bl_illumination = []
                                list_of_uniq_gridind_raveled_around_bl = []
                                list_of_bl_labels = []
    
                                for j in xrange(self.grid_mapper[cpol]['bl']['uniq_ind_all'].size): # re-determine gridded visibilities due to each baseline
    
                                    label = self.ordered_labels[self.grid_mapper[cpol]['bl']['uniq_ind_all'][j]]
                                    if self.grid_mapper[cpol]['bl']['rev_ind_all'][j] < self.grid_mapper[cpol]['bl']['rev_ind_all'][j+1]:
    
                                        self.grid_mapper[cpol]['labels'][label] = {}
                                        self.grid_mapper[cpol]['labels'][label]['flag'] = self.interferometers[label].crosspol.flag[cpol]
    
                                        select_bl_ind = self.grid_mapper[cpol]['bl']['rev_ind_all'][self.grid_mapper[cpol]['bl']['rev_ind_all'][j]:self.grid_mapper[cpol]['bl']['rev_ind_all'][j+1]]
                                        gridind_raveled_around_bl = self.grid_mapper[cpol]['grid']['ind_all'][select_bl_ind]
                                        uniq_gridind_raveled_around_bl = NP.unique(gridind_raveled_around_bl)
                                        self.grid_mapper[cpol]['labels'][label]['gridind'] = uniq_gridind_raveled_around_bl
                                        
                                        list_of_bl_labels += [label]
                                        list_of_gridind_raveled_around_bl += [gridind_raveled_around_bl]
                                        list_of_uniq_gridind_raveled_around_bl += [NP.append(uniq_gridind_raveled_around_bl, uniq_gridind_raveled_around_bl.max()+1)]
                                        list_of_bl_Vf_contribution += [contributed_bl_grid_Vf[select_bl_ind]]
                                        list_of_bl_illumination += [self.grid_mapper[cpol]['bl']['illumination'][select_bl_ind]]
    
                                pool = MP.Pool(processes=nproc)
                                list_of_bl_grid_values = pool.map(baseline_grid_mapping_arg_splitter, IT.izip(list_of_gridind_raveled_around_bl, list_of_bl_Vf_contribution, list_of_uniq_gridind_raveled_around_bl))
                                pool.close()
                                pool.join()
    
                                for label,grid_values in IT.izip(list_of_bl_labels, list_of_bl_grid_values):    # Unpack the gridded visibility information from the pool output
                                    self.grid_mapper[cpol]['labels'][label]['Vf'] = grid_values
    
                                if nproc is None:
                                    pool = MP.Pool(processes=nproc)
                                else:
                                    pool = MP.Pool()
                                list_of_bl_grid_values = pool.map(baseline_grid_mapping_arg_splitter, IT.izip(list_of_gridind_raveled_around_bl, list_of_bl_illumination, list_of_uniq_gridind_raveled_around_bl))
                                pool.close()
                                pool.join()
    
                                for label,grid_values in IT.izip(list_of_bl_labels, list_of_bl_grid_values):    # Unpack the gridded visibility and aperture illumination information from the pool output
                                    self.grid_mapper[cpol]['labels'][label]['illumination'] = grid_values
                                
                                del list_of_bl_grid_values, list_of_gridind_raveled_around_bl, list_of_bl_Vf_contribution, list_of_bl_illumination, list_of_uniq_gridind_raveled_around_bl, list_of_bl_labels

                            else:
                                raise ValueError('Parallel processing method specified by input parameter ppmethod has to be "pool" or "queue"')

                        else:    # Use serial processing over baselines to determine baseline-grid mapping of gridded aperture illumination and visibilities

                            if verbose:
                                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Baselines '.format(self.grid_mapper[cpol]['bl']['uniq_ind_all'].size), PGB.ETA()], maxval=self.grid_mapper[cpol]['bl']['uniq_ind_all'].size).start()

                            for j in xrange(self.grid_mapper[cpol]['bl']['uniq_ind_all'].size):
                                label = self.ordered_labels[self.grid_mapper[cpol]['bl']['uniq_ind_all'][j]]
                                if self.grid_mapper[cpol]['bl']['rev_ind_all'][j] < self.grid_mapper[cpol]['bl']['rev_ind_all'][j+1]:
                                    select_bl_ind = self.grid_mapper[cpol]['bl']['rev_ind_all'][self.grid_mapper[cpol]['bl']['rev_ind_all'][j]:self.grid_mapper[cpol]['bl']['rev_ind_all'][j+1]]
                                    self.grid_mapper[cpol]['labels'][label] = {}
                                    self.grid_mapper[cpol]['labels'][label]['flag'] = self.interferometers[label].crosspol.flag[cpol]
                                    if mapping == 'weighted':
                                        gridind_raveled_around_bl = self.grid_mapper[cpol]['grid']['ind_all'][select_bl_ind]
                                        uniq_gridind_raveled_around_bl = NP.unique(gridind_raveled_around_bl)
                                        self.grid_mapper[cpol]['labels'][label]['gridind'] = uniq_gridind_raveled_around_bl
    
                                        self.grid_mapper[cpol]['labels'][label]['Vf'] = OPS.binned_statistic(gridind_raveled_around_bl, contributed_bl_grid_Vf[select_bl_ind].real, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_bl, uniq_gridind_raveled_around_bl.max()+1))[0]
                                        self.grid_mapper[cpol]['labels'][label]['Vf'] = self.grid_mapper[cpol]['labels'][label]['Vf'].astype(NP.complex64)
                                        self.grid_mapper[cpol]['labels'][label]['Vf'] += 1j * OPS.binned_statistic(gridind_raveled_around_bl, contributed_bl_grid_Vf[select_bl_ind].imag, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_bl, uniq_gridind_raveled_around_bl.max()+1))[0]
    
                                        self.grid_mapper[cpol]['labels'][label]['illumination'] = OPS.binned_statistic(gridind_raveled_around_bl, self.grid_mapper[cpol]['bl']['illumination'][select_bl_ind].real, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_bl, uniq_gridind_raveled_around_bl.max()+1))[0]
                                        self.grid_mapper[cpol]['labels'][label]['illumination'] = self.grid_mapper[cpol]['labels'][label]['illumination'].astype(NP.complex64)
                                        self.grid_mapper[cpol]['labels'][label]['illumination'] += 1j * OPS.binned_statistic(gridind_raveled_around_bl, self.grid_mapper[cpol]['bl']['illumination'][select_bl_ind].imag, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_bl, uniq_gridind_raveled_around_bl.max()+1))[0]
    
                                    else:
                                        self.grid_mapper[cpol]['labels'][label]['gridind'] = self.grid_mapper[cpol]['grid']['ind_all'][select_bl_ind]
                                        self.grid_mapper[cpol]['labels'][label]['Vf'] = contributed_bl_grid_Vf[select_bl_ind]
                                        self.grid_mapper[cpol]['labels'][label]['illumination'] = self.grid_mapper[cpol]['bl']['illumination'][select_bl_ind]
                                        
                                if verbose:
                                    progress.update(j+1)
                            if verbose:
                                progress.finish()
                            
                    else: # Only re-determine gridded visibilities

                        if verbose:
                            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Frequency channels '.format(self.f.size), PGB.ETA()], maxval=self.f.size).start()

                        for i in xrange(self.f.size): # Only re-estimate visibilities contributed by baselines
                            bl_refwts = self.grid_mapper[cpol]['refwts'][self.grid_mapper[cpol]['refind'][i]]
                            bl_Vf = Vf[self.grid_mapper[cpol]['bl']['ind_freq'][i],i]
                            if i == 0:
                                contributed_bl_grid_Vf = bl_refwts * bl_Vf
                            else:
                                contributed_bl_grid_Vf = NP.append(contributed_bl_grid_Vf, bl_refwts * bl_Vf)

                            if verbose:
                                progress.update(i+1)
                        if verbose:
                            progress.finish()

                        if parallel and (mapping == 'weighted'):    # Use parallel processing

                            if nproc is None:
                                nproc = max(MP.cpu_count()-1, 1) 
                            else:
                                nproc = min(nproc, max(MP.cpu_count()-1, 1))                            

                            if pp_method == 'queue':   ## Use MP.Queue(): useful for memory intensive parallelizing but can be slow

                                num_bl = self.grid_mapper[cpol]['bl']['uniq_ind_all'].size
                                job_chunk_begin = range(0,num_bl,nproc)
                                if verbose:
                                    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} job chunks '.format(len(job_chunk_begin)), PGB.ETA()], maxval=len(job_chunk_begin)).start()

                                for ijob, job_start in enumerate(job_chunk_begin):
                                    pjobs = []
                                    out_q = MP.Queue()
    
                                    for job_ind in xrange(job_start, min(job_start+nproc, num_bl)):    # Start the parallel processes and store the outputs in a queue
                                        label = self.ordered_labels[self.grid_mapper[cpol]['bl']['uniq_ind_all'][job_ind]]
    
                                        if self.grid_mapper[cpol]['bl']['rev_ind_all'][job_ind] < self.grid_mapper[cpol]['bl']['rev_ind_all'][job_ind+1]:
        
                                            select_bl_ind = self.grid_mapper[cpol]['bl']['rev_ind_all'][self.grid_mapper[cpol]['bl']['rev_ind_all'][job_ind]:self.grid_mapper[cpol]['bl']['rev_ind_all'][job_ind+1]]
                                            gridind_raveled_around_bl = self.grid_mapper[cpol]['grid']['ind_all'][select_bl_ind]
                                            uniq_gridind_raveled_around_bl = self.grid_mapper[cpol]['labels'][label]['gridind']
                                            pjob = MP.Process(target=baseline_grid_mapper, args=(gridind_raveled_around_bl, contributed_bl_grid_Vf[select_bl_ind], NP.append(uniq_gridind_raveled_around_bl, uniq_gridind_raveled_around_bl.max()+1), label, out_q), name='process-{0:0d}-{1}-visibility'.format(job_ind, label))
    
                                            pjob.start()
                                            pjobs.append(pjob)
    
                                    for p in xrange(len(pjobs)):    # Unpack the gridded visibility information from the queue
                                        outdict = out_q.get()
                                        label = outdict.keys()[0]
                                        self.grid_mapper[cpol]['labels'][label]['Vf'] = outdict[label]
    
                                    for pjob in pjobs:
                                        pjob.join()
        
                                    del out_q
                                    
                                    if verbose:
                                        progress.update(ijob+1)
                                if verbose:
                                    progress.finish()

                            else:    ## Use MP.Pool.map(): Can be faster if parallelizing is not memory intensive

                                list_of_gridind_raveled_around_bl = []
                                list_of_bl_Vf_contribution = []
                                list_of_uniq_gridind_raveled_around_bl = []
                                list_of_bl_labels = []
                                for j in xrange(self.grid_mapper[cpol]['bl']['uniq_ind_all'].size): # re-determine gridded visibilities due to each baseline
                                    if self.grid_mapper[cpol]['bl']['rev_ind_all'][j] < self.grid_mapper[cpol]['bl']['rev_ind_all'][j+1]:
                                        select_bl_ind = self.grid_mapper[cpol]['bl']['rev_ind_all'][self.grid_mapper[cpol]['bl']['rev_ind_all'][j]:self.grid_mapper[cpol]['bl']['rev_ind_all'][j+1]]
                                        label = self.ordered_labels[self.grid_mapper[cpol]['bl']['uniq_ind_all'][j]]
                                        gridind_raveled_around_bl = self.grid_mapper[cpol]['grid']['ind_all'][select_bl_ind]
                                        uniq_gridind_raveled_around_bl = NP.unique(gridind_raveled_around_bl)
                                        list_of_bl_labels += [label]
                                        list_of_gridind_raveled_around_bl += [gridind_raveled_around_bl]
                                        list_of_uniq_gridind_raveled_around_bl += [NP.append(uniq_gridind_raveled_around_bl, uniq_gridind_raveled_around_bl.max()+1)]
                                        list_of_bl_Vf_contribution += [contributed_bl_grid_Vf[select_bl_ind]]
                                if nproc is None:
                                    nproc = max(MP.cpu_count()-1, 1) 
                                else:
                                    nproc = min(nproc, max(MP.cpu_count()-1, 1))
                                pool = MP.Pool(processes=nproc)
                                list_of_grid_Vf = pool.map(baseline_grid_mapping_arg_splitter, IT.izip(list_of_gridind_raveled_around_bl, list_of_bl_Vf_contribution, list_of_uniq_gridind_raveled_around_bl))
                                pool.close()
                                pool.join()
    
                                for label,grid_Vf in IT.izip(list_of_bl_labels, list_of_grid_Vf):    # Unpack the gridded visibility information from the pool output
                                    self.grid_mapper[cpol]['labels'][label]['Vf'] = grid_Vf
                                
                                del list_of_gridind_raveled_around_bl, list_of_grid_Vf, list_of_bl_Vf_contribution, list_of_uniq_gridind_raveled_around_bl, list_of_bl_labels

                        else:          # use serial processing
                            if verbose:
                                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Baselines '.format(self.grid_mapper[cpol]['bl']['uniq_ind_all'].size), PGB.ETA()], maxval=self.grid_mapper[cpol]['bl']['uniq_ind_all'].size).start()

                            for j in xrange(self.grid_mapper[cpol]['bl']['uniq_ind_all'].size): # re-determine gridded visibilities due to each baseline
                                if self.grid_mapper[cpol]['bl']['rev_ind_all'][j] < self.grid_mapper[cpol]['bl']['rev_ind_all'][j+1]:
                                    select_bl_ind = self.grid_mapper[cpol]['bl']['rev_ind_all'][self.grid_mapper[cpol]['bl']['rev_ind_all'][j]:self.grid_mapper[cpol]['bl']['rev_ind_all'][j+1]]
                                    label = self.ordered_labels[self.grid_mapper[cpol]['bl']['uniq_ind_all'][j]]
                                    self.grid_mapper[cpol]['labels'][label]['Vf'] = {}
                                    if mapping == 'weighted':
                                        gridind_raveled_around_bl = self.grid_mapper[cpol]['grid']['ind_all'][select_bl_ind]
                                        uniq_gridind_raveled_around_bl = self.grid_mapper[cpol]['labels'][label]['gridind']
                                        # uniq_gridind_raveled_around_bl = NP.unique(gridind_raveled_around_bl)
                                        self.grid_mapper[cpol]['labels'][label]['Vf'] = OPS.binned_statistic(gridind_raveled_around_bl, contributed_bl_grid_Vf[select_bl_ind].real, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_bl, uniq_gridind_raveled_around_bl.max()+1))[0]
                                        self.grid_mapper[cpol]['labels'][label]['Vf'] = self.grid_mapper[cpol]['labels'][label]['Vf'].astype(NP.complex64)
                                        self.grid_mapper[cpol]['labels'][label]['Vf'] += 1j * OPS.binned_statistic(gridind_raveled_around_bl, contributed_bl_grid_Vf[select_bl_ind].imag, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_bl, uniq_gridind_raveled_around_bl.max()+1))[0]
                                    else:
                                        self.grid_mapper[cpol]['labels'][label]['Vf'] = contributed_bl_grid_Vf[select_bl_ind]
                                if verbose:
                                    progress.update(j+1)
                            if verbose:
                                progress.finish()

    ################################################################################# 

    def make_grid_cube(self, pol=None, verbose=True):

        """
        ----------------------------------------------------------------------------
        Constructs the grid of complex power illumination and visibilities using 
        the gridding information determined for every baseline. Flags are taken
        into account while constructing this grid.

        Inputs:

        pol     [String] The polarization to be gridded. Can be set to 'P11', 
                'P12', 'P21' or 'P22'. If set to None, gridding for all the
                polarizations is performed. Default = None
        
        verbose [boolean] If True, prints diagnostic and progress messages. 
                If False (default), suppress printing such messages.
        ----------------------------------------------------------------------------
        """

        if pol is None:
            pol = ['P11', 'P12', 'P21', 'P22']

        pol = NP.unique(NP.asarray(pol))
        
        for cpol in pol:

            if verbose:
                print 'Gridding aperture illumination and visibilities for polarization {0} ...'.format(cpol)

            if cpol not in ['P11', 'P12', 'P21', 'P22']:
                raise ValueError('Invalid specification for input parameter pol')

            if cpol not in self._bl_contribution:
                raise KeyError('Key {0} not found in attribute _bl_contribution'.format(cpol))
    
            self.grid_illumination[cpol] = NP.zeros((self.gridu.shape + (self.f.size,)), dtype=NP.complex_)
            self.grid_Vf[cpol] = NP.zeros((self.gridu.shape + (self.f.size,)), dtype=NP.complex_)
    
            labels = self.grid_mapper[cpol]['labels'].keys()
            if verbose:
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(len(labels)), PGB.ETA()], maxval=len(labels)).start()

            loopcount = 0
            num_unflagged = 0
            for blinfo in self.grid_mapper[cpol]['labels'].itervalues():
                if not blinfo['flag']:
                    num_unflagged += 1
                    gridind_unraveled = NP.unravel_index(blinfo['gridind'], self.gridu.shape+(self.f.size,))
                    self.grid_illumination[cpol][gridind_unraveled] += blinfo['illumination']
                    self.grid_Vf[cpol][gridind_unraveled] += blinfo['Vf']

                progress.update(loopcount+1)
                loopcount += 1
            progress.finish()
                
            if verbose:
                print 'Gridded aperture illumination and visibilities for polarization {0} from {1:0d} unflagged contributing baselines'.format(cpol, num_unflagged)

    ################################################################################# 

    def grid_convolve_old(self, pol=None, antpairs=None, unconvolve_existing=False,
                          normalize=False, method='NN', distNN=NP.inf, tol=None,
                          maxmatch=None): 

        """
        ----------------------------------------------------------------------------
        Routine to project the visibility illumination pattern and the visibilities
        on the grid. It can operate on the entire antenna array or
        incrementally project the visibilities and illumination patterns from
        specific antenna pairs on to an already existing grid.

        Inputs:

        pol        [String] The polarization to be gridded. Can be set to 'P1' 
                   or 'P2'. If set to None, gridding for both 'P1' and 'P2' 
                   is performed. Default = None

        ants       [instance of class AntennaArray, single instance or list of
                   instances of class Antenna, or a dictionary holding instances 
                   of class Antenna] If a dictionary is provided, the keys
                   should be the antenna labels and the values should be 
                   instances of class Antenna. If a list is provided, it should 
                   be a list of valid instances of class Antenna. These 
                   instance(s) of class Antenna will be merged to the existing 
                   grid contained in the instance of AntennaArray class. If ants 
                   is not provided (set to None), the gridding operations will 
                   be performed on the entire set of antennas contained in the 
                   instance of class AntennaArray. Default = None.

        unconvolve_existing
                   [Boolean] Default = False. If set to True, the effects of
                   gridding convolution contributed by the antenna(s) specified 
                   will be undone before updating the antenna measurements on 
                   the grid, if the antenna(s) is/are already found to in the 
                   set of antennas held by the instance of AntennaArray. If 
                   False and if one or more antenna instances specified are 
                   already found to be held in the instance of class 
                   AntennaArray, the code will stop raising an error indicating 
                   the gridding operation cannot proceed. 

        normalize  [Boolean] Default = False. If set to True, the gridded 
                   weights are divided by the sum of weights so that the gridded 
                   weights add up to unity. 

        method     [string] The gridding method to be used in applying the 
                   antenna weights on to the antenna array grid. Accepted values 
                   are 'NN' (nearest neighbour - default), 'CS' (cubic spline), 
                   or 'BL' (Bi-linear). In case of applying grid weights by 'NN' 
                   method, an optional distance upper bound for the nearest 
                   neighbour can be provided in the parameter distNN to prune 
                   the search and make it efficient

        distNN     [scalar] A positive value indicating the upper bound on 
                   distance to the nearest neighbour in the gridding process. 
                   It has units of distance, the same units as the antenna 
                   attribute location and antenna array attribute gridx and 
                   gridy. Default is NP.inf (infinite distance). It will be 
                   internally converted to have same units as antenna attributes 
                   wtspos (units in number of wavelengths)

        maxmatch   [scalar] A positive value indicating maximum number of input 
                   locations in the antenna grid to be assigned. Default = None. 
                   If set to None, all the antenna array grid elements specified 
                   are assigned values for each antenna. For instance, to have 
                   only one antenna array grid element to be populated per 
                   antenna, use maxmatch=1. 

        tol        [scalar] If set, only lookup data with abs(val) > tol will be 
                   considered for nearest neighbour lookup. Default = None 
                   implies all lookup values will be considered for nearest 
                   neighbour determination. tol is to be interpreted as a minimum 
                   value considered as significant in the lookup table. 
        ----------------------------------------------------------------------------
        """

        eps = 1.0e-10

        if not self.grid_ready:
            self.grid()

        if (pol is None) or (pol == 'P11'):

            if antpairs is not None:

                if isinstance(antpairs, Interferometer):
                    antpairs = [antpairs]

                if isinstance(antpairs, (dict, InterferometerArray)):
                    # Check if these interferometers are new or old and compatible
                    for key in antpairs: 
                        if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                            if key in self.interferometers:
                                if unconvolve_existing: # Effects on the grid of interferometers already existing must be removed 
                                    if self.interferometers[key]._gridinfo['P11']: # if gridding info is not empty
                                        for i in range(len(self.f)):
                                            self.grid_unconvolve(antpairs[key].label)
                                else:
                                    raise KeyError('Interferometer {0} already found to exist in the dictionary of interferometers but cannot proceed grid_convolve() without unconvolving first.'.format(antpairs[key].label)) 
                            
                        else:
                            del antpairs[key] # remove the dictionary element since it is not an Interferometer instance

                    for key in antpairs:
                        if not antpairs[key].crosspol.flag['P11']:
                            for i in range(len(self.f)):
                                if method == 'NN':
                                    if antpairs[key].wtspos_scale['P11'] is None: 
                                        reflocs = antpairs[key].wtspos['P11'][i] + (self.f[i]/FCNST.c) * NP.asarray([antpairs[key].location.x, antpairs[key].location.y]).reshape(1,-1)
                                        inplocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN(reflocs, antpairs[key].wts['P11'][i], inplocs,
                                                                      distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif antpairs[key].wtspos_scale['P11'] == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            reflocs = antpairs[key].wtspos['P11'][0] + (self.f[0]/FCNST.c) * NP.asarray([antpairs[key].location.x, antpairs[key].location.y]).reshape(1,-1)
                                            inplocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                            ibind, nnval = LKP.lookup_1NN(reflocs, antpairs[key].wts['P11'][0], inplocs,
                                                                          distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                          remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                            roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                            if normalize:
                                                nnval /= NP.sum(nnval)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination['P11'][roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                    self.grid_Vf['P11'][roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += antpairs[key].crosspol.Vf['P11'][i] * nnval
                                else:
                                    if antpairs[key].wtspos_scale['P11'] is None: 
                                        grid_illumination['P11'] = GRD.conv_grid2d(antpairs[key].location.x * (self.f[i]/FCNST.c),
                                                                               antpairs[key].location.y * (self.f[i]/FCNST.c),
                                                                               antpairs[key].wtspos['P11'][i][:,0],
                                                                               antpairs[key].wtspos['P11'][i][:,1],
                                                                               antpairs[key].wts['P11'][i],
                                                                               self.gridu,
                                                                               self.gridv,
                                                                               method=method)
                                        grid_illumination['P11'] = grid_illumination['P11'].reshape(self.gridu.shape)
                                        if normalize:
                                            grid_illumination['P11'] = grid_illumination['P11'] / NP.sum(grid_illumination['P11'])
                                        roi_ind = NP.where(NP.abs(grid_illumination['P11']) >= eps)
                                    elif antpairs[key].wtspos_scale['P11'] == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            grid_illumination['P11'] = GRD.conv_grid2d(antpairs[key].location.x * (self.f[0]/FCNST.c),
                                                                                    antpairs[key].location.y * (self.f[0]/FCNST.c),
                                                                                    antpairs[key].wtspos['P11'][0][:,0],
                                                                                    antpairs[key].wtspos['P11'][0][:,1],
                                                                                    antpairs[key].wts['P11'][0],
                                                                                    self.gridu,
                                                                                    self.gridv,
                                                                                    method=method)
                                            grid_illumination['P11'] = grid_illumination['P11'].reshape(self.gridu.shape)
                                            if normalize:
                                                grid_illumination['P11'] = grid_illumination['P11'] / NP.sum(grid_illumination['P11'])
                                            roi_ind = NP.where(NP.abs(grid_illumination['P11']) >= eps)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination['P11'][:,:,i] += grid_illumination['P11']
                                    self.grid_Vf['P11'][:,:,i] += antpairs[key].crosspol.Vf['P11'][i] * grid_illumination['P11']

                                if key in self.interferometers:
                                    if i not in self.interferometers[key]._gridinfo['P11']:
                                        self.interferometers[key]._gridinfo['P11'] = {} # Create an empty dictionary for each channel to hold grid info
                                    self.interferometers[key]._gridinfo['P11'][i]['f'] = self.f[i]
                                    self.interferometers[key]._gridinfo['P11'][i]['flag'] = False
                                    self.interferometers[key]._gridinfo['P11'][i]['gridxy_ind'] = zip(*roi_ind)
                                    self.interferometers[key].wtspos_scale['P11'] = antpairs[key].wtspos_scale['P11']
                                    if method == 'NN':
                                        self.interferometers[key]._gridinfo['P11'][i]['illumination'] = nnval
                                        self.interferometers[key]._gridinfo['P11'][i]['Vf'] = antpairs[key].crosspol.Vf['P11'][i] * nnval
                                    else:
                                        self.interferometers[key]._gridinfo['P11'][i]['illumination'] = grid_illumination['P11'][roi_ind]
                                        self.interferometers[key]._gridinfo['P11'][i]['Vf'] = antpairs[key].crosspol.Vf['P11'][i] * grid_illumination['P11'][roi_ind]

                elif isinstance(antpairs, list):
                    # Check if these interferometers are new or old and compatible
                    for key in range(len(antpairs)): 
                        if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                            if antpairs[key].label in self.interferometers:
                                if unconvolve_existing: # Effects on the grid of interferometers already existing must be removed 
                                    if self.interferometers[antpairs[key].label]._gridinfo['P11']: # if gridding info is not empty
                                        for i in range(len(self.f)):
                                            self.grid_unconvolve(antpairs[key].label)
                                else:
                                    raise KeyError('Interferometer {0} already found to exist in the dictionary of interferometers but cannot proceed grid_convolve() without unconvolving first.'.format(antpairs[key].label))
                            
                        else:
                            del antpairs[key] # remove the dictionary element since it is not an Interferometer instance

                    for key in range(len(antpairs)):
                        if not antpairs[key].crosspol.flag['P11']:
                            for i in range(len(self.f)):
                                if method == 'NN':
                                    if antpairs[key].wtspos_scale['P11'] is None: 
                                        reflocs = antpairs[key].wtspos['P11'][i] + (self.f[i]/FCNST.c) * NP.asarray([antpairs[key].location.x, antpairs[key].location.y]).reshape(1,-1)
                                        inplocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN(reflocs, antpairs[key].wts['P11'][i], inplocs,
                                                                      distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]

                                        roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif antpairs[key].wtspos_scale['P11'] == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            reflocs = antpairs[key].wtspos['P11'][0] + (self.f[0]/FCNST.c) * NP.asarray([antpairs[key].location.x, antpairs[key].location.y]).reshape(1,-1)
                                            inplocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                            ibind, nnval = LKP.lookup_1NN(reflocs, antpairs[key].wts['P11'][0], inplocs,
                                                                          distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                          remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]

                                            roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                            if normalize:
                                                nnval /= NP.sum(nnval)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination['P11'][roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                    self.grid_Vf['P11'][roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += antpairs[key].crosspol.Vf['P11'][i] * nnval
                                else:
                                    if antpairs[key].wtspos_scale['P11'] is None:
                                        grid_illumination['P11'] = GRD.conv_grid2d(antpairs[key].location.x * (self.f[i]/FCNST.c),
                                                                               antpairs[key].location.y * (self.f[i]/FCNST.c),
                                                                               antpairs[key].wtspos['P11'][i][:,0],
                                                                               antpairs[key].wtspos['P11'][i][:,1],
                                                                               antpairs[key].wts['P11'][i],
                                                                               self.gridu,
                                                                               self.gridv,
                                                                               method=method)
                                        grid_illumination['P11'] = grid_illumination['P11'].reshape(self.gridu.shape)
                                        if normalize:
                                            grid_illumination['P11'] = grid_illumination['P11'] / NP.sum(grid_illumination['P11'])
                                        roi_ind = NP.where(NP.abs(grid_illumination['P11']) >= eps)
                                    elif antpairs[key].wtspos_scale['P11'] == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            grid_illumination['P11'] = GRD.conv_grid2d(antpairs[key].location.x * (self.f[0]/FCNST.c),
                                                                                   antpairs[key].location.y * (self.f[0]/FCNST.c),
                                                                                   antpairs[key].wtspos['P11'][0][:,0],
                                                                                   antpairs[key].wtspos['P11'][0][:,1],
                                                                                   antpairs[key].wts['P11'][0],
                                                                                   self.gridu,
                                                                                   self.gridv,
                                                                                   method=method)
                                            grid_illumination['P11'] = grid_illumination['P11'].reshape(self.gridu.shape)
                                            if normalize:
                                                grid_illumination['P11'] = grid_illumination['P11'] / NP.sum(grid_illumination['P11'])
                                            roi_ind = NP.where(NP.abs(grid_illumination['P11']) >= eps)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination['P11'][:,:,i] += grid_illumination['P11']
                                    self.grid_Vf['P11'][:,:,i] += antpairs[key].crosspol.Vf['P11'][i] * grid_illumination['P11']

                                if antpairs[key].label in self.interferometers:
                                    if i not in self.interferometers[key]._gridinfo['P11']:
                                        self.interferometers[key]._gridinfo['P11'] = {} # Create an empty dictionary for each channel to hold grid info
                                    self.interferometers[antpairs[key].label]._gridinfo['P11'][i]['f'] = self.f[i]
                                    self.interferometers[antpairs[key].label]._gridinfo['P11'][i]['flag'] = False
                                    self.interferometers[antpairs[key].label]._gridinfo['P11'][i]['gridxy_ind'] = zip(*roi_ind)
                                    self.interferometers[key].wtspos_scale['P11'] = antpairs[key].wtspos_scale['P11']
                                    if method == 'NN':
                                        self.interferometers[antpairs[key].label]._gridinfo['P11'][i]['illumination'] = nnval
                                        self.interferometers[antpairs[key].label]._gridinfo['P11'][i]['Vf'] = antpairs[key].crosspol.Vf['P11'][i] * nnval
                                    else:
                                        self.interferometers[antpairs[key].label]._gridinfo['P11'][i]['illumination'] = grid_illumination['P11'][roi_ind]
                                        self.interferometers[antpairs[key].label]._gridinfo['P11'][i]['Vf'] = antpairs[key].crosspol.Vf['P11'][i] * grid_illumination['P11'][roi_ind] 
                else:
                    raise TypeError('antpairs must be an instance of InterferometerArray, a dictionary of Interferometer instances, a list of Interferometer instances or an Interferometer instance.')

            else:

                self.grid_illumination['P11'] = NP.zeros((self.gridu.shape[0],
                                                      self.gridu.shape[1],
                                                      len(self.f)),
                                                     dtype=NP.complex_)
                self.grid_Vf['P11'] = NP.zeros((self.gridu.shape[0],
                                            self.gridu.shape[1],
                                            len(self.f)), dtype=NP.complex_)

                for key in self.interferometers:
                    if not self.interferometers[key].crosspol.flag['P11']:
                        for i in range(len(self.f)):
                            if method == 'NN':
                                if self.interferometers[key].wtspos_scale['P11'] is None: 
                                    reflocs = self.interferometers[key].wtspos['P11'][i] + (self.f[i]/FCNST.c) * NP.asarray([self.interferometers[key].location.x, self.interferometers[key].location.y]).reshape(1,-1)
                                    inplocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                    ibind, nnval = LKP.lookup_1NN(reflocs, self.interferometers[key].wts['P11'][i], inplocs,
                                                                  distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                  remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]

                                    roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                    if normalize:
                                        nnval /= NP.sum(nnval)
                                elif self.interferometers[key].wtspos_scale['P11'] == 'scale':
                                    if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                        reflocs = self.interferometers[key].wtspos['P11'][0] + (self.f[0]/FCNST.c) * NP.asarray([self.interferometers[key].location.x, self.interferometers[key].location.y]).reshape(1,-1)
                                        inplocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN(reflocs, self.interferometers[key].wts['P11'][0], inplocs,
                                                                      distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                else:
                                    raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                self.grid_illumination['P11'][roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                self.grid_Vf['P11'][roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += self.interferometers[key].crosspol.Vf['P11'][i] * nnval
                            else:
                                if self.interferometers[key].wtspos_scale['P11'] is None:
                                    grid_illumination['P11'] = GRD.conv_grid2d(self.interferometers[key].location.x * (self.f[i]/FCNST.c),
                                                                           self.interferometers[key].location.y * (self.f[i]/FCNST.c),
                                                                           self.interferometers[key].wtspos['P11'][i][:,0],
                                                                           self.interferometers[key].wtspos['P11'][i][:,1],
                                                                           self.interferometers[key].wts['P11'][i],
                                                                           self.gridu,
                                                                           self.gridv,
                                                                           method=method)
                                    grid_illumination['P11'] = grid_illumination['P11'].reshape(self.gridu.shape)
                                    if normalize:
                                        grid_illumination['P11'] = grid_illumination['P11'] / NP.sum(grid_illumination['P11'])
                                    roi_ind = NP.where(NP.abs(grid_illumination['P11']) >= eps)
                                elif self.interferometers[key].wtspos_scale['P11'] == 'scale':
                                    if i == 0:
                                        grid_illumination['P11'] = GRD.conv_grid2d(self.interferometers[key].location.x * (self.f[0]/FCNST.c),
                                                                               self.interferometers[key].location.y * (self.f[0]/FCNST.c),
                                                                               self.interferometers[key].wtspos['P11'][0][:,0],
                                                                               self.interferometers[key].wtspos['P11'][0][:,1],
                                                                               self.interferometers[key].wts['P11'][0],
                                                                               self.gridu,
                                                                               self.gridv,
                                                                               method=method)
                                        grid_illumination['P11'] = grid_illumination['P11'].reshape(self.gridu.shape)
                                        if normalize:
                                            grid_illumination['P11'] = grid_illumination['P11'] / NP.sum(grid_illumination['P11'])
                                        roi_ind = NP.where(NP.abs(grid_illumination['P11']) >= eps)
                                else:
                                    raise ValueError('Invalid scale option specified. Aborting grid_convolve().')
                                
                                self.grid_illumination['P11'][:,:,i] += grid_illumination['P11']
                                self.grid_Vf['P11'][:,:,i] += self.interferometers[key].crosspol.Vf['P11'][i] * grid_illumination['P11']

                            self.interferometers[key]._gridinfo['P11'][i] = {} # Create a nested dictionary to hold channel info
                            self.interferometers[key]._gridinfo['P11'][i]['f'] = self.f[i]
                            self.interferometers[key]._gridinfo['P11'][i]['flag'] = False
                            self.interferometers[key]._gridinfo['P11'][i]['gridxy_ind'] = zip(*roi_ind)
                            if method == 'NN':
                                self.interferometers[key]._gridinfo['P11'][i]['illumination'] = nnval
                                self.interferometers[key]._gridinfo['P11'][i]['Vf'] = self.interferometers[key].crosspol.Vf['P11'][i] * nnval  
                            else:
                                self.interferometers[key]._gridinfo['P11'][i]['illumination'] = grid_illumination['P11'][roi_ind]
                                self.interferometers[key]._gridinfo['P11'][i]['Vf'] = self.interferometers[key].crosspol.Vf['P11'][i] * grid_illumination['P11'][roi_ind]

        if (pol is None) or (pol == 'P22'):

            if antpairs is not None:

                if isinstance(antpairs, Interferometer):
                    antpairs = [antpairs]

                if isinstance(antpairs, (dict, InterferometerArray)):
                    # Check if these interferometers are new or old and compatible
                    for key in antpairs: 
                        if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                            if key in self.interferometers:
                                if unconvolve_existing: # Effects on the grid of interferometers already existing must be removed 
                                    if self.interferometers[key]._gridinfo_P22: # if gridding info is not empty
                                        for i in range(len(self.f)):
                                            self.grid_unconvolve(antpairs[key].label)
                                else:
                                    raise KeyError('Interferometer {0} already found to exist in the dictionary of interferometers but cannot proceed grid_convolve() without unconvolving first.'.format(antpairs[key].label)) 
                            
                        else:
                            del antpairs[key] # remove the dictionary element since it is not an Interferometer instance

                    for key in antpairs:
                        if not antpairs[key].crosspol.flag_P22:
                            for i in range(len(self.f)):
                                if method == 'NN':
                                    if antpairs[key].wtspos_P22_scale is None: 
                                        reflocs = antpairs[key].wtspos_P22[i] + (self.f[i]/FCNST.c) * NP.asarray([antpairs[key].location.x, antpairs[key].location.y]).reshape(1,-1)
                                        inplocs = (self.f[i]/FCNST.c) * NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN(reflocs, antpairs[key].wts_P22[i], inplocs,
                                                                      distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif antpairs[key].wtspos_P22_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            reflocs = antpairs[key].wtspos_P22[0] + (self.f[0]/FCNST.c) * NP.asarray([antpairs[key].location.x, antpairs[key].location.y]).reshape(1,-1)
                                            inplocs = (self.f[0]/FCNST.c) * NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                            ibind, nnval = LKP.lookup_1NN(reflocs, antpairs[key].wts_P22[0], inplocs,
                                                                          distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                          remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                            roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                            if normalize:
                                                nnval /= NP.sum(nnval)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P22[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                    self.grid_Vf_P22[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += antpairs[key].crosspol.Vf_P22[i] * nnval
                                else:
                                    if antpairs[key].wtspos_P22_scale is None: 
                                        grid_illumination_P22 = GRD.conv_grid2d(antpairs[key].location.x * (self.f[i]/FCNST.c),
                                                                               antpairs[key].location.y * (self.f[i]/FCNST.c),
                                                                               antpairs[key].wtspos_P22[i][:,0],
                                                                               antpairs[key].wtspos_P22[i][:,1],
                                                                               antpairs[key].wts_P22[i],
                                                                               self.gridu * (self.f[i]/FCNST.c),
                                                                               self.gridv * (self.f[i]/FCNST.c),
                                                                               method=method)
                                        grid_illumination_P22 = grid_illumination_P22.reshape(self.gridu.shape)
                                        if normalize:
                                            grid_illumination_P22 = grid_illumination_P22 / NP.sum(grid_illumination_P22)
                                        roi_ind = NP.where(NP.abs(grid_illumination_P22) >= eps)
                                    elif antpairs[key].wtspos_P22_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            grid_illumination_P22 = GRD.conv_grid2d(antpairs[key].location.x * (self.f[0]/FCNST.c),
                                                                                   antpairs[key].location.y * (self.f[0]/FCNST.c),
                                                                                   antpairs[key].wtspos_P22[0][:,0],
                                                                                   antpairs[key].wtspos_P22[0][:,1],
                                                                                   antpairs[key].wts_P22[0],
                                                                                   self.gridu * (self.f[0]/FCNST.c),
                                                                                   self.gridv * (self.f[0]/FCNST.c),
                                                                                   method=method)
                                            grid_illumination_P22 = grid_illumination_P22.reshape(self.gridu.shape)
                                            if normalize:
                                                grid_illumination_P22 = grid_illumination_P22 / NP.sum(grid_illumination_P22)
                                            roi_ind = NP.where(NP.abs(grid_illumination_P22) >= eps)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P22[:,:,i] += grid_illumination_P22
                                    self.grid_Vf_P22[:,:,i] += antpairs[key].crosspol.Vf_P22[i] * grid_illumination_P22

                                if key in self.interferometers:
                                    if i not in self.interferometers[key]._gridinfo_P22:
                                        self.interferometers[key]._gridinfo_P22 = {} # Create an empty dictionary for each channel to hold grid info
                                    self.interferometers[key]._gridinfo_P22[i]['f'] = self.f[i]
                                    self.interferometers[key]._gridinfo_P22[i]['flag'] = False
                                    self.interferometers[key]._gridinfo_P22[i]['gridxy_ind'] = zip(*roi_ind)
                                    self.interferometers[key].wtspos_P22_scale = antpairs[key].wtspos_P22_scale
                                    if method == 'NN':
                                        self.interferometers[key]._gridinfo_P22[i]['illumination'] = nnval
                                        self.interferometers[key]._gridinfo_P22[i]['Vf'] = antpairs[key].crosspol.Vf_P22[i] * nnval
                                    else:
                                        self.interferometers[key]._gridinfo_P22[i]['illumination'] = grid_illumination_P22[roi_ind]
                                        self.interferometers[key]._gridinfo_P22[i]['Vf'] = antpairs[key].crosspol.Vf_P22[i] * grid_illumination_P22[roi_ind]

                elif isinstance(antpairs, list):
                    # Check if these interferometers are new or old and compatible
                    for key in range(len(antpairs)): 
                        if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                            if antpairs[key].label in self.interferometers:
                                if unconvolve_existing: # Effects on the grid of interferometers already existing must be removed 
                                    if self.interferometers[antpairs[key].label]._gridinfo_P22: # if gridding info is not empty
                                        for i in range(len(self.f)):
                                            self.grid_unconvolve(antpairs[key].label)
                                else:
                                    raise KeyError('Interferometer {0} already found to exist in the dictionary of interferometers but cannot proceed grid_convolve() without unconvolving first.'.format(antpairs[key].label))
                            
                        else:
                            del antpairs[key] # remove the dictionary element since it is not an Interferometer instance

                    for key in range(len(antpairs)):
                        if not antpairs[key].crosspol.flag_P22:
                            for i in range(len(self.f)):
                                if method == 'NN':
                                    if antpairs[key].wtspos_P22_scale is None: 
                                        reflocs = antpairs[key].wtspos_P22[i] + (self.f[i]/FCNST.c) * NP.asarray([antpairs[key].location.x, antpairs[key].location.y]).reshape(1,-1)
                                        inplocs = (self.f[i]/FCNST.c) * NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN(reflocs, antpairs[key].wts_P22[i], inplocs,
                                                                      distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]

                                        roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                    elif antpairs[key].wtspos_P22_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            reflocs = antpairs[key].wtspos_P22[0] + (self.f[0]/FCNST.c) * NP.asarray([antpairs[key].location.x, antpairs[key].location.y]).reshape(1,-1)
                                            inplocs = (self.f[0]/FCNST.c) * NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                            ibind, nnval = LKP.lookup_1NN(reflocs, antpairs[key].wts_P22[0], inplocs,
                                                                          distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                          remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]

                                            roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                            if normalize:
                                                nnval /= NP.sum(nnval)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P22[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                    self.grid_Vf_P22[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += antpairs[key].crosspol.Vf_P22[i] * nnval
                                else:
                                    if antpairs[key].wtspos_P22_scale is None:
                                        grid_illumination_P22 = GRD.conv_grid2d(antpairs[key].location.x * (self.f[i]/FCNST.c),
                                                                               antpairs[key].location.y * (self.f[i]/FCNST.c),
                                                                               antpairs[key].wtspos_P22[i][:,0],
                                                                               antpairs[key].wtspos_P22[i][:,1],
                                                                               antpairs[key].wts_P22[i],
                                                                               self.gridu * (self.f[i]/FCNST.c),
                                                                               self.gridv * (self.f[i]/FCNST.c),
                                                                               method=method)
                                        grid_illumination_P22 = grid_illumination_P22.reshape(self.gridu.shape)
                                        if normalize:
                                            grid_illumination_P22 = grid_illumination_P22 / NP.sum(grid_illumination_P22)
                                        roi_ind = NP.where(NP.abs(grid_illumination_P22) >= eps)
                                    elif antpairs[key].wtspos_P22_scale == 'scale':
                                        if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                            grid_illumination_P22 = GRD.conv_grid2d(antpairs[key].location.x * (self.f[0]/FCNST.c),
                                                                                   antpairs[key].location.y * (self.f[0]/FCNST.c),
                                                                                   antpairs[key].wtspos_P22[0][:,0],
                                                                                   antpairs[key].wtspos_P22[0][:,1],
                                                                                   antpairs[key].wts_P22[0],
                                                                                   self.gridu * (self.f[0]/FCNST.c),
                                                                                   self.gridv * (self.f[0]/FCNST.c),
                                                                                   method=method)
                                            grid_illumination_P22 = grid_illumination_P22.reshape(self.gridu.shape)
                                            if normalize:
                                                grid_illumination_P22 = grid_illumination_P22 / NP.sum(grid_illumination_P22)
                                            roi_ind = NP.where(NP.abs(grid_illumination_P22) >= eps)
                                    else:
                                        raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                    self.grid_illumination_P22[:,:,i] += grid_illumination_P22
                                    self.grid_Vf_P22[:,:,i] += antpairs[key].crosspol.Vf_P22[i] * grid_illumination_P22

                                if antpairs[key].label in self.interferometers:
                                    if i not in self.interferometers[key]._gridinfo_P22:
                                        self.interferometers[key]._gridinfo_P22 = {} # Create an empty dictionary for each channel to hold grid info
                                    self.interferometers[antpairs[key].label]._gridinfo_P22[i]['f'] = self.f[i]
                                    self.interferometers[antpairs[key].label]._gridinfo_P22[i]['flag'] = False
                                    self.interferometers[antpairs[key].label]._gridinfo_P22[i]['gridxy_ind'] = zip(*roi_ind)
                                    self.interferometers[key].wtspos_P22_scale = antpairs[key].wtspos_P22_scale
                                    if method == 'NN':
                                        self.interferometers[antpairs[key].label]._gridinfo_P22[i]['illumination'] = nnval
                                        self.interferometers[antpairs[key].label]._gridinfo_P22[i]['Vf'] = antpairs[key].crosspol.Vf_P22[i] * nnval
                                    else:
                                        self.interferometers[antpairs[key].label]._gridinfo_P22[i]['illumination'] = grid_illumination_P22[roi_ind]
                                        self.interferometers[antpairs[key].label]._gridinfo_P22[i]['Vf'] = antpairs[key].crosspol.Vf_P22[i] * grid_illumination_P22[roi_ind] 
                else:
                    raise TypeError('antpairs must be an instance of InterferometerArray, a dictionary of Interferometer instances, a list of Interferometer instances or an Interferometer instance.')

            else:

                self.grid_illumination_P22 = NP.zeros((self.gridu.shape[0],
                                                      self.gridu.shape[1],
                                                      len(self.f)),
                                                     dtype=NP.complex_)
                self.grid_Vf_P22 = NP.zeros((self.gridu.shape[0],
                                            self.gridu.shape[1],
                                            len(self.f)), dtype=NP.complex_)

                for key in self.interferometers:
                    if not self.interferometers[key].crosspol.flag_P22:
                        for i in range(len(self.f)):
                            if method == 'NN':
                                if self.interferometers[key].wtspos_P22_scale is None: 
                                    reflocs = self.interferometers[key].wtspos_P22[i] + (self.f[i]/FCNST.c) * NP.asarray([self.interferometers[key].location.x, self.interferometers[key].location.y]).reshape(1,-1)
                                    inplocs = (self.f[i]/FCNST.c) * NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                    ibind, nnval = LKP.lookup_1NN(reflocs, self.interferometers[key].wts_P22[i], inplocs,
                                                                  distance_ULIM=distNN*self.f[i]/FCNST.c,
                                                                  remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]

                                    roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                    if normalize:
                                        nnval /= NP.sum(nnval)
                                elif self.interferometers[key].wtspos_P22_scale == 'scale':
                                    if i == 0: # Determine some parameters only for zeroth channel if scaling is set
                                        reflocs = self.interferometers[key].wtspos_P22[0] + (self.f[0]/FCNST.c) * NP.asarray([self.interferometers[key].location.x, self.interferometers[key].location.y]).reshape(1,-1)
                                        inplocs = (self.f[0]/FCNST.c) * NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                                        ibind, nnval = LKP.lookup_1NN(reflocs, self.interferometers[key].wts_P22[0], inplocs,
                                                                      distance_ULIM=distNN*self.f[0]/FCNST.c,
                                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                                        roi_ind = NP.unravel_index(ibind, self.gridu.shape)
                                        if normalize:
                                            nnval /= NP.sum(nnval)
                                else:
                                    raise ValueError('Invalid scale option specified. Aborting grid_convolve().')

                                self.grid_illumination_P22[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += nnval
                                self.grid_Vf_P22[roi_ind+(i+NP.zeros(ibind.size, dtype=NP.int),)] += self.interferometers[key].crosspol.Vf_P22[i] * nnval
                            else:
                                if self.interferometers[key].wtspos_P22_scale is None:
                                    grid_illumination_P22 = GRD.conv_grid2d(self.interferometers[key].location.x * (self.f[i]/FCNST.c),
                                                                           self.interferometers[key].location.y * (self.f[i]/FCNST.c),
                                                                           self.interferometers[key].wtspos_P22[i][:,0],
                                                                           self.interferometers[key].wtspos_P22[i][:,1],
                                                                           self.interferometers[key].wts_P22[i],
                                                                           self.gridu * (self.f[i]/FCNST.c),
                                                                           self.gridv * (self.f[i]/FCNST.c),
                                                                           method=method)
                                    grid_illumination_P22 = grid_illumination_P22.reshape(self.gridu.shape)
                                    if normalize:
                                        grid_illumination_P22 = grid_illumination_P22 / NP.sum(grid_illumination_P22)
                                    roi_ind = NP.where(NP.abs(grid_illumination_P22) >= eps)
                                elif self.interferometers[key].wtspos_P22_scale == 'scale':
                                    if i == 0:
                                        grid_illumination_P22 = GRD.conv_grid2d(self.interferometers[key].location.x * (self.f[0]/FCNST.c),
                                                                               self.interferometers[key].location.y * (self.f[0]/FCNST.c),
                                                                               self.interferometers[key].wtspos_P22[0][:,0],
                                                                               self.interferometers[key].wtspos_P22[0][:,1],
                                                                               self.interferometers[key].wts_P22[0],
                                                                               self.gridu * (self.f[0]/FCNST.c),
                                                                               self.gridv * (self.f[0]/FCNST.c),
                                                                               method=method)
                                        grid_illumination_P22 = grid_illumination_P22.reshape(self.gridu.shape)
                                        if normalize:
                                            grid_illumination_P22 = grid_illumination_P22 / NP.sum(grid_illumination_P22)
                                        roi_ind = NP.where(NP.abs(grid_illumination_P22) >= eps)
                                else:
                                    raise ValueError('Invalid scale option specified. Aborting grid_convolve().')
                                
                                self.grid_illumination_P22[:,:,i] += grid_illumination_P22
                                self.grid_Vf_P22[:,:,i] += self.interferometers[key].crosspol.Vf_P22[i] * grid_illumination_P22

                            self.interferometers[key]._gridinfo_P22[i] = {} # Create a nested dictionary to hold channel info
                            self.interferometers[key]._gridinfo_P22[i]['f'] = self.f[i]
                            self.interferometers[key]._gridinfo_P22[i]['flag'] = False
                            self.interferometers[key]._gridinfo_P22[i]['gridxy_ind'] = zip(*roi_ind)
                            if method == 'NN':
                                self.interferometers[key]._gridinfo_P22[i]['illumination'] = nnval
                                self.interferometers[key]._gridinfo_P22[i]['Vf'] = self.interferometers[key].crosspol.Vf_P22[i] * nnval  
                            else:
                                self.interferometers[key]._gridinfo_P22[i]['illumination'] = grid_illumination_P22[roi_ind]
                                self.interferometers[key]._gridinfo_P22[i]['Vf'] = self.interferometers[key].crosspol.Vf_P22[i] * grid_illumination_P22[roi_ind]

    ################################################################################

    def grid_unconvolve(self, antpairs, pol=None):

        """
        ----------------------------------------------------------------------------
        [Needs to be re-written]

        Routine to de-project the visibility illumination pattern and the
        visibilities on the grid. It can operate on the entire interferometer array 
        or incrementally de-project the visibilities and illumination patterns of
        specific antenna pairs from an already existing grid.

        Inputs:

        antpairs    [instance of class InterferometerArray, single instance or 
                    list of instances of class Interferometer, or a dictionary 
                    holding instances of class Interferometer] If a dictionary is 
                    provided, the keys should be the interferometer labels and 
                    the values should be instances of class Interferometer. If a 
                    list is provided, it should be a list of valid instances of 
                    class Interferometer. These instance(s) of class 
                    Interferometer will be merged to the existing grid contained 
                    in the instance of InterferometerArray class. If any of the 
                    interferoemters are not found to be in the already existing 
                    set of interferometers, an exception is raised accordingly
                    and code execution stops.

        pol         [String] The polarization to be gridded. Can be set to 'P11', 
                    'P12', 'P21', or 'P22'. If set to None, gridding for all
                    polarizations is performed. Default = None

        ----------------------------------------------------------------------------
        """

        try:
            antpairs
        except NameError:
            raise NameError('No antenna pair(s) supplied.')

        if (pol is None) or (pol == 'P11'):

            if isinstance(ants, (Interferometer, str)):
                antpairs = [antpairs]

            if isinstance(antpairs, (dict, InterferometerArray)):
                # Check if these interferometers are new or old and compatible
                for key in antpairs: 
                    if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                        if key in self.interferometers:
                            if self.interferometers[key]._gridinfo_P11: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[key]._gridinfo_P11[i]['gridxy_ind'])
                                    self.grid_illumination_P11[xind, yind, i] -= self.interferometers[key]._gridinfo_P11[i]['illumination']
                                    self.grid_Vf_P11[xind, yind, i] -= self.interferometers[key]._gridinfo_P11[i]['Vf']
                                self.interferometers[key]._gridinfo_P11 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key].label))
                                
            elif isinstance(antpairs, list):
                # Check if these interferometers are new or old and compatible
                for key in range(len(antpairs)): 
                    if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                        if antpairs[key].label in self.interferometers:
                            if self.interferometers[antpairs[key].label]._gridinfo_P11: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[antpairs[key].label]._gridinfo_P11[i]['gridxy_ind'])
                                    self.grid_illumination_P11[xind, yind, i] -= self.interferometers[antpairs[key].label]._gridinfo_P11[i]['illumination']
                                    self.grid_Vf_P11[xind, yind, i] -= self.interferometers[antpairs[key].label]._gridinfo_P11[i]['Vf']
                                self.interferometers[antpairs[key].label]._gridinfo_P11 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key].label))
                    elif isinstance(antpairs[key], str):
                        if antpairs[key] in self.interferometers:
                            if self.interferometers[antpairs[key]]._gridinfo_P11: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[antpairs[key]]._gridinfo_P11[i]['gridxy_ind'])
                                    self.grid_illumination_P11[xind, yind, i] -= self.interferometers[antpairs[key]]._gridinfo_P11[i]['illumination']
                                    self.grid_Vf_P11[xind, yind, i] -= self.interferometers[antpairs[key]]._gridinfo_P11[i]['Vf']
                                self.interferometers[antpairs[key]]._gridinfo_P11 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key]))
                    else:
                        raise TypeError('antpairs must be an instance of class InterferometerArray, a list of instances of class Interferometer, a dictionary of instances of class Interferometer or a list of antenna labels.')
            else:
                raise TypeError('antpairs must be an instance of InterferometerArray, a dictionary of Interferometer instances, a list of Interferometer instances, an Interferometer instance, or a list of antenna labels.')

        if (pol is None) or (pol == 'P22'):

            if isinstance(ants, (Interferometer, str)):
                antpairs = [antpairs]

            if isinstance(antpairs, (dict, InterferometerArray)):
                # Check if these interferometers are new or old and compatible
                for key in antpairs: 
                    if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                        if key in self.interferometers:
                            if self.interferometers[key]._gridinfo_P22: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[key]._gridinfo_P22[i]['gridxy_ind'])
                                    self.grid_illumination_P22[xind, yind, i] -= self.interferometers[key]._gridinfo_P22[i]['illumination']
                                    self.grid_Vf_P22[xind, yind, i] -= self.interferometers[key]._gridinfo_P22[i]['Vf']
                                self.interferometers[key]._gridinfo_P22 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key].label))
                                
            elif isinstance(antpairs, list):
                # Check if these interferometers are new or old and compatible
                for key in range(len(antpairs)): 
                    if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                        if antpairs[key].label in self.interferometers:
                            if self.interferometers[antpairs[key].label]._gridinfo_P22: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[antpairs[key].label]._gridinfo_P22[i]['gridxy_ind'])
                                    self.grid_illumination_P22[xind, yind, i] -= self.interferometers[antpairs[key].label]._gridinfo_P22[i]['illumination']
                                    self.grid_Vf_P22[xind, yind, i] -= self.interferometers[antpairs[key].label]._gridinfo_P22[i]['Vf']
                                self.interferometers[antpairs[key].label]._gridinfo_P22 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key].label))
                    elif isinstance(antpairs[key], str):
                        if antpairs[key] in self.interferometers:
                            if self.interferometers[antpairs[key]]._gridinfo_P22: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[antpairs[key]]._gridinfo_P22[i]['gridxy_ind'])
                                    self.grid_illumination_P22[xind, yind, i] -= self.interferometers[antpairs[key]]._gridinfo_P22[i]['illumination']
                                    self.grid_Vf_P22[xind, yind, i] -= self.interferometers[antpairs[key]]._gridinfo_P22[i]['Vf']
                                self.interferometers[antpairs[key]]._gridinfo_P22 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key]))
                    else:
                        raise TypeError('antpairs must be an instance of class InterferometerArray, a list of instances of class Interferometer, a dictionary of instances of class Interferometer or a list of antenna labels.')
            else:
                raise TypeError('antpairs must be an instance of InterferometerArray, a dictionary of Interferometer instances, a list of Interferometer instances, an Interferometer instance, or a list of antenna labels.')

        if (pol is None) or (pol == 'P12'):

            if isinstance(ants, (Interferometer, str)):
                antpairs = [antpairs]

            if isinstance(antpairs, (dict, InterferometerArray)):
                # Check if these interferometers are new or old and compatible
                for key in antpairs: 
                    if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                        if key in self.interferometers:
                            if self.interferometers[key]._gridinfo_P12: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[key]._gridinfo_P12[i]['gridxy_ind'])
                                    self.grid_illumination_P12[xind, yind, i] -= self.interferometers[key]._gridinfo_P12[i]['illumination']
                                    self.grid_Vf_P12[xind, yind, i] -= self.interferometers[key]._gridinfo_P12[i]['Vf']
                                self.interferometers[key]._gridinfo_P12 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key].label))
                                
            elif isinstance(antpairs, list):
                # Check if these interferometers are new or old and compatible
                for key in range(len(antpairs)): 
                    if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                        if antpairs[key].label in self.interferometers:
                            if self.interferometers[antpairs[key].label]._gridinfo_P12: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[antpairs[key].label]._gridinfo_P12[i]['gridxy_ind'])
                                    self.grid_illumination_P12[xind, yind, i] -= self.interferometers[antpairs[key].label]._gridinfo_P12[i]['illumination']
                                    self.grid_Vf_P12[xind, yind, i] -= self.interferometers[antpairs[key].label]._gridinfo_P12[i]['Vf']
                                self.interferometers[antpairs[key].label]._gridinfo_P12 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key].label))
                    elif isinstance(antpairs[key], str):
                        if antpairs[key] in self.interferometers:
                            if self.interferometers[antpairs[key]]._gridinfo_P12: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[antpairs[key]]._gridinfo_P12[i]['gridxy_ind'])
                                    self.grid_illumination_P12[xind, yind, i] -= self.interferometers[antpairs[key]]._gridinfo_P12[i]['illumination']
                                    self.grid_Vf_P12[xind, yind, i] -= self.interferometers[antpairs[key]]._gridinfo_P12[i]['Vf']
                                self.interferometers[antpairs[key]]._gridinfo_P12 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key]))
                    else:
                        raise TypeError('antpairs must be an instance of class InterferometerArray, a list of instances of class Interferometer, a dictionary of instances of class Interferometer or a list of antenna labels.')
            else:
                raise TypeError('antpairs must be an instance of InterferometerArray, a dictionary of Interferometer instances, a list of Interferometer instances, an Interferometer instance, or a list of antenna labels.')

        if (pol is None) or (pol == 'P21'):

            if isinstance(ants, (Interferometer, str)):
                antpairs = [antpairs]

            if isinstance(antpairs, (dict, InterferometerArray)):
                # Check if these interferometers are new or old and compatible
                for key in antpairs: 
                    if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                        if key in self.interferometers:
                            if self.interferometers[key]._gridinfo_P21: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[key]._gridinfo_P21[i]['gridxy_ind'])
                                    self.grid_illumination_P21[xind, yind, i] -= self.interferometers[key]._gridinfo_P21[i]['illumination']
                                    self.grid_Vf_P21[xind, yind, i] -= self.interferometers[key]._gridinfo_P21[i]['Vf']
                                self.interferometers[key]._gridinfo_P21 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key].label))
                                
            elif isinstance(antpairs, list):
                # Check if these interferometers are new or old and compatible
                for key in range(len(antpairs)): 
                    if isinstance(antpairs[key], Interferometer): # required if antpairs is a dictionary and not instance of InterferometerArray
                        if antpairs[key].label in self.interferometers:
                            if self.interferometers[antpairs[key].label]._gridinfo_P21: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[antpairs[key].label]._gridinfo_P21[i]['gridxy_ind'])
                                    self.grid_illumination_P21[xind, yind, i] -= self.interferometers[antpairs[key].label]._gridinfo_P21[i]['illumination']
                                    self.grid_Vf_P21[xind, yind, i] -= self.interferometers[antpairs[key].label]._gridinfo_P21[i]['Vf']
                                self.interferometers[antpairs[key].label]._gridinfo_P21 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key].label))
                    elif isinstance(antpairs[key], str):
                        if antpairs[key] in self.interferometers:
                            if self.interferometers[antpairs[key]]._gridinfo_P21: # if gridding info is not empty
                                for i in range(len(self.f)):
                                    xind, yind = zip(*self.interferometers[antpairs[key]]._gridinfo_P21[i]['gridxy_ind'])
                                    self.grid_illumination_P21[xind, yind, i] -= self.interferometers[antpairs[key]]._gridinfo_P21[i]['illumination']
                                    self.grid_Vf_P21[xind, yind, i] -= self.interferometers[antpairs[key]]._gridinfo_P21[i]['Vf']
                                self.interferometers[antpairs[key]]._gridinfo_P21 = {}
                        else:
                            raise KeyError('Interferometer {0} not found to exist in the dictionary of interferometers.'.format(antpairs[key]))
                    else:
                        raise TypeError('antpairs must be an instance of class InterferometerArray, a list of instances of class Interferometer, a dictionary of instances of class Interferometer or a list of antenna labels.')
            else:
                raise TypeError('antpairs must be an instance of InterferometerArray, a dictionary of Interferometer instances, a list of Interferometer instances, an Interferometer instance, or a list of antenna labels.')

    ##################################################################################

    def update_flags(self, dictflags=None):

        """
        ----------------------------------------------------------------------------
        Updates all flags in the interferometer array followed by any flags that
        need overriding through inputs of specific flag information

        Inputs:

        dictflags  [dictionary] contains flag information overriding after default
                   flag updates are determined. Baseline based flags are given as 
                   further dictionaries with each under under a key which is the
                   same as the interferometer label. Flags for each baseline are
                   specified as a dictionary holding boolean flags for each of the 
                   four cross-polarizations which are stored under keys 'P11', 
                   'P12', 'P21', and 'P22'. An absent key just means it is not a
                   part of the update. Flag information under each baseline must be 
                   of same type as input parameter flags in member function 
                   update_flags() of class CrossPolInfo
        ----------------------------------------------------------------------------
        """

        for label in self.interferometers:
            self.interferometers[label].update_flags()

        if dictflags is not None:
            if not isinstance(dictflags, dict):
                raise TypeError('Input parameter dictflags must be a dictionary')
            
            for label in dictflags:
                if label in self.interferometers:
                    self.interferometers[label].crosspol.update_flags(flags=dictflags[label])

    ##################################################################################

    def update(self, antenna_level_updates=None, interferometer_level_updates=None,
               do_correlate=None, parallel=False, nproc=None, verbose=False):

        """
        -------------------------------------------------------------------------
        Updates the interferometer array instance with newer attribute values.
        Can also be used to add and/or remove interferometers with/without
        affecting the existing grid.

        Inputs:

        antenna_level_updates
                    [Dictionary] Provides update information on individual
                    antennas and antenna array as a whole. Should be of same 
                    type as input parameter updates in member function update() 
                    of class AntennaArray. It consists of information updates 
                    under the following principal keys:
                    'antenna_array': Consists of updates for the AntennaArray
                                instance. This is a dictionary which consists of
                                the following keys:
                                'timestamp'   Unique identifier of the time 
                                              series. It is optional to set this 
                                              to a scalar. If not given, no 
                                              change is made to the existing
                                              timestamp attribute
                                'do_grid'     [boolean] If set to True, create or
                                              recreate a grid. To be specified 
                                              when the antenna locations are
                                              updated.
                    'antennas': Holds a list of dictionaries consisting of 
                                updates for individual antennas. Each element 
                                in the list contains update for one antenna. 
                                For each of these dictionaries, one of the keys 
                                is 'label' which indicates an antenna label. If 
                                absent, the code execution stops by throwing an 
                                exception. The other optional keys and the 
                                information they hold are listed below:
                                'action'      [String scalar] Indicates the type 
                                              of update operation. 'add' adds the 
                                              Antenna instance to the 
                                              AntennaArray instance. 'remove' 
                                              removes the antenna from the
                                              antenna array instance. 'modify'
                                              modifies the antenna attributes in 
                                              the antenna array instance. This 
                                              key has to be set. No default.
                                'grid_action' [Boolean] If set to True, will 
                                              apply the grdding operations 
                                              (grid(), grid_convolve(), and 
                                              grid_unconvolve()) appropriately 
                                              according to the value of the 
                                              'action' key. If set to None or 
                                              False, gridding effects will remain
                                              unchanged. Default=None(=False).
                                'antenna'     [instance of class Antenna] Updated 
                                              Antenna class instance. Can work 
                                              for action key 'remove' even if not 
                                              set (=None) or set to an empty 
                                              string '' as long as 'label' key is 
                                              specified. 
                                'gridpol'     [Optional. String scalar] Initiates 
                                              the specified action on 
                                              polarization 'P1' or 'P2'. Can be 
                                              set to 'P1' or 'P2'. If not 
                                              provided (=None), then the 
                                              specified action applies to both
                                              polarizations. Default = None.
                                'Et_P1'       [Optional. Numpy array] Complex 
                                              Electric field time series in 
                                              polarization P1. Is used only if 
                                              set and if 'action' key value is 
                                              set to 'modify'. Default = None.
                                'Et_P2'       [Optional. Numpy array] Complex 
                                              Electric field time series in 
                                              polarization P2. Is used only if 
                                              set and if 'action' key value is 
                                              set to 'modify'. Default = None.
                                't'           [Optional. Numpy array] Time axis 
                                              of the time series. Is used only 
                                              if set and if 'action' key value is
                                              set to 'modify'. Default = None.
                                'timestamp'   [Optional. Scalar] Unique 
                                              identifier of the time series. Is 
                                              used only if set and if 'action' 
                                              key value is set to 'modify'.
                                              Default = None.
                                'location'    [Optional. instance of GEOM.Point
                                              class] 
                                              Antenna location in the local ENU 
                                              coordinate system. Used only if 
                                              set and if 'action' key value is 
                                              set to 'modify'. Default = None.
                                'wtsinfo_P1'  [Optional. List of dictionaries] 
                                              See description in Antenna class 
                                              member function update(). Is used 
                                              only if set and if 'action' key 
                                              value is set to 'modify'.
                                              Default = None.
                                'wtsinfo_P2'  [Optional. List of dictionaries] 
                                              See description in Antenna class 
                                              member function update(). Is used 
                                              only if set and if 'action' key 
                                              value is set to 'modify'.
                                              Default = None.
                                'flag_P1'     [Optional. Boolean] Flagging status 
                                              update for polarization P1 of the 
                                              antenna. If True, polarization P1 
                                              measurements of the antenna will be
                                              flagged. If not set (=None), the 
                                              previous or default flag status 
                                              will continue to apply. If set to 
                                              False, the antenna status will be
                                              updated to become unflagged.
                                              Default = None.
                                'flag_P2'     [Optional. Boolean] Flagging status 
                                              update for polarization P2 of the 
                                              antenna. If True, polarization P2 
                                              measurements of the antenna will be
                                              flagged. If not set (=None), the 
                                              previous or default flag status 
                                              will continue to apply. If set to 
                                              False, the antenna status will be
                                              updated to become unflagged.
                                              Default = None.
                                'gridfunc_freq'
                                              [Optional. String scalar] Read the 
                                              description of inputs to Antenna 
                                              class member function update(). If 
                                              set to None (not provided), this
                                              attribute is determined based on 
                                              the size of wtspos_P1 and wtspos_P2. 
                                              It is applicable only when 'action' 
                                              key is set to 'modify'. 
                                              Default = None.
                                'delaydict_P1'
                                              Dictionary containing information 
                                              on delay compensation to be applied 
                                              to the fourier transformed electric 
                                              fields for polarization P1. Default
                                              is None (no delay compensation to 
                                              be applied). Refer to the docstring 
                                              of member function
                                              delay_compensation() of class 
                                              PolInfo for more details.
                                'delaydict_P2'
                                              Dictionary containing information 
                                              on delay compensation to be applied 
                                              to the fourier transformed electric 
                                              fields for polarization P2. Default
                                              is None (no delay compensation to 
                                              be applied). Refer to the docstring 
                                              of member function
                                              delay_compensation() of class 
                                              PolInfo for more details.
                                'ref_freq'    [Optional. Scalar] Positive value 
                                              (in Hz) of reference frequency 
                                              (used if gridfunc_freq is set to
                                              'scale') at which wtspos_P1 and 
                                              wtspos_P2 in wtsinfo_P1 and 
                                              wtsinfo_P2, respectively, are 
                                              provided. If set to None, the 
                                              reference frequency already set in
                                              antenna array instance remains
                                              unchanged. Default = None.
                                'pol_type'    [Optional. String scalar] 'Linear' 
                                              or 'Circular'. Used only when 
                                              action key is set to 'modify'. If 
                                              not provided, then the previous
                                              value remains in effect.
                                              Default = None.
                                'norm_wts'    [Optional. Boolean] Default=False. 
                                              If set to True, the gridded weights 
                                              are divided by the sum of weights 
                                              so that the gridded weights add up 
                                              to unity. This is used only when
                                              grid_action keyword is set when
                                              action keyword is set to 'add' or
                                              'modify'
                                'gridmethod'  [Optional. String] Indicates 
                                              gridding method. It accepts the 
                                              following values 'NN' (nearest 
                                              neighbour), 'BL' (Bi-linear
                                              interpolation), and'CS' (Cubic
                                              Spline interpolation). Default='NN'
                                'distNN'      [Optional. Scalar] Indicates the 
                                              upper bound on distance for a 
                                              nearest neighbour search if the 
                                              value of 'gridmethod' is set to
                                              'NN'. The units are of physical
                                              distance, the same as what is 
                                              used for antenna locations.
                                              Default = NP.inf
                                'maxmatch'    [scalar] A positive value 
                                              indicating maximum number of input
                                              locations in the antenna grid to 
                                              be assigned. Default = None. If 
                                              set to None, all the antenna array 
                                              grid elements specified are 
                                              assigned values for each antenna.
                                              For instance, to have only one
                                              antenna array grid element to be
                                              populated per antenna, use
                                              maxmatch=1. 
                                'tol'         [scalar] If set, only lookup data 
                                              with abs(val) > tol will be
                                              considered for nearest neighbour 
                                              lookup. Default = None implies 
                                              all lookup values will be 
                                              considered for nearest neighbour
                                              determination. tol is to be
                                              interpreted as a minimum value
                                              considered as significant in the
                                              lookup table. 

        interferometer_level_updates
                    [Dictionary] Consists of information updates for individual
                    interferoemters and interferometer array as a whole under the
                    following principal keys:
                    'interferometer_array': Consists of updates for the
                                InterferometerArray instance. This is a
                                dictionary which consists of the following keys:
                                'timestamp': Unique identifier of the time
                                       series. It is optional to set this to a
                                       scalar. If not given, no change is made to 
                                       the existing timestamp attribute
                    'interferometers': Holds a list of dictionaries where element
                                consists of updates for individual 
                                interferometers. Each dictionary must contain a 
                                key 'label' which indicates an interferometer 
                                label. If absent, the code execution stops by 
                                throwing an exception. The other optional keys 
                                and the information they hold are listed below:
                                'action'      [String scalar] Indicates the type 
                                              of update operation. 'add' adds the 
                                              Interferometer instance to the 
                                              InterferometerArray instance. 
                                              'remove' removes the interferometer 
                                              from the interferometer array
                                              instance. 'modify' modifies the
                                              interferometer attributes in the 
                                              interferometer array instance. This 
                                              key has to be set. No default.
                                'grid_action' [Boolean] If set to True, will 
                                              apply the grdding operations 
                                              (grid(), grid_convolve(), and 
                                              grid_unconvolve()) appropriately 
                                              according to the value of the 
                                              'action' key. If set to None or 
                                              False, gridding effects will remain
                                              unchanged. Default=None(=False).
                                'interferometer' 
                                              [instance of class Interferometer] 
                                              Updated Interferometer class 
                                              instance. Can work for action key
                                              'remove' even if not set (=None) or
                                              set to an empty string '' as long as 
                                              'label' key is specified. 
                                'gridpol'     [Optional. String scalar] Initiates 
                                              the specified action on 
                                              polarization 'P11' or 'P22'. Can be 
                                              set to 'P11' or 'P22'. If not 
                                              provided (=None), then the 
                                              specified action applies to both
                                              polarizations. Default = None.
                                'Vt'          [Optional. Dictionary] Complex 
                                              visibility time series for each 
                                              of the four cross-polarization 
                                              specified as keys 'P11', 'P12', 
                                              'P21' and 'P22'. Is used only if 
                                              set and if 'action' key value is 
                                              set to 'modify'. Default = None.
                                't'           [Optional. Numpy array] Time axis 
                                              of the time series. Is used only 
                                              if set and if 'action' key value is
                                              set to 'modify'. Default = None.
                                'timestamp'   [Optional. Scalar] Unique 
                                              identifier of the time series. Is 
                                              used only if set and if 'action' 
                                              key value is set to 'modify'.
                                              Default = None.
                                'location'    [Optional. instance of GEOM.Point
                                              class] Interferometer location in 
                                              the local ENU coordinate system. 
                                              Used only if set and if 'action' 
                                              key value is set to 'modify'.
                                              Default=None.
                                'wtsinfo'     [Optional. Dictionary] See 
                                              description in Interferometer 
                                              class member function update(). 
                                              Is used only if set and if 
                                              'action' key value is set to 
                                              'modify'. Default = None.
                                'flags'       [Optional. Dictionary] holds 
                                              boolean flags for each of the 4 
                                              cross-polarizations which are 
                                              stored under keys 'P11', 'P12', 
                                              'P21' and 'P2'. Default=None means 
                                              no updates for flags. If True, 
                                              that polarization will be flagged. 
                                              If not set (=None), the previous or 
                                              default flag status will continue 
                                              to apply. If set to False, the 
                                              antenna status will be updated to 
                                              become unflagged.
                                'gridfunc_freq'
                                              [Optional. String scalar] Read the 
                                              description of inputs to 
                                              Interferometer class member 
                                              function update(). If set to None 
                                              (not provided), this attribute is 
                                              determined based on the size of 
                                              wtspos under each polarization. 
                                              It is applicable only when 'action' 
                                              key is set to 'modify'. 
                                              Default = None.
                                'ref_freq'    [Optional. Scalar] Positive value 
                                              (in Hz) of reference frequency 
                                              (used if gridfunc_freq is set to
                                              'scale') at which wtspos in 
                                              wtsinfo are provided. If set to 
                                              None, the reference frequency 
                                              already set in interferometer 
                                              array instance remains unchanged. 
                                              Default = None.
                                'pol_type'    [Optional. String scalar] 'Linear' 
                                              or 'Circular'. Used only when 
                                              action key is set to 'modify'. If 
                                              not provided, then the previous
                                              value remains in effect.
                                              Default = None.
                                'norm_wts'    [Optional. Boolean] Default=False. 
                                              If set to True, the gridded weights 
                                              are divided by the sum of weights 
                                              so that the gridded weights add up 
                                              to unity. This is used only when
                                              grid_action keyword is set when
                                              action keyword is set to 'add' or
                                              'modify'
                                'gridmethod'  [Optional. String] Indicates 
                                              gridding method. It accepts the 
                                              following values 'NN' (nearest 
                                              neighbour), 'BL' (Bi-linear
                                              interpolation), and'CS' (Cubic
                                              Spline interpolation). Default='NN'
                                'distNN'      [Optional. Scalar] Indicates the 
                                              upper bound on distance for a 
                                              nearest neighbour search if the 
                                              value of 'gridmethod' is set to
                                              'NN'. The units are of physical
                                              distance, the same as what is 
                                              used for interferometer locations.
                                              Default = NP.inf
                                'maxmatch'    [scalar] A positive value 
                                              indicating maximum number of input
                                              locations in the interferometer 
                                              grid to be assigned. Default=None. 
                                              If set to None, all the 
                                              interferometer array grid elements 
                                              specified are assigned values for 
                                              each interferometer. For instance, 
                                              to have only one interferometer 
                                              array grid element to be populated 
                                              per interferometer, use maxmatch=1. 
                                'tol'         [scalar] If set, only lookup data 
                                              with abs(val) > tol will be
                                              considered for nearest neighbour 
                                              lookup. Default = None implies 
                                              all lookup values will be 
                                              considered for nearest neighbour
                                              determination. tol is to be
                                              interpreted as a minimum value
                                              considered as significant in the
                                              lookup table. 

        do_correlate
                    [string] Indicates whether correlation operation is to be
                    performed after updates. Accepted values are 'FX' (for FX
                    operation) and 'XF' (for XF operation). Default=None means
                    no correlating operation is to be performed after updates.

        verbose     [Boolean] Default = False. If set to True, prints some 
                    diagnotic or progress messages.
        -------------------------------------------------------------------------
        """

        if verbose:
            print 'Updating interferometer array ...'

        if antenna_level_updates is not None:
            self.antenna_array.update(updates=antenna_level_updates)

        self.timestamp = self.antenna_array.timestamp
        self.t = self.antenna_array.t
        self.update_flags()  # Update interferometer flags using antenna level flags

        if interferometer_level_updates is not None:
            if not isinstance(interferometer_level_updates, dict):
                raise TypeError('Input parameter interferometer_level_updates must be a dictionary')

            if 'interferometers' in interferometer_level_updates:
                if not isinstance(interferometer_level_updates['interferometers'], list):
                    interferometer_level_updates['interferometers'] = [interferometer_level_updates['interferometers']]
                if parallel:
                    list_of_interferometer_updates = []
                    list_of_interferometers = []
                for dictitem in interferometer_level_updates['interferometers']:
                    if not isinstance(dictitem, dict):
                        raise TypeError('Interferometer_Level_Updates to {0} instance should be provided in the form of a list of dictionaries.'.format(self.__class__.__name__))
                    elif 'label' not in dictitem:
                        raise KeyError('No interferometer label specified in the dictionary item to be updated.')
    
                    if 'action' not in dictitem:
                        raise KeyError('No action specified for update. Action key should be set to "add", "remove" or "modify".')
                    elif dictitem['action'] == 'add':
                        if dictitem['label'] in self.interferometers:
                            if verbose:
                                print 'Interferometer {0} for adding already exists in current instance of {1}. Skipping over to the next item to be updated.'.format(dictitem['label'], self.__class__.__name__)
                        else:
                            if verbose:
                                print 'Adding interferometer {0}...'.format(dictitem['label'])
                            self.add_interferometers(dictitem['interferometer'])
                            if 'grid_action' in dictitem:
                                self.grid_convolve(pol=dictitem['gridpol'], antpairs=dictitem['interferometer'], unconvolve_existing=False)
                    elif dictitem['action'] == 'remove':
                        if dictitem['label'] not in self.interferometers:
                            if verbose:
                                print 'Interferometer {0} for removal not found in current instance of {1}. Skipping over to the next item to be updated.'.format(dictitem['label'], self.__class__.__name__) 
                        else:
                            if verbose:
                                print 'Removing interferometer {0}...'.format(dictitem['label'])
                            if 'grid_action' in dictitem:
                                self.grid_unconvolve(dictitem['label'], dictitem['gridpol'])
                            self.remove_interferometers(dictitem['label'])
                    elif dictitem['action'] == 'modify':
                        if dictitem['label'] not in self.interferometers:
                            if verbose:
                                print 'Interferometer {0} for modification not found in current instance of {1}. Skipping over to the next item to be updated.'.format(dictitem['label'], self.__class__.__name__)
                        else:
                            if verbose:
                                print 'Modifying interferometer {0}...'.format(dictitem['label'])
                            if 'Vt' not in dictitem: dictitem['Vt']=None
                            if 't' not in dictitem: dictitem['t']=None
                            if 'timestamp' not in dictitem: dictitem['timestamp']=None
                            if 'location' not in dictitem: dictitem['location']=None
                            if 'wtsinfo' not in dictitem: dictitem['wtsinfo']=None
                            if 'flags' not in dictitem: dictitem['flags']=None
                            if 'gridfunc_freq' not in dictitem: dictitem['gridfunc_freq']=None
                            if 'ref_freq' not in dictitem: dictitem['ref_freq']=None
                            if 'pol_type' not in dictitem: dictitem['pol_type']=None
                            if 'norm_wts' not in dictitem: dictitem['norm_wts']=False
                            if 'gridmethod' not in dictitem: dictitem['gridmethod']='NN'
                            if 'distNN' not in dictitem: dictitem['distNN']=NP.inf
                            if 'maxmatch' not in dictitem: dictitem['maxmatch']=None
                            if 'tol' not in dictitem: dictitem['tol']=None
                            if 'do_correlate' not in dictitem: dictitem['do_correlate']=None

                            if not parallel:
                                self.interferometers[dictitem['label']].update(dictitem['label'], dictitem['Vt'], dictitem['t'], dictitem['timestamp'], dictitem['location'], dictitem['wtsinfo'], dictitem['flags'], dictitem['gridfunc_freq'], dictitem['ref_freq'], dictitem['do_correlate'], verbose)
                            else:
                                list_of_interferometers += [self.interferometers[dictitem['label']]]
                                list_of_interferometer_updates += [dictitem]

                            if 'gric_action' in dictitem:
                                self.grid_convolve(pol=dictitem['gridpol'], antpairs=dictitem['interferometer'], unconvolve_existing=True, normalize=dictitem['norm_wts'], method=dictitem['gridmethod'], distNN=dictitem['distNN'], tol=dictitem['tol'], maxmatch=dictitem['maxmatch'])
                    else:
                        raise ValueError('Update action should be set to "add", "remove" or "modify".')

                if parallel:
                    if nproc is None:
                        nproc = max(MP.cpu_count()-1, 1) 
                    else:
                        nproc = min(nproc, max(MP.cpu_count()-1, 1))
                    pool = MP.Pool(processes=nproc)
                    updated_interferometers = pool.map(unwrap_interferometer_update, IT.izip(list_of_interferometers, list_of_interferometer_updates))
                    pool.close()
                    pool.join()

                    # Necessary to make the returned and updated interferometers current, otherwise they stay unrelated
                    for interferometer in updated_interferometers: 
                        self.interferometers[interferometer.label] = interferometer
                    del updated_interferometers

        if do_correlate is not None:
            if do_correlate == 'FX':
                self.FX(parallel=parallel, nproc=nproc)
            elif do_correlate == 'XF':
                self.XF(parallel=parallel, nproc=nproc)
            else:
                raise ValueError('Invalid specification for input parameter do_correlate.')

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

            sum_wts = NP.sum(self.grid_illumination_P1, axis=(0,1))

            self.holograph_P1 = NP.fft.fftshift(NP.fft.fft2(NP.pad(self.grid_Ef_P1, ((0,grid_shape[0]), (0,grid_shape[1]), (0,0)), 'constant', constant_values=(0,)), axes=(0,1))) / sum_wts
            if verbose:
                print '\t\tComputed complex holographic voltage image from antenna array.'

            self.holograph_PB_P1 = NP.fft.fftshift(NP.fft.fft2(NP.pad(self.grid_illumination_P1, ((0,grid_shape[0]), (0,grid_shape[1]), (0,0)), 'constant', constant_values=(0,)), axes=(0,1))) / sum_wts
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

            sum_wts = NP.sum(self.grid_illumination_P1, axis=(0,1))

            self.holograph_P2 = NP.fft.fftshift(NP.fft.fft2(NP.pad(self.grid_Ef_P2, ((0,grid_shape[0]), (0,grid_shape[1]), (0,0)), 'constant', constant_values=(0,)), axes=(0,1))) / sum_wts
            if verbose:
                print '\t\tComputed complex holographic voltage image from antenna array.'

            self.holograph_PB_P2 = NP.fft.fftshift(NP.fft.fft2(NP.pad(self.grid_illumination_P2, ((0,grid_shape[0]), (0,grid_shape[1]), (0,0)), 'constant', constant_values=(0,)), axes=(0,1))) / sum_wts
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

#################################################################################

class NewImage:

    """
    -----------------------------------------------------------------------------
    Class to manage image information and processing pertaining to the class 
    holding antenna array or interferometer array information.

    [Docstring is outdated. Needs to be updated definitely]

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
                 interferometer_array=None, infile=None, timestamp=None,
                 verbose=True):
        
        """
        -------------------------------------------------------------------------
        Initializes an instance of class Image which manages information and
        processing of images from data obtained by an antenna array or 
        interferometer array. It can be initialized either by values in an
        instance of class AntennaArray, by values in an instance of class
        InterferometerArray, or by values in a fits file containing information
        about the antenna array or interferometer array, or to defaults.

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

        self.measured_type = None

        if (infile is None) and (antenna_array is None) and (interferometer_array is None):
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
            raise ValueError('Both gridded data file and antenna array information are specified. One and only one of these should be specified. Cannot initialize an instance of class Image.')     

        if (infile is not None) and (interferometer_array is not None):
            raise ValueError('Both gridded data file and interferometer array information are specified. One and only one of these should be specified. Cannot initialize an instance of class Image.')     

        if (antenna_array is not None) and (interferometer_array is not None):
            raise ValueError('Both antenna array and interferometer array information are specified. One and only one of these should be specified. Cannot initialize an instance of class Image.')     

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
            
                self.grid_illumination = {}
                self.grid_Ef = {}
                self.img = {}
                self.beam = {}
                self.gridl = {}
                self.gridm = {}
                self.grid_wts = {}

                if pol is None:
                    pol = ['P1', 'P2']
                pol = NP.unique(NP.asarray(pol))

                self.gridu, self.gridv = antenna_array.gridu, antenna_array.gridv
                antenna_array.make_grid_cube(verbose=verbose, pol=pol)

                for apol in pol:
                    if apol in ['P1', 'P2']:
                        if verbose:
                            print '\n\t\tWorking on polarization {0}'.format(apol)
                                
                        self.img[apol] = None
                        self.beam[apol] = None
                        self.grid_wts[apol] = NP.zeros(self.gridu.shape+(self.f.size,))
                        if apol in antenna_array.grid_illumination:
                            self.grid_illumination[apol] = antenna_array.grid_illumination[apol]
                            self.grid_Ef[apol] = antenna_array.grid_Ef[apol]
                        else:
                            self.grid_illumination[apol] = None
                            self.grid_Ef[apol] = None
    
                self.measured_type = 'E-field'

                if verbose:
                    print '\t\tInitialized gridx, gridy, grid_illumination, and grid_Ef.'
                    print '\t\tInitialized gridl, gridm, and img'
            else:
                raise TypeError('antenna_array is not an instance of class AntennaArray. Cannot initiate instance of class Image.')

        if interferometer_array is not None:
            if verbose:
                print '\tInitializing from an instance of class InterferometerArray...'

            if isinstance(interferometer_array, InterferometerArray):
                self.f = interferometer_array.f
                if verbose:
                    print '\t\tInitialized frequency channels.'

                self.f0 = interferometer_array.f0
                if verbose:
                    print '\t\tInitialized center frequency to {0} Hz from interferometer array info.'.format(self.f0)

                self.timestamp = interferometer_array.timestamp
                if verbose:
                    print '\t\tInitialized time stamp to {0} from interferometer array info.'.format(self.timestamp)
            
                self.grid_illumination = {}
                self.grid_Vf = {}
                self.img = {}
                self.beam = {}
                self.gridl = {}
                self.gridm = {}
                self.grid_wts = {}

                if pol is None:
                    pol = ['P11', 'P12', 'P21', 'P22']
                pol = NP.unique(NP.asarray(pol))

                self.gridu, self.gridv = interferometer_array.gridu, interferometer_array.gridv
                interferometer_array.make_grid_cube(verbose=verbose, pol=pol)

                for cpol in pol:
                    if cpol in ['P11', 'P12', 'P21', 'P22']:
                        if verbose:
                            print '\n\t\tWorking on polarization {0}'.format(cpol)
                                
                        self.img[cpol] = None
                        self.beam[cpol] = None
                        self.grid_wts[cpol] = NP.zeros(self.gridu.shape+(self.f.size,))
                        if cpol in interferometer_array.grid_illumination:
                            self.grid_illumination[cpol] = interferometer_array.grid_illumination[cpol]
                            self.grid_Vf[cpol] = interferometer_array.grid_Vf[cpol]
                        else:
                            self.grid_illumination[cpol] = None
                            self.grid_Vf[cpol] = None
    
                self.measured_type = 'visibility'

                if verbose:
                    print '\t\tInitialized gridx, gridy, grid_illumination, and grid_Vf.'
                    print '\t\tInitialized gridl, gridm, and img'
                        
            else:
                raise TypeError('interferometer_array is not an instance of class InterferometerArray. Cannot initiate instance of class Image.')

        if verbose:
            print '\nSuccessfully initialized an instance of class Image\n'

    #############################################################################

    def imagr(self, pol=None, weighting='natural', verbose=True):

        """
        -------------------------------------------------------------------------
        Imaging engine that performs inverse fourier transforms of appropriate
        electric fields or visibilities associated with the antenna array or
        interferometer array respectively.

        Keyword Inputs:

        pol       [string] indicates which polarization information to be 
                  imaged. Allowed values are 'P1', 'P2' or None (default). If 
                  None, both polarizations are imaged.
        
        weighting [string] indicates weighting scheme. Default='natural'. 
                  Accepted values are 'natural' and 'uniform'

        verbose   [boolean] If True (default), prints diagnostic and progress
                  messages. If False, suppress printing such messages.
        -------------------------------------------------------------------------
        """

        if verbose:
            print '\nPreparing to image...\n'

        if self.f is None:
            raise ValueError('Frequency channels have not been initialized. Cannot proceed with imaging.')

        if self.measured_type is None:
            raise ValueError('Measured type is unknown.')

        if self.measured_type == 'E-field':
            if pol is None: pol = ['P1', 'P2']
            pol = NP.unique(NP.asarray(pol))
            grid_shape = self.gridu.shape
            for apol in pol:
                if apol in ['P1', 'P2']:
                    if verbose: print 'Preparing to Inverse Fourier Transform...'
                    if weighting == 'uniform':
                        self.grid_wts[apol][NP.abs(self.grid_illumination[apol]) > 0.0] = 1.0/NP.abs(self.grid_illumination[apol][NP.abs(self.grid_illumination[apol]) > 0.0])
                    else:
                        self.grid_wts[apol][NP.abs(self.grid_illumination[apol]) > 0.0] = 1.0

                    sum_wts = NP.sum(NP.abs(self.grid_wts[apol] * self.grid_illumination[apol]), axis=(0,1), keepdims=True)

                    self.beam[apol] = NP.fft.fftshift(NP.fft.fft2(self.grid_wts[apol]*self.grid_illumination[apol],axes=(0,1)).real, axes=(0,1)) / sum_wts
                    self.img[apol] = NP.fft.fftshift(NP.fft.fft2(self.grid_wts[apol]*self.grid_Ef[apol],axes=(0,1)).real, axes=(0,1)) / sum_wts

        if self.measured_type == 'visibility':
            if pol is None: pol = ['P11', 'P12', 'P21', 'P22']
            pol = NP.unique(NP.asarray(pol))
            grid_shape = self.gridu.shape
            for cpol in pol:
                if cpol in ['P11', 'P12', 'P21', 'P22']:
                    if verbose: print 'Preparing to Inverse Fourier Transform...'
                    if weighting == 'uniform':
                        self.grid_wts[cpol][NP.abs(self.grid_illumination[cpol]) > 0.0] = 1.0/NP.abs(self.grid_illumination[cpol][NP.abs(self.grid_illumination[cpol]) > 0.0])
                    else:
                        self.grid_wts[cpol][NP.abs(self.grid_illumination[cpol]) > 0.0] = 1.0

                    sum_wts = NP.sum(NP.abs(self.grid_wts[cpol] * self.grid_illumination[cpol]), axis=(0,1), keepdims=True)

                    self.beam[cpol] = NP.fft.fftshift(NP.fft.fft2(self.grid_wts[cpol]*self.grid_illumination[cpol],axes=(0,1)).real, axes=(0,1)) / sum_wts
                    self.img[cpol] = NP.fft.fftshift(NP.fft.fft2(self.grid_wts[cpol]*self.grid_Vf[cpol],axes=(0,1)).real, axes=(0,1)) / sum_wts
                    
        du = self.gridu[0,1] - self.gridu[0,0]
        dv = self.gridv[1,0] - self.gridv[0,0]
        self.gridl, self.gridm = NP.meshgrid(NP.fft.fftshift(NP.fft.fftfreq(grid_shape[1], du)), NP.fft.fftshift(NP.fft.fftfreq(grid_shape[0], dv)))
        nan_ind = NP.where(self.gridl**2 + self.gridm**2 > 1.0)
        # nan_ind_unraveled = NP.unravel_index(nan_ind, self.gridl.shape)
        # self.beam[cpol][nan_ind_unraveled,:] = NP.nan
        # self.img[cpol][nan_ind_unraveled,:] = NP.nan    

        if verbose:
            print 'Successfully imaged.'

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

#################################################################################

class PolInfo:

    """
    ----------------------------------------------------------------------------
    Class to manage polarization information of an antenna. 

    Attributes:

    Et       [dictionary] holds measured complex time series under 2 
             polarizations which are stored under keys 'P1', and 'P2'

    Ef       [dictionary] holds complex electric field spectra under 2 
             polarizations which are stored under keys 'P1', and 'P2'. The 
             length of the spectra is twice that of the time series.

    flag     [dictionary] holds boolean flags for each of the 2 polarizations
             which are stored under keys 'P1', and 'P2'. Default=True means  
             that polarization is flagged.

    Member functions:

    __init__():    Initializes an instance of class PolInfo

    __str__():     Prints a summary of current attributes.

    FT():          Perform a Fourier transform of an Electric field time series

    update():      Routine to update the Electric field and flag information.
    
    delay_compensation():
                   Routine to apply delay compensation to Electric field spectra 
                   through additional phase.

    Read the member function docstrings for details. 
    ----------------------------------------------------------------------------
    """

    def __init__(self, nsamples=1):
        """
        ------------------------------------------------------------------------
        Initialize the PolInfo Class which manages polarization information of
        an antenna. 

        Class attributes initialized are:
        Et, Ef, flag
     
        Read docstring of class PolInfo for details on these attributes.
        ------------------------------------------------------------------------
        """

        self.Et = {}
        self.Ef = {}
        self.flag = {}

        if not isinstance(nsamples, int):
            raise TypeError('nsamples must be an integer')
        elif nsamples <= 0:
            nsamples = 1

        for pol in ['P1', 'P2']:
            self.Et[pol] = NP.empty(nsamples, dtype=NP.complex64)
            self.Ef[pol] = NP.empty(2*nsamples, dtype=NP.complex64)
            
            self.Et[pol].fill(NP.nan)
            self.Ef[pol].fill(NP.nan)

            self.flag[pol] = True

    ############################################################################ 

    def __str__(self):
        return ' Instance of class "{0}" in module "{1}" \n flag (P1): {2} \n flag (P2): {3} '.format(self.__class__.__name__, self.__module__, self.flag_P1, self.flag_P2)

    ############################################################################ 

    def FT(self, pol=None):

        """
        ------------------------------------------------------------------------
        Perform a Fourier transform of an Electric field time series after 
        doubling the length of the sequence with zero padding (in order to be 
        identical to what would be obtained from a XF oepration)

        Keyword Input(s):

        pol     polarization to be Fourier transformed. Set to 'P1' or 'P2'. If 
                None provided, time series of both polarizations are Fourier 
                transformed.
        ------------------------------------------------------------------------
        """

        if pol is None:
            pol = ['P1', 'P2']

        for p in pol:
            if p in ['P1', 'P2']:
                Et = NP.pad(self.Et[p], (0,len(self.Et[p])), 'constant', constant_values=(0,0))
                self.Ef[p] = DSP.FT1D(Et, ax=0, use_real=False, inverse=False, shift=True)
            else:
                raise ValueError('polarization string "{0}" unrecognized. Verify inputs. Aborting {1}.{2}()'.format(p, self.__class__.__name__, 'FT'))

    ############################################################################ 

    def delay_compensation(self, delaydict):
        
        """
        -------------------------------------------------------------------------
        Routine to apply delay compensation to Electric field spectra through
        additional phase. This assumes that the spectra have already been made

        Keyword input(s):

        delaydict   [dictionary] contains one or both polarization keys, namely,
                    'P1' and 'P2'. The value under each of these keys is another 
                    dictionary with the following keys and values:
                    'frequencies': scalar, list or numpy vector specifying the 
                           frequencie(s) (in Hz) for which delays are specified. 
                           If a scalar is specified, the delays are assumed to be
                           frequency independent and the delays are assumed to be
                           valid for all frequencies. If a vector is specified, 
                           it must be of same size as the delays and as the 
                           number of samples in the electric field timeseries. 
                           These frequencies are assumed to match those of the 
                           electric field spectrum. No default.
                    'delays': list or numpy vector specifying the delays (in 
                           seconds) at the respective frequencies which are to be 
                           compensated through additional phase in the electric 
                           field spectrum. Must be of same size as frequencies 
                           and the size of the electric field timeseries. No
                           default.
                    'fftshifted': boolean scalar indicating if the frequencies
                           provided have already been fft-shifted. If True 
                           (default) or this key is absent, the frequencies are 
                           assumed to have been fft-shifted. If False, they have 
                           to be fft-shifted before applying the delay 
                           compensation to rightly align with the fft-shifted 
                           electric field spectrum computed in member function 
                           FT(). 
        -------------------------------------------------------------------------
        """

        try:
            delaydict
        except NameError:
            raise NameError('Delay information must be supplied for delay correction in the dictionary delaydict.')

        if not isinstance(delaydict, dict):
            raise TypeError('delaydict must be a dictionary')

        for pol in delaydict:
            if pol not in ['P1','P2']:
                raise ValueError('Invalid specification for polarization')
                
            if 'delays' in delaydict[pol]:
                if NP.asarray(delaydict[pol]['delays']).size == 1:
                    delays = delaydict[pol]['delays'] + NP.zeros(self.Et[pol].size)
                else:
                    if (NP.asarray(delaydict[pol]['delays']).size == self.Et[pol].size):
                        delays = NP.asarray(delaydict[pol]['delays']).ravel()
                    else:
                        raise IndexError('Size of delays in delaydict must be equal to 1 or match that of the timeseries.')

                if 'frequencies' in delaydict[pol]:
                    frequencies = NP.asarray(delaydict[pol]['frequencies']).ravel()
                    if frequencies.size != self.Et[pol].size:
                        raise IndexError('Size of frequencies must match that of the Electric field time series.')
                else:
                    raise KeyError('Key "frequencies" not found in dictionary delaydict[{0}] holding delay information.'.format(pol))

                temp_phases = 2 * NP.pi * delays * frequencies

                # Convert phases to fft-shifted arrangement based on key "fftshifted" in delaydict
                if 'fftshifted' in delaydict[pol]:
                    if not isinstance(delaydict[pol]['fftshifted'], bool):
                        raise TypeError('Value under key "fftshifted" must be boolean')

                    if not delaydict[pol]['fftshifted']:
                        temp_phases = NP.fft.fftshift(temp_phases)

                # Expand the size to account for the fact that the Fourier transform of the timeseries is obtained after zero padding
                phases = NP.empty(2*frequencies.size) 
                phases[0::2] = temp_phases
                phases[1::2] = temp_phases
  
                self.Ef[pol] *= NP.exp(1j * phases)
                    
        ## INSERT FEATURE: yet to modify the timeseries after application of delay compensation ##

    ############################################################################ 

    def update_flags(self, flags=None):

        """
        ------------------------------------------------------------------------
        Updates the flags based on current inputs
    
        Inputs:
    
        flags    [dictionary] holds boolean flags for each of the 2 
                 polarizations which are stored under keys 'P1', and 'P2'.
                 Default=None means no new flagging to be applied. If 
                 the value under the polarization key is True, it is to be 
                 flagged and if False, it is to be unflagged.
        ------------------------------------------------------------------------
        """

        if flags is not None:
            if not isinstance(flags, dict):
                raise TypeError('Input parameter flags must be a dictionary')
            for pol in ['P1', 'P2']:
                if pol in flags:
                    if isinstance(flags[pol], bool):
                        self.flag[pol] = flags[pol]
                    else:
                        raise TypeError('flag values must be boolean')

    ############################################################################ 

    def update(self, Et=None, Ef=None, flags=None, delaydict=None):
        
        """
        ------------------------------------------------------------------------
        Updates the electric field time series and spectra for different
        polarizations

        Inputs:
        
        Et     [dictionary] holds time series under 2 polarizations which are 
               stored under keys 'P1', and 'P2'. Default=None implies no updates 
               for Et.

        Ef     [dictionary] holds spectra under 2 polarizations which are 
               stored under keys 'P1', and 'P2'. Default=None implies no updates 
               for Ef.

        flag   [dictionary] holds boolean flags for each of the 2 polarizations 
               which are stored under keys 'P1', and 'P2'. Default=None means 
               no updates for flags.

        delaydict
               [dictionary] contains one or both polarization keys, namely,
               'P1' and 'P2'. The value under each of these keys is another 
               dictionary with the following keys and values:
               'frequencies': scalar, list or numpy vector specifying the 
                      frequencie(s) (in Hz) for which delays are specified. 
                      If a scalar is specified, the delays are assumed to be
                      frequency independent and the delays are assumed to be
                      valid for all frequencies. If a vector is specified, 
                      it must be of same size as the delays and as the 
                      number of samples in the electric field timeseries. 
                      These frequencies are assumed to match those of the 
                      electric field spectrum. No default.
               'delays': list or numpy vector specifying the delays (in 
                      seconds) at the respective frequencies which are to be 
                      compensated through additional phase in the electric 
                      field spectrum. Must be of same size as frequencies 
                      and the size of the electric field timeseries. No
                      default.
               'fftshifted': boolean scalar indicating if the frequencies
                      provided have already been fft-shifted. If True 
                      (default) or this key is absent, the frequencies are 
                      assumed to have been fft-shifted. If False, they have 
                      to be fft-shifted before applying the delay 
                      compensation to rightly align with the fft-shifted 
                      electric field spectrum computed in member function 
                      FT(). 
        ------------------------------------------------------------------------
        """

        if flags is not None:
            self.update_flags(flags)

        if Et is not None:
            if isinstance(Et, dict):
                for pol in ['P1', 'P2']:
                    if pol in Et:
                        self.Et[pol] = Et[pol]
                        if NP.any(NP.isnan(Et[pol])):
                            # self.Et[pol] = NP.nan
                            self.flag[pol] = True
                self.FT()  # Update the spectrum
            else:
                raise TypeError('Input parameter Et must be a dictionary')

        if Ef is not None:
            if isinstance(Ef, dict):
                for pol in ['P1', 'P2']:
                    if pol in Ef:
                        self.Ef[pol] = Ef[pol]
                        if NP.any(NP.isnan(Ef[pol])):
                            # self.Ef[pol] = NP.nan
                            self.flag[pol] = True
            else:
                raise TypeError('Input parameter Ef must be a dictionary')

        if delaydict is not None:
            self.delay_compensation(delaydict)

#################################################################################

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

    f0:         [Scalar] Center frequency in Hz.

    antpol:     [Instance of class PolInfo] polarization information for the 
                antenna. Read docstring of class PolInfo for details

    wts:        [dictionary] The gridding weights for antenna. Different 
                polarizations 'P1' and 'P2' form the keys 
                of this dictionary. These values are in general complex. Under 
                each key, the values are maintained as a list of numpy vectors, 
                where each vector corresponds to a frequency channel. See 
                wtspos_scale for more requirements.

    wtspos      [dictionary] two-dimensional locations of the gridding weights in
                wts for each polarization under keys 'P1' and 'P2'. The locations 
                are in ENU coordinate system as a list of 2-column numpy arrays. 
                Each 2-column array in the list is the position of the gridding 
                weights for a corresponding frequency 
                channel. The size of the list must be the same as wts and the 
                number of channels. Units are in number of wavelengths. See 
                wtspos_scale for more requirements.

    wtspos_scale [dictionary] The scaling of weights is specified for each 
                 polarization under one of the keys 'P1' and 'P2'. 
                 The values under these keys can be either None (default) or 
                 'scale'. If None, numpy vectors in wts and wtspos under
                 corresponding keys are provided for each frequency channel. If 
                 set to 'scale' wts and wtspos contain a list of only one 
                 numpy array corresponding to a reference frequency. This is
                 scaled internally to correspond to the first channel.
                 The gridding positions are correspondingly scaled to all the 
                 frequency channels.

    blc          [2-element numpy array] Bottom Left corner where the
                 antenna contributes non-zero weight to the grid. Same 
                 for all polarizations

    trc          [2-element numpy array] Top right corner where the
                 antenna contributes non-zero weight to the grid. Same 
                 for all polarizations

    Member Functions:

    __init__():  Initializes an instance of class Antenna

    __str__():   Prints a summary of current attributes

    channels():  Computes the frequency channels from a temporal Fourier 
                 Transform

    update_flags()
                 Updates flags for polarizations provided as input parameters

    update():    Updates the antenna instance with newer attribute values
                 Updates the electric field spectrum and timeseries. It also
                 applies Fourier transform if timeseries is updated

    save():      Saves the antenna information to disk. Needs serious 
                 development. 

    Read the member function docstrings for details.
    ----------------------------------------------------------------------------
    """

    def __init__(self, label, latitude, location, center_freq, nsamples=1):

        """
        ------------------------------------------------------------------------
        Initialize the Antenna Class which manages an antenna's information 

        Class attributes initialized are:
        label, latitude, location, pol, t, timestamp, f0, f, wts, wtspos, 
        wtspos_scale, blc, trc, and antpol
     
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

        self.antpol = PolInfo(nsamples=nsamples)
        self.t = 0.0
        self.timestamp = 0.0
        self.f0 = center_freq
        self.f = self.f0

        self.wts = {}
        self.wtspos = {}
        self.wtspos_scale = {}
        self._gridinfo = {}

        for pol in ['P1', 'P2']:
            self.wtspos[pol] = []
            self.wts[pol] = []
            self.wtspos_scale[pol] = None
            self._gridinfo[pol] = {}
        
        self.blc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1)
        self.trc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1)

    #################################################################################

    def __str__(self):
        return ' Instance of class "{0}" in module "{1}" \n label: ({2[0]}, {2[1]}) \n location: {3}'.format(self.__class__.__name__, self.__module__, self.label, self.location.__str__())

    #################################################################################

    def channels(self):

        """
        ------------------------------------------------------------------------
        Computes the frequency channels from a temporal Fourier Transform 

        Output(s):

        Frequencies corresponding to channels obtained by a Fourier Transform
        of the time series.
        ------------------------------------------------------------------------
        """

        return DSP.spectax(2*self.t.size, self.t[1]-self.t[0], shift=True)

    #############################################################################

    def FT(self, pol=None):

        """
        ----------------------------------------------------------------------------
        Computes the Fourier transform of the time series of the antennas in the 
        antenna array to compute the visibility spectra
        ----------------------------------------------------------------------------
        """
        
        self.antpol.FT(pol=pol)
        
    ################################################################################# 

    def FT_new(self, pol=None):

        """
        ----------------------------------------------------------------------------
        Computes the Fourier transform of the time series of the antennas in the 
        antenna array to compute the visibility spectra
        ----------------------------------------------------------------------------
        """
        
        self.antpol.FT(pol=pol)
        return self
        
    ################################################################################# 

    def update_flags(self, flags=None):

        """
        ------------------------------------------------------------------------
        Updates flags for antenna polarizations 

        Inputs:

        flags  [dictionary] boolean flags for each of the 2 polarizations 
               of the antenna which are stored under keys 'P1' and 'P2',
               Default=None means no updates for flags.
        ------------------------------------------------------------------------
        """

        if flags is not None:
            self.antpol.update_flags(flags)

    ############################################################################

    def update(self, update_dict=None, verbose=True):

        """
        -------------------------------------------------------------------------
        Updates the antenna instance with newer attribute values. Updates 
        the electric field spectrum and timeseries

        Inputs:

        update_dict [dictionary] contains the following keys and values:

            label      [Scalar] A unique identifier (preferably a string) for 
                       the antenna. Default=None means no update to apply
    
            latitude   [Scalar] Latitude of the antenna's location. Default=None 
                       means no update to apply
    
            location   [Instance of GEOM.Point class] The location of the 
                       antenna in local East, North, Up (ENU) coordinate system. 
                       Default=None means no update to apply
    
            timestamp  [Scalar] String or float representing the timestamp for 
                       the current attributes. Default=None means no update to 
                       apply
    
            t          [vector] The time axis for the electric field time 
                       series. Default=None means no update to apply
    
            flags      [dictionary] holds boolean flags for each of the 2
                       polarizations which are stored under keys 'P1' and 'P22'. 
                       Default=None means no updates for flags.
    
            Et         [dictionary] holds time series under 2 polarizations 
                       which are stored under keys 'P1' and 'P22'. Default=None 
                       implies no updates for Et.
    
            wtsinfo    [dictionary] consists of weights information for each of 
                       the two polarizations under keys 'P1' and 'P22'. Each of 
                       the values under the keys is a list of dictionaries. 
                       Length of list is equal to the number of frequency 
                       channels or one (equivalent to setting wtspos_scale to 
                       'scale'.). The list is indexed by the frequency channel 
                       number. Each element in the list consists of a dictionary 
                       corresponding to that frequency channel. Each dictionary 
                       consists of these items with the following keys:
                       wtspos      [2-column Numpy array, optional] u- and v- 
                                   positions for the gridding weights. Units
                                   are in number of wavelengths.
                       wts         [Numpy array] Complex gridding weights. Size 
                                   is equal to the number of rows in wtspos 
                                   above
                       orientation [scalar] Orientation (in radians) of the 
                                   wtspos coordinate system relative to the 
                                   local ENU coordinate system. It is measured 
                                   North of East. 
                       lookup      [string] If set, refers to a file location
                                   containing the wtspos and wts information 
                                   above as columns (x-loc [float], y-loc 
                                   [float], wts[real], wts[imag if any]). If 
                                   set, wtspos and wts information are obtained 
                                   from this lookup table and the wtspos and wts 
                                   keywords in the dictionary are ignored. Note 
                                   that wtspos values are obtained after 
                                   dividing x- and y-loc lookup values by the 
                                   wavelength
    
            gridfunc_freq
                       [String scalar] If set to None (not provided) or to 
                       'scale' assumes that wtspos in wtsinfo are given for a
                       reference frequency which need to be scaled for the 
                       frequency channels. Will be ignored if the list of 
                       dictionaries under the polarization keys in wtsinfo have 
                       number of elements equal to the number of frequency 
                       channels.
   
            ref_freq   [Scalar] Positive value (in Hz) of reference frequency 
                       (used if gridfunc_freq is set to None or 'scale') at 
                       which wtspos is provided. If set to None, ref_freq is 
                       assumed to be equal to the center frequency in the class
                       Antenna's attribute. 

            delaydict  [Dictionary] contains information on delay compensation 
                       to be applied to the fourier transformed electric fields 
                       under each polarization which are stored under keys 'P1' 
                       and 'P2'. Default is None (no delay compensation to be 
                       applied). Refer to the docstring of member function
                       delay_compensation() of class PolInfo for more details.

        verbose    [boolean] If True, prints diagnostic and progress messages. 
                   If False (default), suppress printing such messages.
        -------------------------------------------------------------------------
        """

        label = None
        location = None
        timestamp = None
        t = None
        flags = None
        Et = None
        wtsinfo = None
        gridfunc_freq = None
        ref_freq = None
        delaydict = None
            
        if update_dict is not None:
            if not isinstance(update_dict, dict):
                raise TypeError('Input parameter containing updates must be a dictionary')

            if 'label' in update_dict: label = update_dict['label']
            if 'location' in update_dict: location = update_dict['location']
            if 'timestamp' in update_dict: timestamp = update_dict['timestamp']
            if 't' in update_dict: t = update_dict['t']
            if 'Et' in update_dict: Et = update_dict['Et']
            if 'flags' in update_dict: flags = update_dict['flags']
            if 'wtsinfo' in update_dict: wtsinfo = update_dict['wtsinfo']
            if 'gridfunc_freq' in update_dict: gridfunc_freq = update_dict['gridfunc_freq']
            if 'ref_freq' in update_dict: ref_freq = update_dict['ref_freq']
            if 'delaydict' in update_dict: delaydict = update_dict['delaydict']

        if label is not None: self.label = label
        if location is not None: self.location = location
        if timestamp is not None: self.timestamp = timestamp

        if t is not None:
            self.t = t
            self.f = self.f0 + self.channels()     

        if flags is not None:        
            self.update_flags(flags) 

        if (Et is not None) or (delaydict is not None):
            self.antpol.update(Et=Et, delaydict=delaydict)
        
        blc_orig = NP.copy(self.blc)
        trc_orig = NP.copy(self.trc)
        eps = 1e-6

        if wtsinfo is not None:
            if not isinstance(wtsinfo, dict):
                raise TypeError('Input parameter wtsinfo must be a dictionary.')

            self.wtspos = {}
            self.wts = {}
            self.wtspos_scale = {}
            angles = []
            
            max_wtspos = []
            for pol in ['P1', 'P2']:
                self.wts[pol] = []
                self.wtspos[pol] = []
                self.wtspos_scale[pol] = None
                if pol in wtsinfo:
                    if len(wtsinfo[pol]) == len(self.f):
                        angles += [elem['orientation'] for elem in wtsinfo[pol]]
                        for i in xrange(len(self.f)):
                            rotation_matrix = NP.asarray([[NP.cos(-angles[i]),  NP.sin(-angles[i])],
                                                          [-NP.sin(-angles[i]), NP.cos(-angles[i])]])
                            if ('lookup' not in wtsinfo[pol][i]) or (wtsinfo[pol][i]['lookup'] is None):
                                self.wts[pol] += [wtsinfo[pol][i]['wts']]
                                wtspos = wtsinfo[pol][i]['wtspos']
                            else:
                                lookupdata = LKP.read_lookup(wtsinfo[pol][i]['lookup'])
                                wtspos = NP.hstack((lookupdata[0].reshape(-1,1),lookupdata[1].reshape(-1,1))) * (self.f[i]/FCNST.c)
                                self.wts[pol] += [lookupdata[2]]
                            self.wtspos[pol] += [ NP.dot(NP.asarray(wtspos), rotation_matrix.T) ]
                            max_wtspos += [NP.amax(NP.abs(self.wtspos[pol][-1]), axis=0)]
                    elif len(wtsinfo[pol]) == 1:
                        if (gridfunc_freq is None) or (gridfunc_freq == 'scale'):
                            self.wtspos_scale[pol] = 'scale'
                            if ref_freq is None:
                                ref_freq = self.f0
                            angles = wtsinfo[pol][0]['orientation']
                            rotation_matrix = NP.asarray([[NP.cos(-angles),  NP.sin(-angles)],
                                                          [-NP.sin(-angles), NP.cos(-angles)]])
                            if ('lookup' not in wtsinfo[pol][0]) or (wtsinfo[pol][0]['lookup'] is None):
                                self.wts[pol] += [ wtsinfo[pol][0]['wts'] ]
                                wtspos = wtsinfo[pol][0]['wtspos']
                            else:
                                lookupdata = LKP.read_lookup(wtsinfo[pol][0]['lookup'])
                                wtspos = NP.hstack((lookupdata[0].reshape(-1,1),lookupdata[1].reshape(-1,1))) * (ref_freq/FCNST.c)
                                self.wts[pol] += [lookupdata[2]]
                            self.wtspos[pol] += [ (self.f[0]/ref_freq) * NP.dot(NP.asarray(wtspos), rotation_matrix.T) ]     
                            max_wtspos += [NP.amax(NP.abs(self.wtspos[pol][-1]), axis=0)]
                        else:
                            raise ValueError('gridfunc_freq must be set to None, "scale" or "noscale".')
    
                        self.blc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) - FCNST.c/self.f.min() * NP.amin(NP.abs(self.wtspos[pol][0]), 0)
                        self.trc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) + FCNST.c/self.f.min() * NP.amax(NP.abs(self.wtspos[pol][0]), 0)
    
                    else:
                        raise ValueError('Number of elements in wtsinfo for {0} is incompatible with the number of channels.'.format(pol))
               
            max_wtspos = NP.amax(NP.asarray(max_wtspos).reshape(-1,blc_orig.size), axis=0)
            self.blc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) - FCNST.c/self.f.min() * max_wtspos
            self.trc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1) + FCNST.c/self.f.min() * max_wtspos

        if (NP.abs(NP.linalg.norm(blc_orig)-NP.linalg.norm(self.blc)) > eps) or (NP.abs(NP.linalg.norm(trc_orig)-NP.linalg.norm(self.trc)) > eps):
            if verbose:
                print 'Grid corner(s) of antenna {0} have changed. Should re-grid the antenna array.'.format(self.label)

    ############################################################################

    def update_new(self, update_dict=None, verbose=True):

        """
        -------------------------------------------------------------------------
        Wrapper for member function update() and returns the updated instance of
        this class. Mostly intended to be used when parallel processing is 
        applicable and not to be used directly. Use update() instead when 
        updates are to be applied directly.

        See member function update() for details on inputs.
        -------------------------------------------------------------------------
        """

        self.update(update_dict=update_dict, verbose=verbose)
        return self

#################################################################################

class AntennaArray:

    """
    ----------------------------------------------------------------------------
    Class to manage collective information on a group of antennas.

    Attributes:

    antennas:    [Dictionary] Dictionary consisting of keys which hold instances
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

    grid_ready_P1 
                 [boolean] True if the grid has been created for P1 polarization,
                 False otherwise

    grid_ready_P2
                 [boolean] True if the grid has been created for P2 polarization,
                 False otherwise

    gridx_P1     [Numpy array] x-locations of the grid lattice for P1
                 polarization

    gridy_P1     [Numpy array] y-locations of the grid lattice for P1
                 polarization

    gridx_P2     [Numpy array] x-locations of the grid lattice for P2
                 polarization

    gridy_P2     [Numpy array] y-locations of the grid lattice for P2
                 polarization

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
                      
    add_antennas()    Routine to add antenna(s) to the antenna array instance. 
                      A wrapper for operator overloading __add__() and __radd__()
                      
    remove_antennas() Routine to remove antenna(s) from the antenna array 
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

    def __init__(self, antenna_array=None):

        """
        ------------------------------------------------------------------------
        Initialize the AntennaArray Class which manages information about an 
        array of antennas.

        Class attributes initialized are:
        antennas, blc, trc, gridx, gridy, gridu, gridv, grid_ready, timestamp, 
        grid_illumination, grid_Ef, f, f0, t, ordered_labels, grid_mapper
     
        Read docstring of class AntennaArray for details on these attributes.
        ------------------------------------------------------------------------
        """

        self.antennas = {}
        self.blc = NP.zeros(2)
        self.trc = NP.zeros(2)
        self.grid_blc = NP.zeros(2)
        self.grid_trc = NP.zeros(2)
        self.gridx, self.gridy = None, None
        self.gridu, self.gridv = None, None
        self.grid_ready = False
        self.grid_illumination = {}
        self.grid_Ef = {}
        self.f = None
        self.f0 = None
        self.t = None
        self.timestamp = None

        self._ant_contribution = {}

        self.ordered_labels = [] # Usually output from member function baseline_vectors() or get_visibilities()
        self.grid_mapper = {}

        for pol in ['P1', 'P2']:
            self.grid_mapper[pol] = {}
            self.grid_mapper[pol]['labels'] = {}
            self.grid_mapper[pol]['refind'] = []
            # self.grid_mapper[pol]['ant_ind'] = []
            self.grid_mapper[pol]['gridind'] = []
            self.grid_mapper[pol]['refwts'] = None
            self.grid_mapper[pol]['ant'] = {}
            self.grid_mapper[pol]['ant']['ind_freq'] = []
            self.grid_mapper[pol]['ant']['ind_all'] = None
            self.grid_mapper[pol]['ant']['uniq_ind_all'] = None
            self.grid_mapper[pol]['ant']['rev_ind_all'] = None
            self.grid_mapper[pol]['ant']['illumination'] = None
            self.grid_mapper[pol]['grid'] = {}
            self.grid_mapper[pol]['grid']['ind_all'] = None

            self.grid_illumination[pol] = None
            self.grid_Ef[pol] = None
            self._ant_contribution[pol] = {}

        if antenna_array is not None:
            self += antenna_array
            self.f = NP.copy(self.antennas.itervalues().next().f)
            self.f = NP.copy(self.antennas.itervalues().next().f0)
            self.t = NP.copy(self.antennas.itervalues().next().t)
            self.timestamp = copy.deepcopy(self.antennas.itervalues().next().timestamp)
        
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
                print "For updating, use the update() method. Ignoring antenna {0}".format(others.label)
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

    def antenna_positions(self, pol=None, flag=False, sort=True):
        
        """
        ----------------------------------------------------------------------------
        Routine to return the antenna label and position vectors (sorted by
        antenna label if specified)

        Keyword Inputs:

        pol      [string] select positions of this polarization that are either 
                 flagged or unflagged as specified by input parameter flag. 
                 Allowed values are 'P1' and 'P2'. Default=None. 
                 This means all positions are returned irrespective of the flags

        flag     [boolean] If False, return unflagged positions, otherwise return
                 flagged ones. Default=None means return all positions
                 independent of flagging or polarization

        sort     [boolean] If True, returned antenna information is sorted 
                 by antenna label. Default = True.

        Output:

        outdict  [dictionary] Output consists of a dictionary with the following 
                 keys and information:
                 'labels':    Contains a numpy array of strings of antenna 
                              labels
                 'positions': position vectors of antennas (3-column 
                              array)
        ----------------------------------------------------------------------------
        """

        if not isinstance(sort, bool):
            raise TypeError('sort keyword has to be a Boolean value.')

        if flag is not None:
            if not isinstance(flag, bool):
                raise TypeError('flag keyword has to be a Boolean value.')

        if pol is None:
            if sort: # sort by first antenna label
                xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in sorted(self.antennas.keys(), key=lambda tup: tup[0])])
                labels = sorted(self.antennas.keys(), key=lambda tup: tup[0])
            else:
                xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in self.antennas.keys()])
                labels = self.antennas.keys()
        else:
            if not isinstance(pol, str):
                raise TypeError('Input parameter must be a string')
            
            if pol not in ['P1', 'P2']:
                raise ValueError('Invalid specification for input parameter pol')

            if sort:                   # sort by first antenna label
                if flag is None:       # get all positions
                    xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in sorted(self.antennas.keys(), key=lambda tup: tup[0])])
                    labels = [label for label in sorted(self.antennas.keys(), key=lambda tup: tup[0])]
                else:
                    if flag:           # get flagged positions
                        xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in sorted(self.antennas.keys(), key=lambda tup: tup[0]) if self.antennas[label].antpol.flag[pol]])
                        labels = [label for label in sorted(self.antennas.keys(), key=lambda tup: tup[0]) if self.antennas[label].antpol.flag[pol]]                    
                    else:              # get unflagged positions
                        xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in sorted(self.antennas.keys(), key=lambda tup: tup[0]) if not self.antennas[label].antpol.flag[pol]])
                        labels = [label for label in sorted(self.antennas.keys(), key=lambda tup: tup[0]) if not self.antennas[label].antpol.flag[pol]]
            else:                      # no sorting
                if flag is None:       # get all positions
                    xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in self.antennas.keys()])
                    labels = [label for label in self.antennas.keys()]
                else:
                    if flag:           # get flagged positions
                        xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in self.antennas.keys() if self.antennas[label].antpol.flag[pol]])
                        labels = [label for label in self.antennas.keys() if self.antennas[label].antpol.flag[pol]]
                    else:              # get unflagged positions
                        xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in self.antennas.keys() if not self.antennas[label].antpol.flag[pol]])
                        labels = [label for label in self.antennas.keys() if not self.antennas[label].antpol.flag[pol]]

        outdict = {}
        outdict['labels'] = labels
        outdict['positions'] = xyz

        return outdict

    ################################################################################# 

    def get_E_fields(self, pol, flag=False, sort=True):

        """
        ----------------------------------------------------------------------------
        Routine to return the antenna label and Electric fields (sorted by
        antenna label if specified)

        Keyword Inputs:

        pol      [string] select antenna positions of this polarization that are 
                 either flagged or unflagged as specified by input parameter flag. 
                 Allowed values are 'P1' and 'P22'. Only one of these values must 
                 be specified.

        flag     [boolean] If False, return electric fields of unflagged antennas,
                 otherwise return flagged ones. Default=None means all electric 
                 fields independent of flagging are returned.

        sort     [boolean] If True, returned antenna information is sorted 
                 by antenna label. Default = True.

        Output:

        outdict  [dictionary] Output consists of a dictionary with the following 
                 keys and information:
                 'labels':    Contains a numpy array of strings of antenna 
                              labels
                 'E-fields':    measured electric fields (n_ant x nchan array)
        ----------------------------------------------------------------------------
        """

        try: 
            pol 
        except NameError:
            raise NameError('Input parameter pol must be specified.')

        if not isinstance(pol, str):
            raise TypeError('Input parameter must be a string')
        
        if not pol in ['P1', 'P2']:
            raise ValueError('Invalid specification for input parameter pol')

        if not isinstance(sort, bool):
            raise TypeError('sort keyword has to be a Boolean value.')

        if flag is not None:
            if not isinstance(flag, bool):
                raise TypeError('flag keyword has to be a Boolean value.')

        if sort:                   # sort by first antenna label
            if flag is None:       # get all antenna positions
                    efields = NP.asarray([self.antennas[label].antpol.Ef[pol] for label in sorted(self.antennas.keys(), key=lambda tup: tup[0])])
                    labels = [label for label in sorted(self.antennas.keys(), key=lambda tup: tup[0])]   
            else:
                if flag:           # get flagged antenna positions
                    efields = NP.asarray([self.antennas[label].antpol.Ef[pol] for label in sorted(self.antennas.keys(), key=lambda tup: tup[0]) if self.antennas[label].antpol.flag[pol]])
                    labels = [label for label in sorted(self.antennas.keys(), key=lambda tup: tup[0]) if self.antennas[label].antpol.flag[pol]]                    
                else:              # get unflagged antenna positions
                    efields = NP.asarray([self.antennas[label].antpol.Ef[pol] for label in sorted(self.antennas.keys(), key=lambda tup: tup[0]) if not self.antennas[label].antpol.flag[pol]])
                    labels = [label for label in sorted(self.antennas.keys(), key=lambda tup: tup[0]) if not self.antennas[label].antpol.flag[pol]]
        else:                      # no sorting
            if flag is None:
                efields = NP.asarray([self.antennas[label].antpol.Ef[pol] for label in self.antennas.keys()])
                labels = [label for label in self.antennas.keys()]
            else:
                if flag:               # get flagged antenna positions
                    efields = NP.asarray([self.antennas[label].antpol.Ef[pol] for label in self.antennas.keys() if self.antennas[label].antpol.flag[pol]])
                    labels = [label for label in self.antennas.keys() if self.antennas[label].antpol.flag[pol]]                    
                else:                  # get unflagged antenna positions
                    efields = NP.asarray([self.antennas[label].antpol.Ef[pol] for label in self.antennas.keys() if not self.antennas[label].antpol.flag[pol]])
                    labels = [label for label in sorted(self.antennas.keys(), key=lambda tup: tup[0]) if not self.antennas[label].antpol.flag[pol]]

        outdict = {}
        outdict['labels'] = labels
        outdict['E-fields'] = efields

        return outdict

    ################################################################################# 

    def FT(self, pol=None, parallel=False, nproc=None):

        """
        ----------------------------------------------------------------------------
        Computes the Fourier transform of the time series of the antennas in the 
        antenna array to compute the visibility spectra
        ----------------------------------------------------------------------------
        """
        
        if not parallel:
            for label in self.antennas:
                self.antennas[label].FX()
        elif parallel or (nproc is not None):
            if nproc is None:
                nproc = max(MP.cpu_count()-1, 1) 
            else:
                nproc = min(nproc, max(MP.cpu_count()-1, 1))
            pool = MP.Pool(processes=nproc)
            updated_antennas = pool.map(unwrap_antenna_FT, IT.izip(self.antennas.values()))
            pool.close()
            pool.join()

            for antenna in updated_antennas: 
                self.antennas[antenna.label] = antenna
            del updated_antennas
        
    ################################################################################# 

    def grid(self, uvspacing=0.5, uvpad=None, pow2=True, pol=None):
        
        """
        ----------------------------------------------------------------------------
        Routine to produce a grid based on the antenna array 

        Inputs:

        uvspacing   [Scalar] Positive value indicating the maximum uv-spacing
                    desirable at the lowest wavelength (max frequency). 
                    Default = 0.5

        xypad       [List] Padding to be applied around the antenna locations 
                    before forming a grid. List elements should be positive. If it 
                    is a one-element list, the element is applicable to both x and 
                    y axes. If list contains three or more elements, only the 
                    first two elements are considered one for each axis. 
                    Default = None.

        pow2        [Boolean] If set to True, the grid is forced to have a size a 
                    next power of 2 relative to the actual sie required. If False,
                    gridding is done with the appropriate size as determined by
                    uvspacing. Default = True.

        pol         [String] The polarization to be gridded. Can be set to 'P11', 
                    'P12', 'P21', or 'P22'. If set to None, gridding for all the
                    polarizations is performed. 
        ----------------------------------------------------------------------------
        """

        if self.f is None:
            self.f = self.antennas.itervalues().next().f

        if self.f0 is None:
            self.f0 = self.antennas.itervalues().next().f0

        wavelength = FCNST.c / self.f
        min_lambda = NP.abs(wavelength).min()

        # Change itervalues() to values() when porting to Python 3.x
        # May have to change *blc and *trc with zip(*blc) and zip(*trc) when using Python 3.x

        blc = [[self.antennas[label].blc[0,0], self.antennas[label].blc[0,1]] for label in self.antennas]
        trc = [[self.antennas[label].trc[0,0], self.antennas[label].trc[0,1]] for label in self.antennas]

        self.trc = NP.amax(NP.abs(NP.vstack((NP.asarray(blc), NP.asarray(trc)))), axis=0).ravel() / min_lambda
        self.blc = -1 * self.trc

        self.gridu, self.gridv = GRD.grid_2d([(self.blc[0], self.trc[0]), (self.blc[1], self.trc[1])], pad=uvpad, spacing=uvspacing, pow2=True)

        self.grid_blc = NP.asarray([self.gridu.min(), self.gridv.min()])
        self.grid_trc = NP.asarray([self.gridu.max(), self.gridv.max()])

        self.grid_ready = True

    ################################################################################# 

    def grid_convolve(self, pol=None, ants=None, unconvolve_existing=False,
                      normalize=False, method='NN', distNN=NP.inf, tol=None,
                      maxmatch=None, identical_antennas=True,
                      gridfunc_freq=None, mapping='weighted', wts_change=False,
                      parallel=False, nproc=None, pp_method='pool', verbose=True): 

        """
        ----------------------------------------------------------------------------
        Routine to project the complex illumination field pattern and the electric
        fields on the grid. It can operate on the entire antenna array 
        or incrementally project the electric fields and complex illumination field 
        patterns from specific antennas on to an already existing grid. (The
        latter is not implemented yet)

        Inputs:

        pol        [String] The polarization to be gridded. Can be set to 'P1' or 
                   'P2'. If set to None, gridding for all the polarizations is 
                   performed. Default = None

        ants       [instance of class AntennaArray, single instance or list 
                   of instances of class Antenna, or a dictionary holding 
                   instances of class Antenna] If a dictionary is provided, 
                   the keys should be the antenna labels and the values 
                   should be instances of class Antenna. If a list is 
                   provided, it should be a list of valid instances of class 
                   Antenna. These instance(s) of class Antenna will 
                   be merged to the existing grid contained in the instance of 
                   AntennaArray class. If ants is not provided (set to 
                   None), the gridding operations will be performed on the entire
                   set of antennas contained in the instance of class 
                   AntennaArray. Default = None.

        unconvolve_existing
                   [Boolean] Default = False. If set to True, the effects of
                   gridding convolution contributed by the antenna(s) 
                   specified will be undone before updating the antenna 
                   measurements on the grid, if the antenna(s) is/are 
                   already found to in the set of antennas held by the 
                   instance of AntennaArray. If False and if one or more 
                   antenna instances specified are already found to be held 
                   in the instance of class AntennaArray, the code will stop
                   raising an error indicating the gridding oepration cannot
                   proceed. 

        normalize  [Boolean] Default = False. If set to True, the gridded weights
                   are divided by the sum of weights so that the gridded weights 
                   add up to unity. (Need to work on normaliation)

        method     [string] The gridding method to be used in applying the 
                   antenna weights on to the antenna array grid. 
                   Accepted values are 'NN' (nearest neighbour - default), 'CS' 
                   (cubic spline), or 'BL' (Bi-linear). In case of applying grid 
                   weights by 'NN' method, an optional distance upper bound for 
                   the nearest neighbour can be provided in the parameter distNN 
                   to prune the search and make it efficient. Currently, only the
                   nearest neighbour method is operational.

        distNN     [scalar] A positive value indicating the upper bound on 
                   distance to the nearest neighbour in the gridding process. It 
                   has units of distance, the same units as the antenna 
                   attribute location and antenna array attribute gridx 
                   and gridy. Default is NP.inf (infinite distance). It will be 
                   internally converted to have same units as antenna 
                   attributes wtspos (units in number of wavelengths)

        maxmatch   [scalar] A positive value indicating maximum number of input 
                   locations in the antenna grid to be assigned. Default = None. 
                   If set to None, all the antenna array grid elements specified 
                   are assigned values for each antenna. For instance, to have 
                   only one antenna array grid element to be populated per 
                   antenna, use maxmatch=1. 

        tol        [scalar] If set, only lookup data with abs(val) > tol will be 
                   considered for nearest neighbour lookup. Default = None implies 
                   all lookup values will be considered for nearest neighbour 
                   determination. tol is to be interpreted as a minimum value 
                   considered as significant in the lookup table. 

        identical_antennas
                   [boolean] indicates if all antenna elements are to be
                   treated as identical. If True (default), they are identical
                   and their gridding kernels are identical. If False, they are
                   not identical and each one has its own gridding kernel.

        gridfunc_freq
                   [String scalar] If set to None (not provided) or to 'scale'
                   assumes that attribute wtspos is given for a
                   reference frequency which need to be scaled for the frequency
                   channels. Will be ignored if the number of elements of list 
                   in this attribute under the specific polarization are the same
                   as the number of frequency channels.

        mapping    [string] indicates the type of mapping between antenna locations
                   and the grid locations. Allowed values are 'sampled' and 
                   'weighted' (default). 'sampled' means only the antenna measurement 
                   closest ot a grid location contributes to that grid location, 
                   whereas, 'weighted' means that all the antennas contribute in
                   a weighted fashion to their nearest grid location. The former 
                   is faster but possibly discards antenna data whereas the latter
                   is slower but includes all data along with their weights.

        wts_change [boolean] indicates if weights and/or their lcoations have 
                   changed from the previous intergration or snapshot. 
                   Default=False means they have not changed. In such a case the 
                   antenna-to-grid mapping and grid illumination pattern do not 
                   have to be determined, and mapping and values from the previous 
                   snapshot can be used. If True, a new mapping has to be 
                   determined.

        parallel   [boolean] specifies if parallelization is to be invoked. 
                   False (default) means only serial processing

        nproc      [integer] specifies number of independent processes to spawn.
                   Default = None, means automatically determines the number of 
                   process cores in the system and use one less than that to 
                   avoid locking the system for other processes. Applies only 
                   if input parameter 'parallel' (see above) is set to True. 
                   If nproc is set to a value more than the number of process
                   cores in the system, it will be reset to number of process 
                   cores in the system minus one to avoid locking the system out 
                   for other processes

        pp_method  [string] specifies if the parallelization method is handled
                   automatically using multirocessing pool or managed manually
                   by individual processes and collecting results in a queue.
                   The former is specified by 'pool' (default) and the latter
                   by 'queue'. These are the two allowed values. The pool method 
                   has easier bookkeeping and can be fast if the computations 
                   not expected to be memory bound. The queue method is more
                   suited for memory bound processes but can be slower or 
                   inefficient in terms of CPU management.

        verbose    [boolean] If True, prints diagnostic and progress messages. 
                   If False (default), suppress printing such messages.
        ----------------------------------------------------------------------------
        """

        eps = 1.0e-10
        if pol is None:
            pol = ['P1', 'P2']
        elif not isinstance(pol, list):
            pol = [pol]

        if not self.grid_ready:
            self.grid()

        antpol = ['P1', 'P2']

        for apol in antpol:
            if apol in pol:

                if ants is not None:
    
                    if isinstance(ants, Antenna):
                        ants = [ants]
    
                    if isinstance(ants, (dict, AntennaArray)):
                        # Check if these antennas are new or old and compatible
                        for key in ants: 
                            if isinstance(ants[key], Antenna): # required if ants is a dictionary and not instance of AntennaArray
                                if key in self.antennas:
                                    if unconvolve_existing: # Effects on the grid of antennas already existing must be removed 
                                        if self.antennas[key]._gridinfo[apol]: # if gridding info is not empty
                                            for i in range(len(self.f)):
                                                self.grid_unconvolve(ants[key].label)
                                    else:
                                        raise KeyError('Antenna {0} already found to exist in the dictionary of antennas but cannot proceed grid_convolve() without unconvolving first.'.format(ants[key].label)) 
                                
                            else:
                                del ants[key] # remove the dictionary element since it is not an Antenna instance
                
                    if identical_antennas and (gridfunc_freq == 'scale'):
                        ant_dict = self.antenna_positions(pol=apol, flag=False, sort=True)
                        ant_xy = ant_dict['positions'][:,:2]
                        self.ordered_labels = ant_dict['labels']
                        n_ant = ant_xy.shape[0]

                        Ef_dict = self.get_E_fields(apol, flag=False, sort=True)
                        Ef = Ef_dict['E-fields'].astype(NP.complex64)

                        # Since antennas are identical, read from first antenna, since wtspos are scaled with frequency, read from first frequency channel
                        wtspos_xy = ants[0].wtspos[apol][0] * FCNST.c/self.f[0] 
                        wts = ants[0].wts[apol][0]
                        n_wts = wts.size

                        reflocs_xy = ant_xy[:,NP.newaxis,:] + wtspos_xy[NP.newaxis,:,:]
                        refwts_xy = wts.reshape(1,-1) * NP.ones((n_ant,1))

                        reflocs_xy = reflocs_xy.reshape(-1,ant_xy.shape[1])
                        refwts_xy = refwts_xy.reshape(-1,1).astype(NP.complex64)
                        reflocs_uv = reflocs_xy[:,NP.newaxis,:] * self.f.reshape(1,-1,1) / FCNST.c
                        refwts_uv = refwts_xy * NP.ones((1,self.f.size))
                        reflocs_uv = reflocs_uv.reshape(-1,ant_xy.shape[1])
                        refwts_uv = refwts_uv.reshape(-1,1).ravel()

                        inplocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                        ibind, nnval = LKP.lookup_1NN(reflocs_uv, refwts_uv, inplocs,
                                                      distance_ULIM=distNN*self.f.max()/FCNST.c,
                                                      remove_oob=True, tol=tol, maxmatch=maxmatch)[:2]
                        
                else:  
                    ant_dict = self.antenna_positions(pol=apol, flag=None, sort=True)
                    self.ordered_labels = ant_dict['labels']
                    ant_xy = ant_dict['positions'][:,:2] # n_ant x 2
                    n_ant = ant_xy.shape[0]

                    Ef_dict = self.get_E_fields(apol, flag=None, sort=True)
                    Ef = Ef_dict['E-fields'].astype(NP.complex64)  # n_ant x nchan
                    if Ef.shape[0] != n_ant:
                        raise ValueError('Encountered unexpected behavior. Need to debug.')
                    if verbose:
                        print 'Gathered antenna data for gridding convolution for timestamp {0}'.format(self.timestamp)

                    if wts_change or (not self.grid_mapper[apol]['labels']):
    
                        if gridfunc_freq == 'scale':
                            if identical_antennas:
    
                                wts_tol = 1e-6
    
                                # Since antennas are identical, read from first antenna, since wtspos are scaled with frequency, read from first frequency channel
    
                                wtspos_xy = self.antennas.itervalues().next().wtspos[apol][0] * FCNST.c/self.f[0] 
                                wts = self.antennas.itervalues().next().wts[apol][0].astype(NP.complex64)
                                wtspos_xy = wtspos_xy[NP.abs(wts) >= wts_tol, :]
                                wts = wts[NP.abs(wts) >= wts_tol]
                                n_wts = wts.size
        
                                reflocs_xy = ant_xy[:,NP.newaxis,:] + wtspos_xy[NP.newaxis,:,:] # n_ant x n_wts x 2 
                                refwts = wts.reshape(1,-1) * NP.ones((n_ant,1))  # n_ant x n_wts
                            else:
                                for i,label in enumerate(self.ordered_labels):
                                    ant_wtspos = self.antennas[label].wtspos[apol][0]
                                    ant_wts = self.antennas[label].wts[apol][0].astype(NP.complex64)
                                    if i == 0:
                                        wtspos = ant_wtspos[NP.newaxis,:,:] # 1 x n_wts x 2
                                        refwts = ant_wts.reshape(1,-1) # 1 x n_wts
                                    else:
                                        wtspos = NP.vstack((wtspos, ant_wtspos[NP.newaxis,:,:])) # n_ant x n_wts x 2
                                        refwts = NP.vstack((refwts, ant_wts.reshape(1,-1))) # n_ant x n_wts
                                    reflocs_xy = ant_xy[:,NP.newaxis,:] + wtspos * FCNST.c/self.f[0] # n_ant x n_wts x 2
                                    
                            reflocs_xy = reflocs_xy.reshape(-1,ant_xy.shape[1])  # (n_ant x n_wts) x 2
                            refwts = refwts.ravel()
                            self.grid_mapper[apol]['refwts'] = NP.copy(refwts.ravel()) # (n_ant x n_wts)
                            
                        else: # Weights do not scale with frequency (needs serious development)
                            pass
                            
                        gridlocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                        contributed_ant_grid_Ef = None

                        if parallel:    # Use parallelization over frequency to determine gridding convolution
                            if nproc is None:
                                nproc = max(MP.cpu_count()-1, 1) 
                            else:
                                nproc = min(nproc, max(MP.cpu_count()-1, 1))
                            
                            if pp_method == 'queue':  ## Use MP.Queue(): useful for memory intensive parallelizing but can be slow
                                job_chunk_begin = range(0,self.f.size,nproc)
                                if verbose:
                                    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} job chunks '.format(len(job_chunk_begin)), PGB.ETA()], maxval=len(job_chunk_begin)).start()

                                for ijob, job_start in enumerate(job_chunk_begin):
                                    pjobs = []
                                    out_q = MP.Queue()
                                    for job_ind in xrange(job_start, min(job_start+nproc, self.f.size)):    # Start the processes and store outputs in the queue
                                        if mapping == 'weighted':
                                            pjob = MP.Process(target=LKP.find_1NN_pp, args=(gridlocs, reflocs_xy * self.f[job_ind]/FCNST.c, job_ind, out_q, distNN*self.f.max()/FCNST.c, True), name='process-{0:0d}-channel-{1:0d}'.format(job_ind-job_start, job_ind))
                                        else:
                                            pjob = MP.Process(target=LKP.find_1NN_pp, args=(reflocs_xy * self.f[job_ind]/FCNST.c, gridlocs, job_ind, out_q, distNN*self.f.max()/FCNST.c, True), name='process-{0:0d}-channel-{1:0d}'.format(job_ind-job_start, job_ind))             
                                        pjob.start()
                                        pjobs.append(pjob)
                                   
                                    for p in xrange(len(pjobs)):   # Unpack the queue output
                                        outdict = out_q.get()
                                        chan = outdict.keys()[0]
                                        if mapping == 'weighted':
                                            refind, gridind = outdict[chan]['inpind'], outdict[chan]['refind']
                                        else:
                                            gridind, refind = outdict[chan]['inpind'], outdict[chan]['refind']                                            
                                        self.grid_mapper[apol]['refind'] += [refind]
                                        self.grid_mapper[apol]['gridind'] += [gridind]

                                        ant_ind, lkp_ind = NP.unravel_index(refind, (n_ant, n_wts))
                                        self.grid_mapper[apol]['ant']['ind_freq'] += [ant_ind]
                                        gridind_unraveled = NP.unravel_index(gridind, self.gridu.shape) + (chan+NP.zeros(gridind.size,dtype=int),)
                                        gridind_raveled = NP.ravel_multi_index(gridind_unraveled, self.gridu.shape+(self.f.size,))

                                        if self.grid_mapper[apol]['ant']['ind_all'] is None:
                                            self.grid_mapper[apol]['ant']['ind_all'] = NP.copy(ant_ind)
                                            self.grid_mapper[apol]['ant']['illumination'] = refwts[refind]
                                            contributed_ant_grid_Ef = refwts[refind] * Ef[ant_ind,chan]
                                            self.grid_mapper[apol]['grid']['ind_all'] = NP.copy(gridind_raveled)
                                        else:
                                            self.grid_mapper[apol]['ant']['ind_all'] = NP.append(self.grid_mapper[apol]['ant']['ind_all'], ant_ind)
                                            self.grid_mapper[apol]['ant']['illumination'] = NP.append(self.grid_mapper[apol]['ant']['illumination'], refwts[refind])
                                            contributed_ant_grid_Ef = NP.append(contributed_ant_grid_Ef, refwts[refind] * Ef[ant_ind,chan])
                                            self.grid_mapper[apol]['grid']['ind_all'] = NP.append(self.grid_mapper[apol]['grid']['ind_all'], gridind_raveled)
    
                                    for pjob in pjobs:
                                        pjob.join()

                                    del out_q

                                    if verbose:
                                        progress.update(ijob+1)
                                if verbose:
                                    progress.finish()

                            elif pp_method == 'pool':   ## Using MP.Pool.map(): Can be faster if parallelizing is not memory intensive
                                list_of_gridlocs = [gridlocs] * self.f.size
                                list_of_reflocs = [reflocs_xy * f/FCNST.c for f in self.f]
                                list_of_dist_NN = [distNN*self.f.max()/FCNST.c] * self.f.size
                                list_of_remove_oob = [True] * self.f.size

                                pool = MP.Pool(processes=nproc)
                                if mapping == 'weighted':
                                    list_of_NNout = pool.map(find_1NN_arg_splitter, IT.izip(list_of_gridlocs, list_of_reflocs, list_of_dist_NN, list_of_remove_oob))
                                else:
                                    list_of_NNout = pool.map(find_1NN_arg_splitter, IT.izip(list_of_reflocs, list_of_gridlocs, list_of_dist_NN, list_of_remove_oob))

                                pool.close()
                                pool.join()

                                for chan, NNout in enumerate(list_of_NNout):    # Unpack the pool output
                                    if mapping == 'weighted':
                                        refind, gridind = NNout[0], NNout[1]
                                    else:
                                        gridind, refind = NNout[0], NNout[1]

                                    self.grid_mapper[apol]['refind'] += [refind]
                                    self.grid_mapper[apol]['gridind'] += [gridind]

                                    ant_ind, lkp_ind = NP.unravel_index(refind, (n_ant, n_wts))
                                    self.grid_mapper[apol]['ant']['ind_freq'] += [ant_ind]
                                    gridind_unraveled = NP.unravel_index(gridind, self.gridu.shape) + (chan+NP.zeros(gridind.size,dtype=int),)
                                    gridind_raveled = NP.ravel_multi_index(gridind_unraveled, self.gridu.shape+(self.f.size,))

                                    if chan == 0:
                                        self.grid_mapper[apol]['ant']['ind_all'] = NP.copy(ant_ind)
                                        self.grid_mapper[apol]['ant']['illumination'] = refwts[refind]
                                        contributed_ant_grid_Ef = refwts[refind] * Ef[ant_ind,chan]
                                        self.grid_mapper[apol]['grid']['ind_all'] = NP.copy(gridind_raveled)
                                    else:
                                        self.grid_mapper[apol]['ant']['ind_all'] = NP.append(self.grid_mapper[apol]['ant']['ind_all'], ant_ind)
                                        self.grid_mapper[apol]['ant']['illumination'] = NP.append(self.grid_mapper[apol]['ant']['illumination'], refwts[refind])
                                        contributed_ant_grid_Ef = NP.append(contributed_ant_grid_Ef, refwts[refind] * Ef[ant_ind,chan])
                                        self.grid_mapper[apol]['grid']['ind_all'] = NP.append(self.grid_mapper[apol]['grid']['ind_all'], gridind_raveled)

                            else:
                                raise ValueError('Parallel processing method specified by input parameter ppmethod has to be "pool" or "queue"')
                            
                        else:    # Use serial processing over frequency to determine gridding convolution

                            if verbose:
                                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Frequency channels '.format(self.f.size), PGB.ETA()], maxval=self.f.size).start()
    
                            for i in xrange(self.f.size):
                                if mapping == 'weighted':
                                    refind, gridind = LKP.find_1NN(gridlocs, reflocs_xy * self.f[i]/FCNST.c, 
                                                                  distance_ULIM=distNN*self.f.max()/FCNST.c,
                                                                  remove_oob=True)[:2]
                                else:
                                    gridind, refind = LKP.find_1NN(reflocs_xy * self.f[i]/FCNST.c, gridlocs,
                                                                  distance_ULIM=distNN*self.f.max()/FCNST.c,
                                                                  remove_oob=True)[:2]
                                
                                self.grid_mapper[apol]['refind'] += [refind]
                                self.grid_mapper[apol]['gridind'] += [gridind]
    
                                ant_ind, lkp_ind = NP.unravel_index(refind, (n_ant, n_wts))
                                self.grid_mapper[apol]['ant']['ind_freq'] += [ant_ind]
                                gridind_unraveled = NP.unravel_index(gridind, self.gridu.shape) + (i+NP.zeros(gridind.size,dtype=int),)
                                gridind_raveled = NP.ravel_multi_index(gridind_unraveled, self.gridu.shape+(self.f.size,))
                                if i == 0:
                                    self.grid_mapper[apol]['ant']['ind_all'] = NP.copy(ant_ind)
                                    self.grid_mapper[apol]['ant']['illumination'] = refwts[refind]
                                    contributed_ant_grid_Ef = refwts[refind] * Ef[ant_ind,i]
                                    self.grid_mapper[apol]['grid']['ind_all'] = NP.copy(gridind_raveled)
                                else:
                                    self.grid_mapper[apol]['ant']['ind_all'] = NP.append(self.grid_mapper[apol]['ant']['ind_all'], ant_ind)
                                    self.grid_mapper[apol]['ant']['illumination'] = NP.append(self.grid_mapper[apol]['ant']['illumination'], refwts[refind])
                                    contributed_ant_grid_Ef = NP.append(contributed_ant_grid_Ef, refwts[refind] * Ef[ant_ind,i])
                                    self.grid_mapper[apol]['grid']['ind_all'] = NP.append(self.grid_mapper[apol]['grid']['ind_all'], gridind_raveled)
    
                                if verbose:
                                    progress.update(i+1)
                            if verbose:
                                progress.finish()
                                
                        self.grid_mapper[apol]['ant']['uniq_ind_all'] = NP.unique(self.grid_mapper[apol]['ant']['ind_all'])
                        self.grid_mapper[apol]['ant']['rev_ind_all'] = OPS.binned_statistic(self.grid_mapper[apol]['ant']['ind_all'], statistic='count', bins=NP.append(self.grid_mapper[apol]['ant']['uniq_ind_all'], self.grid_mapper[apol]['ant']['uniq_ind_all'].max()+1))[3]

                        if parallel and (mapping == 'weighted'):    # Use parallel processing over antennas to determine antenna-grid mapping of gridded aperture illumination and electric fields

                            if nproc is None:
                                nproc = max(MP.cpu_count()-1, 1) 
                            else:
                                nproc = min(nproc, max(MP.cpu_count()-1, 1))

                            if pp_method == 'queue':  ## Use MP.Queue(): useful for memory intensive parallelizing but can be slow

                                num_ant = self.grid_mapper[apol]['ant']['uniq_ind_all'].size
                                job_chunk_begin = range(0,num_ant,nproc)
                                if verbose:
                                    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} job chunks '.format(len(job_chunk_begin)), PGB.ETA()], maxval=len(job_chunk_begin)).start()

                                for ijob, job_start in enumerate(job_chunk_begin):
                                    pjobs1 = []
                                    pjobs2 = []
                                    out_q1 = MP.Queue()
                                    out_q2 = MP.Queue()
    
                                    for job_ind in xrange(job_start, min(job_start+nproc, num_ant)):   # Start the parallel processes and store the output in the queue
                                        label = self.ordered_labels[self.grid_mapper[apol]['ant']['uniq_ind_all'][job_ind]]
    
                                        if self.grid_mapper[apol]['ant']['rev_ind_all'][job_ind] < self.grid_mapper[apol]['ant']['rev_ind_all'][job_ind+1]:
        
                                            self.grid_mapper[apol]['labels'][label] = {}
                                            self.grid_mapper[apol]['labels'][label]['flag'] = self.antennas[label].antpol.flag[apol]
        
                                            select_ant_ind = self.grid_mapper[apol]['ant']['rev_ind_all'][self.grid_mapper[apol]['ant']['rev_ind_all'][job_ind]:self.grid_mapper[apol]['ant']['rev_ind_all'][job_ind+1]]
                                            gridind_raveled_around_ant = self.grid_mapper[apol]['grid']['ind_all'][select_ant_ind]
                                            uniq_gridind_raveled_around_ant = NP.unique(gridind_raveled_around_ant)
                                            self.grid_mapper[apol]['labels'][label]['gridind'] = uniq_gridind_raveled_around_ant
                                            pjob1 = MP.Process(target=antenna_grid_mapper, args=(gridind_raveled_around_ant, contributed_ant_grid_Ef[select_ant_ind], NP.append(uniq_gridind_raveled_around_ant, uniq_gridind_raveled_around_ant.max()+1), label, out_q1), name='process-{0:0d}-{1}-E-field'.format(job_ind, label))
                                            pjob2 = MP.Process(target=antenna_grid_mapper, args=(gridind_raveled_around_ant, self.grid_mapper[apol]['ant']['illumination'][select_ant_ind], NP.append(uniq_gridind_raveled_around_ant, uniq_gridind_raveled_around_ant.max()+1), label, out_q2), name='process-{0:0d}-{1}-illumination'.format(job_ind, label))
                                            pjob1.start()
                                            pjob2.start()
                                            pjobs1.append(pjob1)
                                            pjobs2.append(pjob2)

                                    for p in xrange(len(pjobs1)):    # Unpack the E-fields and aperture illumination information from the pool output
                                        outdict = out_q1.get()
                                        label = outdict.keys()[0]
                                        self.grid_mapper[apol]['labels'][label]['Ef'] = outdict[label]
                                        outdict = out_q2.get()
                                        label = outdict.keys()[0]
                                        self.grid_mapper[apol]['labels'][label]['illumination'] = outdict[label]
    
                                    for pjob in pjobs1:
                                        pjob1.join()
                                    for pjob in pjobs2:
                                        pjob2.join()
        
                                    del out_q1, out_q2
                                    
                                    if verbose:
                                        progress.update(ijob+1)
                                if verbose:
                                    progress.finish()
                                    
                            elif pp_method == 'pool':    ## Using MP.Pool.map(): Can be faster if parallelizing is not memory intensive

                                list_of_gridind_raveled_around_ant = []
                                list_of_ant_grid_values = []
                                list_of_ant_Ef_contribution = []
                                list_of_ant_illumination = []
                                list_of_uniq_gridind_raveled_around_ant = []
                                list_of_ant_labels = []
    
                                for j in xrange(self.grid_mapper[apol]['ant']['uniq_ind_all'].size): # re-determine gridded electric fields due to each antenna
    
                                    label = self.ordered_labels[self.grid_mapper[apol]['ant']['uniq_ind_all'][j]]
                                    if self.grid_mapper[apol]['ant']['rev_ind_all'][j] < self.grid_mapper[apol]['ant']['rev_ind_all'][j+1]:
    
                                        self.grid_mapper[apol]['labels'][label] = {}
                                        self.grid_mapper[apol]['labels'][label]['flag'] = self.antennas[label].antpol.flag[apol]
    
                                        select_ant_ind = self.grid_mapper[apol]['ant']['rev_ind_all'][self.grid_mapper[apol]['ant']['rev_ind_all'][j]:self.grid_mapper[apol]['ant']['rev_ind_all'][j+1]]
                                        gridind_raveled_around_ant = self.grid_mapper[apol]['grid']['ind_all'][select_ant_ind]
                                        uniq_gridind_raveled_around_ant = NP.unique(gridind_raveled_around_ant)
                                        self.grid_mapper[apol]['labels'][label]['gridind'] = uniq_gridind_raveled_around_ant
                                        
                                        list_of_ant_labels += [label]
                                        list_of_gridind_raveled_around_ant += [gridind_raveled_around_ant]
                                        list_of_uniq_gridind_raveled_around_ant += [NP.append(uniq_gridind_raveled_around_ant, uniq_gridind_raveled_around_ant.max()+1)]
                                        list_of_ant_Ef_contribution += [contributed_ant_grid_Ef[select_ant_ind]]
                                        list_of_ant_illumination += [self.grid_mapper[apol]['ant']['illumination'][select_ant_ind]]
    
                                pool = MP.Pool(processes=nproc)
                                list_of_ant_grid_values = pool.map(antenna_grid_mapping_arg_splitter, IT.izip(list_of_gridind_raveled_around_ant, list_of_ant_Ef_contribution, list_of_uniq_gridind_raveled_around_ant))
                                pool.close()
                                pool.join()
    
                                for label,grid_values in IT.izip(list_of_ant_labels, list_of_ant_grid_values):    # Unpack the gridded visibility information from the pool output
                                    self.grid_mapper[apol]['labels'][label]['Ef'] = grid_values
    
                                if nproc is None:
                                    pool = MP.Pool(processes=nproc)
                                else:
                                    pool = MP.Pool()
                                list_of_ant_grid_values = pool.map(antenna_grid_mapping_arg_splitter, IT.izip(list_of_gridind_raveled_around_ant, list_of_ant_illumination, list_of_uniq_gridind_raveled_around_ant))
                                pool.close()
                                pool.join()
    
                                for label,grid_values in IT.izip(list_of_ant_labels, list_of_ant_grid_values):    # Unpack the gridded visibility and aperture illumination information from the pool output
                                    self.grid_mapper[apol]['labels'][label]['illumination'] = grid_values
                                
                                del list_of_ant_grid_values, list_of_gridind_raveled_around_ant, list_of_ant_Ef_contribution, list_of_ant_illumination, list_of_uniq_gridind_raveled_around_ant, list_of_ant_labels

                            else:
                                raise ValueError('Parallel processing method specified by input parameter ppmethod has to be "pool" or "queue"')

                        else:    # Use serial processing over antennas to determine antenna-grid mapping of gridded aperture illumination and electric fields

                            if verbose:
                                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(self.grid_mapper[apol]['ant']['uniq_ind_all'].size), PGB.ETA()], maxval=self.grid_mapper[apol]['ant']['uniq_ind_all'].size).start()

                            for j in xrange(self.grid_mapper[apol]['ant']['uniq_ind_all'].size):
                                label = self.ordered_labels[self.grid_mapper[apol]['ant']['uniq_ind_all'][j]]
                                if self.grid_mapper[apol]['ant']['rev_ind_all'][j] < self.grid_mapper[apol]['ant']['rev_ind_all'][j+1]:
                                    select_ant_ind = self.grid_mapper[apol]['ant']['rev_ind_all'][self.grid_mapper[apol]['ant']['rev_ind_all'][j]:self.grid_mapper[apol]['ant']['rev_ind_all'][j+1]]
                                    self.grid_mapper[apol]['labels'][label] = {}
                                    self.grid_mapper[apol]['labels'][label]['flag'] = self.antennas[label].antpol.flag[apol]
                                    if mapping == 'weighted':
                                        gridind_raveled_around_ant = self.grid_mapper[apol]['grid']['ind_all'][select_ant_ind]
                                        uniq_gridind_raveled_around_ant = NP.unique(gridind_raveled_around_ant)
                                        self.grid_mapper[apol]['labels'][label]['gridind'] = uniq_gridind_raveled_around_ant
    
                                        self.grid_mapper[apol]['labels'][label]['Ef'] = OPS.binned_statistic(gridind_raveled_around_ant, contributed_ant_grid_Ef[select_ant_ind].real, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_ant, uniq_gridind_raveled_around_ant.max()+1))[0]
                                        self.grid_mapper[apol]['labels'][label]['Ef'] = self.grid_mapper[apol]['labels'][label]['Ef'].astype(NP.complex64)
                                        self.grid_mapper[apol]['labels'][label]['Ef'] += 1j * OPS.binned_statistic(gridind_raveled_around_ant, contributed_ant_grid_Ef[select_ant_ind].imag, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_ant, uniq_gridind_raveled_around_ant.max()+1))[0]
    
                                        self.grid_mapper[apol]['labels'][label]['illumination'] = OPS.binned_statistic(gridind_raveled_around_ant, self.grid_mapper[apol]['ant']['illumination'][select_ant_ind].real, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_ant, uniq_gridind_raveled_around_ant.max()+1))[0]
                                        self.grid_mapper[apol]['labels'][label]['illumination'] = self.grid_mapper[apol]['labels'][label]['illumination'].astype(NP.complex64)
                                        self.grid_mapper[apol]['labels'][label]['illumination'] += 1j * OPS.binned_statistic(gridind_raveled_around_ant, self.grid_mapper[apol]['ant']['illumination'][select_ant_ind].imag, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_ant, uniq_gridind_raveled_around_ant.max()+1))[0]
    
                                    else:
                                        self.grid_mapper[apol]['labels'][label]['gridind'] = self.grid_mapper[apol]['grid']['ind_all'][select_ant_ind]
                                        self.grid_mapper[apol]['labels'][label]['Ef'] = contributed_ant_grid_Ef[select_ant_ind]
                                        self.grid_mapper[apol]['labels'][label]['illumination'] = self.grid_mapper[apol]['ant']['illumination'][select_ant_ind]
                                        
                                if verbose:
                                    progress.update(j+1)
                            if verbose:
                                progress.finish()
                            
                    else: # Only re-determine gridded electric fields

                        if verbose:
                            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Frequency channels '.format(self.f.size), PGB.ETA()], maxval=self.f.size).start()

                        for i in xrange(self.f.size): # Only re-estimate electric fields contributed by antennas
                            ant_refwts = self.grid_mapper[apol]['refwts'][self.grid_mapper[apol]['refind'][i]]
                            ant_Ef = Ef[self.grid_mapper[apol]['ant']['ind_freq'][i],i]
                            if i == 0:
                                contributed_ant_grid_Ef = ant_refwts * ant_Ef
                            else:
                                contributed_ant_grid_Ef = NP.append(contributed_ant_grid_Ef, ant_refwts * ant_Ef)

                            if verbose:
                                progress.update(i+1)
                        if verbose:
                            progress.finish()

                        if parallel and (mapping == 'weighted'):    # Use parallel processing

                            if nproc is None:
                                nproc = max(MP.cpu_count()-1, 1) 
                            else:
                                nproc = min(nproc, max(MP.cpu_count()-1, 1))                            

                            if pp_method == 'queue':   ## Use MP.Queue(): useful for memory intensive parallelizing but can be slow

                                num_ant = self.grid_mapper[apol]['ant']['uniq_ind_all'].size
                                job_chunk_begin = range(0,num_ant,nproc)
                                if verbose:
                                    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} job chunks '.format(len(job_chunk_begin)), PGB.ETA()], maxval=len(job_chunk_begin)).start()

                                for ijob, job_start in enumerate(job_chunk_begin):
                                    pjobs = []
                                    out_q = MP.Queue()
    
                                    for job_ind in xrange(job_start, min(job_start+nproc, num_ant)):    # Start the parallel processes and store the outputs in a queue
                                        label = self.ordered_labels[self.grid_mapper[apol]['ant']['uniq_ind_all'][job_ind]]
    
                                        if self.grid_mapper[apol]['ant']['rev_ind_all'][job_ind] < self.grid_mapper[apol]['ant']['rev_ind_all'][job_ind+1]:
        
                                            select_ant_ind = self.grid_mapper[apol]['ant']['rev_ind_all'][self.grid_mapper[apol]['ant']['rev_ind_all'][job_ind]:self.grid_mapper[apol]['ant']['rev_ind_all'][job_ind+1]]
                                            gridind_raveled_around_ant = self.grid_mapper[apol]['grid']['ind_all'][select_ant_ind]
                                            uniq_gridind_raveled_around_ant = self.grid_mapper[apol]['labels'][label]['gridind']
                                            pjob = MP.Process(target=antenna_grid_mapper, args=(gridind_raveled_around_ant, contributed_ant_grid_Ef[select_ant_ind], NP.append(uniq_gridind_raveled_around_ant, uniq_gridind_raveled_around_ant.max()+1), label, out_q), name='process-{0:0d}-{1}-E-field'.format(job_ind, label))
    
                                            pjob.start()
                                            pjobs.append(pjob)
    
                                    for p in xrange(len(pjobs)):    # Unpack the gridded visibility information from the queue
                                        outdict = out_q.get()
                                        label = outdict.keys()[0]
                                        self.grid_mapper[apol]['labels'][label]['Ef'] = outdict[label]
    
                                    for pjob in pjobs:
                                        pjob.join()
        
                                    del out_q
                                    
                                    if verbose:
                                        progress.update(ijob+1)
                                if verbose:
                                    progress.finish()

                            else:    ## Use MP.Pool.map(): Can be faster if parallelizing is not memory intensive

                                list_of_gridind_raveled_around_ant = []
                                list_of_ant_Ef_contribution = []
                                list_of_uniq_gridind_raveled_around_ant = []
                                list_of_ant_labels = []
                                for j in xrange(self.grid_mapper[apol]['ant']['uniq_ind_all'].size): # re-determine gridded electric fields due to each antenna
                                    if self.grid_mapper[apol]['ant']['rev_ind_all'][j] < self.grid_mapper[apol]['ant']['rev_ind_all'][j+1]:
                                        select_ant_ind = self.grid_mapper[apol]['ant']['rev_ind_all'][self.grid_mapper[apol]['ant']['rev_ind_all'][j]:self.grid_mapper[apol]['ant']['rev_ind_all'][j+1]]
                                        label = self.ordered_labels[self.grid_mapper[apol]['ant']['uniq_ind_all'][j]]
                                        gridind_raveled_around_ant = self.grid_mapper[apol]['grid']['ind_all'][select_ant_ind]
                                        uniq_gridind_raveled_around_ant = NP.unique(gridind_raveled_around_ant)
                                        list_of_ant_labels += [label]
                                        list_of_gridind_raveled_around_ant += [gridind_raveled_around_ant]
                                        list_of_uniq_gridind_raveled_around_ant += [NP.append(uniq_gridind_raveled_around_ant, uniq_gridind_raveled_around_ant.max()+1)]
                                        list_of_ant_Ef_contribution += [contributed_ant_grid_Ef[select_ant_ind]]
                                if nproc is None:
                                    nproc = max(MP.cpu_count()-1, 1) 
                                else:
                                    nproc = min(nproc, max(MP.cpu_count()-1, 1))
                                pool = MP.Pool(processes=nproc)
                                list_of_grid_Ef = pool.map(antenna_grid_mapping_arg_splitter, IT.izip(list_of_gridind_raveled_around_ant, list_of_ant_Ef_contribution, list_of_uniq_gridind_raveled_around_ant))
                                pool.close()
                                pool.join()
    
                                for label,grid_Ef in IT.izip(list_of_ant_labels, list_of_grid_Ef):    # Unpack the gridded visibility information from the pool output
                                    self.grid_mapper[apol]['labels'][label]['Ef'] = grid_Ef
                                
                                del list_of_gridind_raveled_around_ant, list_of_grid_Ef, list_of_ant_Ef_contribution, list_of_uniq_gridind_raveled_around_ant, list_of_ant_labels

                        else:          # use serial processing
                            if verbose:
                                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(self.grid_mapper[apol]['ant']['uniq_ind_all'].size), PGB.ETA()], maxval=self.grid_mapper[apol]['ant']['uniq_ind_all'].size).start()

                            for j in xrange(self.grid_mapper[apol]['ant']['uniq_ind_all'].size): # re-determine gridded electric fields due to each antenna
                                if self.grid_mapper[apol]['ant']['rev_ind_all'][j] < self.grid_mapper[apol]['ant']['rev_ind_all'][j+1]:
                                    select_ant_ind = self.grid_mapper[apol]['ant']['rev_ind_all'][self.grid_mapper[apol]['ant']['rev_ind_all'][j]:self.grid_mapper[apol]['ant']['rev_ind_all'][j+1]]
                                    label = self.ordered_labels[self.grid_mapper[apol]['ant']['uniq_ind_all'][j]]
                                    self.grid_mapper[apol]['labels'][label]['Ef'] = {}
                                    if mapping == 'weighted':
                                        gridind_raveled_around_ant = self.grid_mapper[apol]['grid']['ind_all'][select_ant_ind]
                                        uniq_gridind_raveled_around_ant = self.grid_mapper[apol]['labels'][label]['gridind']
                                        # uniq_gridind_raveled_around_ant = NP.unique(gridind_raveled_around_ant)
                                        self.grid_mapper[apol]['labels'][label]['Ef'] = OPS.binned_statistic(gridind_raveled_around_ant, contributed_ant_grid_Ef[select_ant_ind].real, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_ant, uniq_gridind_raveled_around_ant.max()+1))[0]
                                        self.grid_mapper[apol]['labels'][label]['Ef'] = self.grid_mapper[apol]['labels'][label]['Ef'].astype(NP.complex64)
                                        self.grid_mapper[apol]['labels'][label]['Ef'] += 1j * OPS.binned_statistic(gridind_raveled_around_ant, contributed_ant_grid_Ef[select_ant_ind].imag, statistic='sum', bins=NP.append(uniq_gridind_raveled_around_ant, uniq_gridind_raveled_around_ant.max()+1))[0]
                                    else:
                                        self.grid_mapper[apol]['labels'][label]['Ef'] = contributed_ant_grid_Ef[select_ant_ind]
                                if verbose:
                                    progress.update(j+1)
                            if verbose:
                                progress.finish()

    ################################################################################# 

    def make_grid_cube(self, pol=None, verbose=True):

        """
        ----------------------------------------------------------------------------
        Constructs the grid of complex field illumination and electric fields using 
        the gridding information determined for every antenna. Flags are taken
        into account while constructing this grid.

        Inputs:

        pol     [String] The polarization to be gridded. Can be set to 'P1' or 'P2'.
                If set to None, gridding for all the polarizations is performed. 
                Default = None
        
        verbose [boolean] If True, prints diagnostic and progress messages. 
                If False (default), suppress printing such messages.
        ----------------------------------------------------------------------------
        """

        if pol is None:
            pol = ['P1', 'P2']

        pol = NP.unique(NP.asarray(pol))
        
        for apol in pol:

            if verbose:
                print 'Gridding aperture illumination and electric fields for polarization {0} ...'.format(apol)

            if apol not in ['P1', 'P2']:
                raise ValueError('Invalid specification for input parameter pol')

            if apol not in self._ant_contribution:
                raise KeyError('Key {0} not found in attribute _ant_contribution'.format(apol))
    
            self.grid_illumination[apol] = NP.zeros((self.gridu.shape + (self.f.size,)), dtype=NP.complex_)
            self.grid_Ef[apol] = NP.zeros((self.gridu.shape + (self.f.size,)), dtype=NP.complex_)
    
            labels = self.grid_mapper[apol]['labels'].keys()
            if verbose:
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(len(labels)), PGB.ETA()], maxval=len(labels)).start()
            loopcount = 0
            num_unflagged = 0
            # while loopcount < len(labels):
            #     antinfo = self.grid_mapper[apol]['labels'].itervalues().next()
            for antinfo in self.grid_mapper[apol]['labels'].itervalues():
                if not antinfo['flag']:
                    num_unflagged += 1
                    gridind_unraveled = NP.unravel_index(antinfo['gridind'], self.gridu.shape+(self.f.size,))
                    self.grid_illumination[apol][gridind_unraveled] += antinfo['illumination']
                    self.grid_Ef[apol][gridind_unraveled] += antinfo['Ef']

                progress.update(loopcount+1)
                loopcount += 1
            progress.finish()
                
            if verbose:
                print 'Gridded aperture illumination and electric fields for polarization {0} from {1:0d} unflagged contributing antennas'.format(apol, num_unflagged)

    ############################################################################ 

    def update_flags(self, dictflags=None):

        """
        ----------------------------------------------------------------------------
        Updates all flags in the antenna array followed by any flags that
        need overriding through inputs of specific flag information

        Inputs:

        dictflags  [dictionary] contains flag information overriding after default
                   flag updates are determined. Antenna based flags are given as 
                   further dictionaries with each under under a key which is the
                   same as the antenna label. Flags for each antenna are
                   specified as a dictionary holding boolean flags for each of the 
                   two polarizations which are stored under keys 'P1' and 'P2'. 
                   An absent key just means it is not a part of the update. Flag 
                   information under each antenna must be of same type as input 
                   parameter flags in member function update_flags() of class 
                   PolInfo
        ----------------------------------------------------------------------------
        """

        for label in self.antennas:
            self.antennas[label].update_flags()

        if dictflags is not None:
            if not isinstance(dictflags, dict):
                raise TypeError('Input parameter dictflags must be a dictionary')
            
            for label in dictflags:
                if label in self.antennas:
                    self.antennas[label].antpol.update_flags(flags=dictflags[label])

    ############################################################################

    def update(self, updates=None, parallel=False, nproc=None, verbose=False):

        """
        -------------------------------------------------------------------------
        Updates the antenna array instance with newer attribute values. Can also 
        be used to add and/or remove antennas with/without affecting the existing
        grid.

        Inputs:

        updates     [Dictionary] Consists of information updates under the
                    following principal keys:
                    'antenna_array': Consists of updates for the AntennaArray
                                instance. This is a dictionary which consists of
                                the following keys:
                                'timestamp'   Unique identifier of the time 
                                              series. It is optional to set this 
                                              to a scalar. If not given, no 
                                              change is made to the existing
                                              timestamp attribute
                                'do_grid'     [boolean] If set to True, create or
                                              recreate a grid. To be specified 
                                              when the antenna locations are
                                              updated.
                    'antennas': Holds a list of dictionaries consisting of 
                                updates for individual antennas. Each element 
                                in the list contains update for one antenna. 
                                For each of these dictionaries, one of the keys 
                                is 'label' which indicates an antenna label. If 
                                absent, the code execution stops by throwing an 
                                exception. The other optional keys and the 
                                information they hold are listed below:
                                'action'      [String scalar] Indicates the type 
                                              of update operation. 'add' adds the 
                                              Antenna instance to the 
                                              AntennaArray instance. 'remove' 
                                              removes the antenna from the
                                              antenna array instance. 'modify'
                                              modifies the antenna attributes in 
                                              the antenna array instance. This 
                                              key has to be set. No default.
                                'grid_action' [Boolean] If set to True, will 
                                              apply the grdding operations 
                                              (grid(), grid_convolve(), and 
                                              grid_unconvolve()) appropriately 
                                              according to the value of the 
                                              'action' key. If set to None or 
                                              False, gridding effects will remain
                                              unchanged. Default=None(=False).
                                'antenna'     [instance of class Antenna] Updated 
                                              Antenna class instance. Can work 
                                              for action key 'remove' even if not 
                                              set (=None) or set to an empty 
                                              string '' as long as 'label' key is 
                                              specified. 
                                'gridpol'     [Optional. String scalar] Initiates 
                                              the specified action on 
                                              polarization 'P1' or 'P2'. Can be 
                                              set to 'P1' or 'P2'. If not 
                                              provided (=None), then the 
                                              specified action applies to both
                                              polarizations. Default = None.
                                'Et'          [Optional. Dictionary] Complex 
                                              Electric field time series under
                                              two polarizations which are under
                                              keys 'P1' and 'P2'. Is used only 
                                              if set and if 'action' key value 
                                              is set to 'modify'. 
                                              Default = None.
                                't'           [Optional. Numpy array] Time axis 
                                              of the time series. Is used only 
                                              if set and if 'action' key value is
                                              set to 'modify'. Default = None.
                                'timestamp'   [Optional. Scalar] Unique 
                                              identifier of the time series. Is 
                                              used only if set and if 'action' 
                                              key value is set to 'modify'.
                                              Default = None.
                                'location'    [Optional. instance of GEOM.Point
                                              class] 
                                              Antenna location in the local ENU 
                                              coordinate system. Used only if 
                                              set and if 'action' key value is 
                                              set to 'modify'. Default = None.
                                'wtsinfo'     [Optional. Dictionary] 
                                              See description in Antenna class 
                                              member function update(). Is used 
                                              only if set and if 'action' key 
                                              value is set to 'modify'.
                                              Default = None.
                                'flags'       [Optional. Dictionary] holds 
                                              boolean flags for each of the 2 
                                              polarizations which are stored 
                                              under keys 'P1' and 'P2'. 
                                              Default=None means no updates for 
                                              flags. If True, that polarization 
                                              will be flagged. If not set 
                                              (=None), the previous or default 
                                              flag status will continue to 
                                              apply. If set to False, the 
                                              antenna status will be updated to 
                                              become unflagged.
                                'gridfunc_freq'
                                              [Optional. String scalar] Read the 
                                              description of inputs to Antenna 
                                              class member function update(). If 
                                              set to None (not provided), this
                                              attribute is determined based on 
                                              the size of wtspos under each 
                                              polarization. It is applicable 
                                              only when 'action' key is set to 
                                              'modify'. Default = None.
                                'delaydict'   [Dictionary] contains information 
                                              on delay compensation to be applied 
                                              to the fourier transformed electric 
                                              fields under each polarization which
                                              are stored under keys 'P1' and 'P2'. 
                                              Default is None (no delay 
                                              compensation to be applied). Refer 
                                              to the docstring of member function
                                              delay_compensation() of class 
                                              PolInfo for more details.
                                'ref_freq'    [Optional. Scalar] Positive value 
                                              (in Hz) of reference frequency 
                                              (used if gridfunc_freq is set to
                                              'scale') at which wtspos in 
                                              wtsinfo are provided. If set to 
                                              None, the reference frequency 
                                              already set in antenna array 
                                              instance remains unchanged. 
                                              Default = None.
                                'pol_type'    [Optional. String scalar] 'Linear' 
                                              or 'Circular'. Used only when 
                                              action key is set to 'modify'. If 
                                              not provided, then the previous
                                              value remains in effect.
                                              Default = None.
                                'norm_wts'    [Optional. Boolean] Default=False. 
                                              If set to True, the gridded weights 
                                              are divided by the sum of weights 
                                              so that the gridded weights add up 
                                              to unity. This is used only when
                                              grid_action keyword is set when
                                              action keyword is set to 'add' or
                                              'modify'
                                'gridmethod'  [Optional. String] Indicates 
                                              gridding method. It accepts the 
                                              following values 'NN' (nearest 
                                              neighbour), 'BL' (Bi-linear
                                              interpolation), and'CS' (Cubic
                                              Spline interpolation). Default='NN'
                                'distNN'      [Optional. Scalar] Indicates the 
                                              upper bound on distance for a 
                                              nearest neighbour search if the 
                                              value of 'gridmethod' is set to
                                              'NN'. The units are of physical
                                              distance, the same as what is 
                                              used for antenna locations.
                                              Default = NP.inf
                                'maxmatch'    [scalar] A positive value 
                                              indicating maximum number of input
                                              locations in the antenna grid to 
                                              be assigned. Default = None. If 
                                              set to None, all the antenna array 
                                              grid elements specified are 
                                              assigned values for each antenna.
                                              For instance, to have only one
                                              antenna array grid element to be
                                              populated per antenna, use
                                              maxmatch=1. 
                                'tol'         [scalar] If set, only lookup data 
                                              with abs(val) > tol will be
                                              considered for nearest neighbour 
                                              lookup. Default = None implies 
                                              all lookup values will be 
                                              considered for nearest neighbour
                                              determination. tol is to be
                                              interpreted as a minimum value
                                              considered as significant in the
                                              lookup table. 

        parallel   [boolean] specifies if parallelization is to be invoked. 
                   False (default) means only serial processing

        nproc      [integer] specifies number of independent processes to spawn.
                   Default = None, means automatically determines the number of 
                   process cores in the system and use one less than that to 
                   avoid locking the system for other processes. Applies only 
                   if input parameter 'parallel' (see above) is set to True. 
                   If nproc is set to a value more than the number of process
                   cores in the system, it will be reset to number of process 
                   cores in the system minus one to avoid locking the system out 
                   for other processes

        verbose     [Boolean] Default = False. If set to True, prints some 
                    diagnotic or progress messages.

        -------------------------------------------------------------------------
        """

        if updates is not None:
            if not isinstance(updates, dict):
                raise TypeError('Input parameter updates must be a dictionary')

            if 'antennas' in updates: # contains updates at level of individual antennas
                if not isinstance(updates['antennas'], list):
                    updates['antennas'] = [updates['antennas']]
                if parallel:
                    list_of_antenna_updates = []
                    list_of_antennas = []
                for dictitem in updates['antennas']:
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
                            if 'Et' not in dictitem: dictitem['Et']=None
                            if 't' not in dictitem: dictitem['t']=None
                            if 'timestamp' not in dictitem: dictitem['timestamp']=None
                            if 'location' not in dictitem: dictitem['location']=None
                            if 'wtsinfo' not in dictitem: dictitem['wtsinfo']=None
                            if 'flags' not in dictitem: dictitem['flags']=None
                            if 'gridfunc_freq' not in dictitem: dictitem['gridfunc_freq']=None
                            if 'ref_freq' not in dictitem: dictitem['ref_freq']=None
                            if 'pol_type' not in dictitem: dictitem['pol_type']=None
                            if 'norm_wts' not in dictitem: dictitem['norm_wts']=False
                            if 'gridmethod' not in dictitem: dictitem['gridmethod']='NN'
                            if 'distNN' not in dictitem: dictitem['distNN']=NP.inf
                            if 'maxmatch' not in dictitem: dictitem['maxmatch']=None
                            if 'tol' not in dictitem: dictitem['tol']=None
                            if 'delaydict' not in dictitem: dictitem['delaydict']=None
                            
                            if not parallel:
                                self.antennas[dictitem['label']].update(dictitem, verbose)
                            else:
                                list_of_antennas += [self.antennas[dictitem['label']]]
                                list_of_antenna_updates += [dictitem]

                            if 'gric_action' in dictitem:
                                self.grid_convolve(pol=dictitem['gridpol'], ants=dictitem['antenna'], unconvolve_existing=True, normalize=dictitem['norm_wts'], method=dictitem['gridmethod'], distNN=dictitem['distNN'], tol=dictitem['tol'], maxmatch=dictitem['maxmatch'])
                    else:
                        raise ValueError('Update action should be set to "add", "remove" or "modify".')

                if parallel:
                    if nproc is None:
                        nproc = max(MP.cpu_count()-1, 1) 
                    else:
                        nproc = min(nproc, max(MP.cpu_count()-1, 1))
                    pool = MP.Pool(processes=nproc)
                    updated_antennas = pool.map(unwrap_antenna_update, IT.izip(list_of_antennas, list_of_antenna_updates))
                    pool.close()
                    pool.join()

                    # Necessary to make the returned and updated antennas current, otherwise they stay unrelated
                    for antenna in updated_antennas: 
                        self.antennas[antenna.label] = antenna
                    del updated_antennas
                    

            if 'antenna_array' in updates: # contains updates at 'antenna array' level
                if not isinstance(updates['antenna_array'], dict):
                    raise TypeError('Input parameter in updates for antenna array must be a dictionary with key "antenna_array"')
                
                if 'timestamp' in updates['antenna_array']:
                    self.timestamp = updates['antenna_array']['timestamp']

                if 'do_grid' in updates['antenna_array']:
                    if isinstance(updates['antenna_array']['do_grid'], boolean):
                        self.grid()
                    else:
                        raise TypeError('Value in key "do_grid" inside key "antenna_array" of input dictionary updates must be boolean.')

        self.t = self.antennas.itervalues().next().t # Update time axis
        self.f = self.antennas.itervalues().next().f # Update frequency axis
        self.update_flags()

    ############################################################################
