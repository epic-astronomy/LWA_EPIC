import numpy as NP
import numpy.ma as MA
import multiprocessing as MP
import itertools as IT
import copy
import scipy.constants as FCNST
import scipy.sparse as SpM
from astropy.io import fits
import matplotlib.pyplot as PLT
import progressbar as PGB
from astroutils import DSP_modules as DSP
from astroutils import geometry as GEOM
from astroutils import gridding_modules as GRD
from astroutils import mathops as OPS
from astroutils import lookup_operations as LKP
import aperture as APR

################### Routines essential for parallel processing ################

def unwrap_antenna_FT(arg, **kwarg):
    return Antenna.FT_pp(*arg, **kwarg)

def unwrap_interferometer_FX(arg, **kwarg):
    return Interferometer.FX_pp(*arg, **kwarg)

def unwrap_interferometer_stack(arg, **kwarg):
    return Interferometer.stack_pp(*arg, **kwarg)

def unwrap_antenna_update(arg, **kwarg):
    return Antenna.update_pp(*arg, **kwarg)

def unwrap_interferometer_update(arg, **kwarg):
    return Interferometer.update_pp(*arg, **kwarg)

def antenna_grid_mapping(gridind_raveled, values, bins=None):
    if bins is None:
        raise ValueError('Input parameter bins must be specified')

    if NP.iscomplexobj(values):
        retval = OPS.binned_statistic(gridind_raveled, values.real, statistic='sum', bins=bins)[0]
        retval = retval.astype(NP.complex64)
        retval += 1j * OPS.binned_statistic(gridind_raveled, values.imag, statistic='sum', bins=bins)[0]
    else:
        retval = OPS.binned_statistic(gridind_raveled, values, statistic='sum', bins=bins)[0]

    # print MP.current_process().name
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
    # print MP.current_process().name
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

    # print MP.current_process().name
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
    # print MP.current_process().name
    outq.put(outdict)

def find_1NN_arg_splitter(args, **kwargs):
    return LKP.find_1NN(*args, **kwargs)

def genMatrixMapper_arg_splitter(args, **kwargs):
    return genMatrixMapper(*args, **kwargs)

def genMatrixMapper(val, ind, shape):
    if not isinstance(val, NP.ndarray):
        raise TypeError('Input parameter val must be a numpy array')
    if not isinstance(ind, (list,tuple)):
        raise TypeError('Input parameter ind must be a list or tuple containing numpy arrays')
    if val.size != ind[0].size:
        raise ValueError('Input parameters val and ind must have the same size')
    if not isinstance(shape, (tuple,list)):
        raise TypeError('Input parameter shape must be a tuple or list')
    if len(ind) != len(shape):
        raise ValueError('Number of index groups in input parameter must match the number of dimensions specified in input parameter shape')
    if len(ind) > 1:
        for i in range(len(ind)-1):
            if ind[i+1].size != ind[i].size:
                raise ValueError('All index groups must have same size')
    return SpM.csr_matrix((val, ind), shape=shape)

################################################################################

def evalCorrWts2PB(wts_grid, ulocs, vlocs, pad=0, skypos=None):

    """
    ------------------------------------------------------------------------
    Evaluate power pattern on sky from weights on the UV-plane
    
    Inputs:

    wts_grid    [numpy array or scipy sparse matrix] Complex weights on 
                the UV-plane and along frequency axis. It can be a numpy 
                array of size nv x nu x nchan or a scipy sparse matrix of
                size (nv x nu) x nchan. 

    ulocs       [numpy array] u-locations on grid. It is of size nu and must
                match the dimension in wts_grid

    vlocs       [numpy array] v-locations on grid. It is of size nv and must
                match the dimension in wts_grid

    pad         [integer] indicates the amount of padding before estimating
                power pattern. Applicable only when skypos is set to None. 
                The output power pattern will be of size 2**pad-1 times the 
                size of the UV-grid along l- and m-axes. Value must 
                not be negative. Default=0 (implies no padding). pad=1 
                implies padding by factor 2 along u- and v-axes

    skypos      [numpy array] Positions on sky at which power pattern is 
                to be esimated. It is a 2- or 3-column numpy array in 
                direction cosine coordinates. It must be of size nsrc x 2 
                or nsrc x 3. If set to None (default), the power pattern is 
                estimated over a grid on the sky. If a numpy array is
                specified, then power pattern at the given locations is 
                estimated.

    Outputs:
    pbinfo is a dictionary with the following keys and values:
    'pb'    [numpy array] If skypos was set to None, the numpy array is 
            3D masked array of size nm x nl x nchan. The mask is based on 
            which parts of the grid are valid direction cosine coordinates 
            on the sky. If skypos was a numpy array denoting specific sky 
            locations, the value in this key is a 2D numpy array of size 
            nsrc x nchan
    'llocs' [None or numpy array] If the power pattern estimated is a grid
            (if input skypos was set to None), it contains the l-locations
            of the grid on the sky. If input skypos was not set to None, 
            the value under this key is set to None
    'mlocs' [None or numpy array] If the power pattern estimated is a grid
            (if input skypos was set to None), it contains the m-locations
            of the grid on the sky. If input skypos was not set to None, 
            the value under this key is set to None
    ------------------------------------------------------------------------
    """

    try:
        wts_grid, ulocs, vlocs
    except NameError:
        raise NameError('Inputs wts_grid, ulocs and vlocs must be specified')

    if skypos is not None:
        if not isinstance(skypos, NP.ndarray):
            raise TypeError('Input skypos must be a numpy array')
        if skypos.ndim != 2:
            raise ValueError('Input skypos must be a 2D numpy array')

        if (skypos.shape[1] < 2) or (skypos.shape[1] > 3):
            raise ValueError('Input skypos must be a 2- or 3-column array')

        skypos = skypos[:,:2]
        if NP.any(NP.sum(skypos**2, axis=1) > 1.0):
            raise ValueError('Magnitude of skypos direction cosine must not exceed unity')

    if not isinstance(ulocs, NP.ndarray):
        raise TypeError('Input ulocs must be a numpy array')

    if not isinstance(vlocs, NP.ndarray):
        raise TypeError('Input vlocs must be a numpy array')

    if not isinstance(pad, int):
        raise TypeError('Input must be an integer')
    if pad < 0:
        raise ValueError('Input pad must be non-negative')

    ulocs = ulocs.ravel()
    vlocs = vlocs.ravel()
    
    wts_shape = wts_grid.shape
    if wts_shape[0] != ulocs.size * vlocs.size:
        raise ValueError('Shape of input wts_grid incompatible with that of ulocs and vlocs')
    
    if SpM.issparse(wts_grid):
        sum_wts = wts_grid.sum(axis=0).A # 1 x nchan
        sum_wts = sum_wts[NP.newaxis,:,:] # 1 x 1 x nchan
    else:
        sum_wts = NP.sum(wts_grid, axis=(0,1), keepdims=True) # 1 x 1 x nchan
        
    llocs = None
    mlocs = None
    if skypos is None:
        if SpM.issparse(wts_grid):
            shape_tuple = (vlocs.size, ulocs.size) + (wts_grid.shape[1],)
            wts_grid = wts_grid.toarray().reshape(shape_tuple)
        padded_wts_grid = NP.pad(wts_grid, (((2**pad-1)*vlocs.size/2,(2**pad-1)*vlocs.size/2),((2**pad-1)*ulocs.size/2,(2**pad-1)*ulocs.size/2),(0,0)), mode='constant', constant_values=0)
        padded_wts_grid = NP.fft.ifftshift(padded_wts_grid, axes=(0,1))
        wts_lmf = NP.fft.fft2(padded_wts_grid, axes=(0,1)) / sum_wts
        pb = NP.fft.fftshift(wts_lmf, axes=(0,1))
        llocs = NP.fft.fftshift(NP.fft.fftfreq(2**pad * ulocs.size, ulocs[1]-ulocs[0]))
        mlocs = NP.fft.fftshift(NP.fft.fftfreq(2**pad * vlocs.size, vlocs[1]-vlocs[0]))
        lmgrid_invalid = llocs.reshape(1,-1)**2 + mlocs.reshape(-1,1)**2 > 1.0
        lmgrid_invalid = lmgrid_invalid[:,:,NP.newaxis] * NP.ones(pb.shape[2], dtype=NP.bool).reshape(1,1,-1)
        pb = MA.array(pb, mask=lmgrid_invalid)
    else:
        gridu, gridv = NP.meshgrid(ulocs, vlocs)
        griduv = NP.hstack((gridu.reshape(-1,1),gridv.reshape(-1,1)))
        if SpM.issparse(wts_grid):
            uvind = SpM.find(wts_grid)[0]
        else:
            eps = 1e-10
            wts_grid = wts_grid.reshape(griduv.shape[0],-1)
            uvind, freqind = NP.where(NP.abs(wts_grid) > eps)
            wts_grid = SpM.csr_matrix((wts_grid[(uvind, freqind)], (uvind, freqind)), shape=(gridu.size,wts_grid.shape[1]), dtype=NP.complex64)
            
        uniq_uvind = NP.unique(uvind)
        matFT = NP.exp(-1j*2*NP.pi*NP.dot(skypos, griduv[uniq_uvind,:].T))
        uvmeshind, srcmeshind = NP.meshgrid(uniq_uvind, NP.arange(skypos.shape[0]))
        uvmeshind = uvmeshind.ravel()
        srcmeshind = srcmeshind.ravel()
        spFTmat = SpM.csr_matrix((matFT.ravel(), (srcmeshind, uvmeshind)), shape=(skypos.shape[0],griduv.shape[0]), dtype=NP.complex64)
        sum_wts = wts_grid.sum(axis=0).A
        pb = spFTmat.dot(wts_grid) / sum_wts
        pb = pb.A
    pb = pb.real
    pbinfo = {'pb': pb, 'llocs': llocs, 'mlocs': mlocs}
    return pbinfo

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

    update_flags() Updates the flags based on current inputs and verifies and 
                   updates flags based on current values of the electric field.

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
        Vt, Vf, flags, internal attributes _init_flags_on and _init_data_on
     
        Read docstring of class PolInfo for details on these attributes.
        ------------------------------------------------------------------------
        """

        self.Vt = {}
        self.Vf = {}
        self.flag = {}
        self._init_flags_on = True
        self._init_data_on = True

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

    def update_flags(self, flags=None, verify=True):

        """
        ------------------------------------------------------------------------
        Updates the flags based on current inputs and verifies and updates flags
        based on current values of the visibilities.
    
        Inputs:
    
        flags    [dictionary] holds boolean flags for each of the 4 cross-
                 polarizations which are stored under keys 'P11', 'P12', 'P21', 
                 and 'P22'. Default=None means no new flagging to be applied. If 
                 the value under the cross-polarization key is True, it is to be 
                 flagged and if False, it is to be unflagged.

        verify   [boolean] If True, verify and update the flags, if necessary.
                 Visibilities are checked for NaN values and if found, the
                 flag in the corresponding polarization is set to True. 
                 Default=True. 

        Flag verification and re-updating happens if flags is set to None or if
        verify is set to True.
        ------------------------------------------------------------------------
        """

        if not isinstance(verify, bool):
            raise TypeError('Input keyword verify must be of boolean type')

        if flags is not None:
            if not isinstance(flags, dict):
                raise TypeError('Input parameter flags must be a dictionary')
            for pol in ['P11', 'P12', 'P21', 'P22']:
                if pol in flags:
                    if isinstance(flags[pol], bool):
                        self.flag[pol] = flags[pol]
                    else:
                        raise TypeError('flag values must be boolean')
                    self._init_flags_on = False

            # self.flags = {pol: flags[pol] for pol in ['P11', 'P12', 'P21', 'P22'] if pol in flags}
            # self._init_flags_on = False
            
        # Perform flag verification and re-update current flags
        if verify or (flags is None):
            if not self._init_data_on:
                for pol in ['P11', 'P12', 'P21', 'P22']:
                    if NP.any(NP.isnan(self.Vt[pol])):
                        self.flag[pol] = True
                self._init_flags_on = False
                    
    ############################################################################

    def update(self, Vt=None, Vf=None, flags=None, verify=False):
        
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

        verify [boolean] If True, verify and update the flags, if necessary.
               Visibilities are checked for NaN values and if found, the
               flag in the corresponding polarization is set to True. 
               Default=False. 
        ------------------------------------------------------------------------
        """

        current_flags = copy.deepcopy(self.flag)
        if flags is None:
            flags = copy.deepcopy(current_flags)
        # if flags is not None:
        #     self.update_flags(flags)

        if Vt is not None:
            if isinstance(Vt, dict):
                for pol in ['P11', 'P12', 'P21', 'P22']:
                    if pol in Vt:
                        self.Vt[pol] = Vt[pol]
                        if NP.any(NP.isnan(Vt[pol])):
                            # self.Vt[pol] = NP.nan
                            flags[pol] = True
                            # self.flag[pol] = True
                        self._init_data_on = False
            else:
                raise TypeError('Input parameter Vt must be a dictionary')

        if Vf is not None:
            if isinstance(Vf, dict):
                for pol in ['P11', 'P12', 'P21', 'P22']:
                    if pol in Vf:
                        self.Vf[pol] = Vf[pol]
                        if NP.any(NP.isnan(Vf[pol])):
                            # self.Vf[pol] = NP.nan
                            flags[pol] = True
                            # self.flag[pol] = True
                        self._init_data_on = False
            else:
                raise TypeError('Input parameter Vf must be a dictionary')

        # Update flags
        self.update_flags(flags=flags, verify=verify)
        
################################################################################

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

    timestamps  [list] consists of a list of timestamps common to both of the
                individual antennas in the antenna pair

    t:          [vector] The time axis for the time series of electric fields

    f:          [vector] Frequency axis obtained by a Fourier Transform of
                the electric field time series. Same length as attribute t 

    f0:         [Scalar] Center frequency in Hz.

    crosspol:   [Instance of class CrossPolInfo] polarization information for 
                the interferometer. Read docstring of class CrossPolInfo for 
                details

    aperture    [Instance of class APR.Aperture] aperture information
                for the interferometer. Read docstring of class Aperture for
                details

    Vt_stack    [dictionary] holds a stack of complex visibility time series 
                measured at various time stamps under 4 polarizations which are 
                stored under keys 'P11', 'P12', 'P21', and 'P22'. Each value 
                under the polarization key is stored as numpy array with rows 
                equal to the number of timestamps and columns equal to the 
                number of samples in a timeseries
                
    Vf_stack    [dictionary] holds a stack of complex visibility spectra 
                measured at various time stamps under 4 polarizations which are 
                stored under keys 'P11', 'P12', 'P21' and 'P22'. Each value 
                under the polarization key is stored as numpy array with rows 
                equal to the number of timestamps and columns equal to the 
                number of spectral channels

    flag_stack  [dictionary] holds a stack of flags appropriate for different 
                time stamps as a numpy array under 4 polarizations which are 
                stored under keys 'P11', 'P12', 'P21' and 'P22'. Each value 
                under the polarization key is stored as numpy array with 
                number of elements equal to the number of timestamps

    Vf_avg      [dictionary] holds in keys 'P11', 'P12', 'P21', 'P22' for each
                polarization the stacked and averaged complex visibility spectra
                as a numpy array where the number of rows is the number of time
                bins after averaging visibilities in those time bins and the 
                number of columns is equal to the number of spectral channels 
                (same as in Vf_stack)

    twts        [dictionary] holds in keys 'P11', 'P12', 'P21', 'P22' for each
                polarization the number of unflagged timestamps in each time 
                bin that contributed to the averaging of visibilities stored in
                Vf_avg. Each array size equal to the number of rows in Vf_avg
                under the corresponding polarization.

    tbinsize    [scalar or dictionary] Contains bin size of timestamps while
                stacking. Default = None means all visibility spectra over all
                timestamps are averaged. If scalar, the same (positive) value 
                applies to all polarizations. If dictionary, timestamp bin size
                (positive) is provided under each key 'P11', 'P12', 'P21', 
                'P22'. If any of the keys is missing the visibilities for that 
                polarization are averaged over all timestamps.

    wts:        [dictionary] The gridding weights for interferometer. Different 
                cross-polarizations 'P11', 'P12', 'P21' and 'P22' form the keys 
                of this dictionary. These values are in general complex. Under 
                each key, the values are maintained as a list of numpy vectors, 
                where each vector corresponds to a frequency channel. See 
                wtspos_scale for more requirements.

    wtspos      [dictionary] two-dimensional locations of the gridding weights 
                in wts for each cross-polarization under keys 'P11', 'P12', 
                'P21', and 'P22'. The locations are in ENU coordinate system 
                as a list of 2-column numpy arrays. Each 2-column array in the 
                list is the position of the gridding weights for a corresponding
                frequency channel. The size of the list must be the same as wts
                and the number of channels. Units are in number of wavelengths.
                See wtspos_scale for more requirements.

    wtspos_scale [dictionary] The scaling of weights is specified for each 
                 cross-polarization under one of the keys 'P11', 'P12', 'P21' 
                 or 'P22'. The values under these keys can be either None 
                 (default) or 'scale'. If None, numpy vectors in wts and 
                 wtspos under corresponding keys are provided for each 
                 frequency channel. If set to 'scale' wts and wtspos contain a 
                 list of only one numpy array corresponding to a reference 
                 frequency. This is scaled internally to correspond to the 
                 first channel. The gridding positions are correspondingly 
                 scaled to all the frequency channels.

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

    FX_pp()      Computes the visibility spectrum using an FX operation, 
                 i.e., Fourier transform (F) followed by multiplication (X). 
                 All four cross polarizations are computed. To be used 
                 internally for parallel processing and not by the user directly

    XF()         Computes the visibility spectrum using an XF operation, 
                 i.e., corss-correlation (X) followed by Fourier transform 
                 (F) using antenna information in attributes A1 and A2. All 
                 four cross polarizations are computed.

    f2t()        Computes the visibility time-series from the spectra for each 
                 cross-polarization
    
    t2f()        Computes the visibility spectra from the time-series for each 
                 cross-polarization

    FX_on_stack()
                 Computes the visibility spectrum using an FX operation on the 
                 time-stacked electric fields in the individual antennas in the 
                 pair, i.e., Fourier transform (F) followed by multiplication 
                 (X). All four cross-polarizations are computed.

    flags_on_stack()
                 Computes the visibility flags from the time-stacked electric 
                 fields for the common timestamps between the pair of antennas. 
                 All four cross-polarizations are computed.

    XF_on_stack()
                 Computes the visibility lags using an XF operation on the 
                 time-stacked electric fields time-series in the individual 
                 antennas in the pair, i.e., Cross-correlation (X) followed by 
                 Fourier transform (F). All four cross-polarizations are 
                 computed.

    f2t_on_stack()
                 Computes the visibility lags from the spectra for each 
                 cross-polarization from time-stacked visibilities

    t2f_on_stack() 
                 Computes the visibility spectra from the time-series for each 
                 cross-polarization from time-stacked visibility lags

    flip_antenna_pair()
                 Flip the antenna pair in the interferometer. This inverts the
                 baseline vector and conjugates the visibility spectra

    refresh_antenna_pairs()
                 Update the individual antenna instances of the antenna pair 
                 forming the interferometer with provided values

    get_visibilities()
                 Returns the visibilities based on selection criteria on 
                 timestamp flags, timestamps and frequency channel indices 
                 and the type of data (most recent, stack or averaged 
                 visibilities)

    update_flags()
                 Updates flags for cross-polarizations from component antenna
                 polarization flags and also overrides with flags if provided 
                 as input parameters

    update():    Updates the interferometer instance with newer attribute values
                 Updates the visibility spectrum and timeseries and applies FX
                 or XF operation.

    update_pp()  Updates the interferometer instance with newer attribute 
                 values. Updates the visibility spectrum and timeseries and 
                 applies FX or XF operation. Used internally when parallel 
                 processing is used. Not to be used by the user directly.

    stack()      Stacks and computes visibilities and flags from the individual 
                 antennas in the pair.

    accumulate() Accumulate and average visibility spectra across timestamps 
                 under different polarizations depending on the time bin size 
                 for the corresponding polarization.

    save():      Saves the interferometer information to disk. Needs serious 
                 development. 

    Read the member function docstrings for details.
    ----------------------------------------------------------------------------
    """

    def __init__(self, antenna1, antenna2, corr_type=None, aperture=None):

        """
        ------------------------------------------------------------------------
        Initialize the Interferometer Class which manages an interferometer's
        information 

        Class attributes initialized are:
        label, latitude, location, pol, t, timestamp, f0, f, wts, wtspos, 
        wtspos_scale, gridinfo, blc, trc, timestamps, Vt_stack, Vf_stack, 
        flag_stack, Vf_avg, twts, tbinsize, aperture
     
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
        self.timestamps = []
        
        if aperture is not None:
            if isinstance(aperture, APR.Aperture):
                if len(aperture.pol) != 4:
                    raise ValueError('Interferometer aperture must contain four cross-polarization types')
                self.aperture = aperture
            else:
                raise TypeError('aperture must be an instance of class Aperture found in module {0}'.format(APR.__name__))
        else:
            self.aperture = APR.Aperture(pol_type='cross')

        self.crosspol = CrossPolInfo(self.f.size)

        self.Vt_stack = {}
        self.Vf_stack = {}
        self.flag_stack = {}

        self.Vf_avg = {}
        self.twts = {}
        self.tbinsize = None

        self.wtspos = {}
        self.wts = {}
        self.wtspos_scale = {}
        self._gridinfo = {}

        for pol in ['P11', 'P12', 'P21', 'P22']:
            self.Vt_stack[pol] = None
            self.Vf_stack[pol] = None
            self.flag_stack[pol] = NP.asarray([])

            self.Vf_avg[pol] = None
            self.twts[pol] = None

            self.wtspos[pol] = []
            self.wts[pol] = []
            self.wtspos_scale[pol] = None
            self._gridinfo[pol] = {}

        self.blc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1)
        self.trc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1)

    ############################################################################

    def __str__(self):
        return ' Instance of class "{0}" in module "{1}" \n label: ({2[0]}, {2[1]}) \n location: {3}'.format(self.__class__.__name__, self.__module__, self.label, self.location.__str__())

    ############################################################################

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

    ############################################################################

    def FX(self):

        """
        ------------------------------------------------------------------------
        Computes the visibility spectrum using an FX operation, i.e., Fourier 
        transform (F) followed by multiplication (X). All four cross
        polarizations are computed.
        ------------------------------------------------------------------------
        """

        self.t = NP.hstack((self.A1.t.ravel(), self.A1.t.max()+self.A2.t.ravel()))
        self.f = self.f0 + self.channels()

        self.crosspol.Vf['P11'] = self.A1.antpol.Ef['P1'] * self.A2.antpol.Ef['P1'].conjugate()
        self.crosspol.Vf['P12'] = self.A1.antpol.Ef['P1'] * self.A2.antpol.Ef['P2'].conjugate()
        self.crosspol.Vf['P21'] = self.A1.antpol.Ef['P2'] * self.A2.antpol.Ef['P1'].conjugate()
        self.crosspol.Vf['P22'] = self.A1.antpol.Ef['P2'] * self.A2.antpol.Ef['P2'].conjugate()

        self.f2t()
        self.crosspol._init_data_on = False
        self.update_flags(flags=None, stack=False, verify=True)

    ############################################################################

    def FX_pp(self):

        """
        ------------------------------------------------------------------------
        Computes the visibility spectrum using an FX operation, i.e., Fourier 
        transform (F) followed by multiplication (X). All four cross
        polarizations are computed. To be used internally for parallel 
        processing and not by the user directly
        ------------------------------------------------------------------------
        """

        self.t = NP.hstack((self.A1.t.ravel(), self.A1.t.max()+self.A2.t.ravel()))
        self.f = self.f0 + self.channels()

        self.crosspol.Vf['P11'] = self.A1.antpol.Ef['P1'] * self.A2.antpol.Ef['P1'].conjugate()
        self.crosspol.Vf['P12'] = self.A1.antpol.Ef['P1'] * self.A2.antpol.Ef['P2'].conjugate()
        self.crosspol.Vf['P21'] = self.A1.antpol.Ef['P2'] * self.A2.antpol.Ef['P1'].conjugate()
        self.crosspol.Vf['P22'] = self.A1.antpol.Ef['P2'] * self.A2.antpol.Ef['P2'].conjugate()

        self.f2t()
        self.crosspol._init_data_on = False
        self.update_flags(flags=None, stack=False, verify=True)
        
        return self

    ############################################################################

    def XF(self):

        """
        ------------------------------------------------------------------------
        Computes the visibility spectrum using an XF operation, i.e., 
        Correlation (X) followed by Fourier transform (X). All four cross 
        polarizations are computed.
        ------------------------------------------------------------------------
        """

        self.t = NP.hstack((self.A1.t.ravel(), self.A1.t.max()+self.A2.t.ravel()))
        self.f = self.f0 + self.channels()

        self.crosspol.Vt['P11'] = DSP.XC(self.A1.antpol.Et['P1'], self.A2.antpol.Et['P1'], shift=False)
        self.crosspol.Vt['P12'] = DSP.XC(self.A1.antpol.Et['P1'], self.A2.antpol.Et['P2'], shift=False)
        self.crosspol.Vt['P21'] = DSP.XC(self.A1.antpol.Et['P2'], self.A2.antpol.Et['P1'], shift=False)
        self.crosspol.Vt['P22'] = DSP.XC(self.A1.antpol.Et['P2'], self.A2.antpol.Et['P2'], shift=False)

        self.t2f()
        self.crosspol._init_data_on = False
        self.update_flags(flags=None, stack=False, verify=True)

    ############################################################################

    def f2t(self):

        """
        ------------------------------------------------------------------------
        Computes the visibility time-series from the spectra for each cross-
        polarization
        ------------------------------------------------------------------------
        """
        
        for pol in ['P11', 'P12', 'P21', 'P22']:

            self.crosspol.Vt[pol] = DSP.FT1D(NP.fft.fftshift(self.crosspol.Vf[pol]), inverse=True, shift=True, verbose=False)

    ############################################################################

    def t2f(self):
        
        """
        ------------------------------------------------------------------------
        Computes the visibility spectra from the time-series for each cross-
        polarization
        ------------------------------------------------------------------------
        """

        for pol in ['P11', 'P12', 'P21', 'P22']:

            self.crosspol.Vf[pol] = DSP.FT1D(NP.fft.ifftshift(self.crosspol.Vt[pol]), shift=True, verbose=False)

    ############################################################################

    def FX_on_stack(self):

        """
        ------------------------------------------------------------------------
        Computes the visibility spectrum using an FX operation on the 
        time-stacked electric fields in the individual antennas in the pair, 
        i.e., Fourier transform (F) followed by multiplication (X). All four 
        cross-polarizations are computed.
        ------------------------------------------------------------------------
        """

        self.t = NP.hstack((self.A1.t.ravel(), self.A1.t.max()+self.A2.t.ravel()))
        self.f = self.f0 + self.channels()

        ts1 = NP.asarray(self.A1.timestamps)
        ts2 = NP.asarray(self.A2.timestamps)
        common_ts = NP.intersect1d(ts1, ts2, assume_unique=True)
        ind1 = NP.in1d(ts1, common_ts, assume_unique=True)
        ind2 = NP.in1d(ts2, common_ts, assume_unique=True)

        self.Vf_stack['P11'] = self.A1.Ef_stack['P1'][ind1,:] * self.A2.Ef_stack['P1'][ind2,:].conjugate()
        self.Vf_stack['P12'] = self.A1.Ef_stack['P1'][ind1,:] * self.A2.Ef_stack['P2'][ind2,:].conjugate()
        self.Vf_stack['P21'] = self.A1.Ef_stack['P2'][ind1,:] * self.A2.Ef_stack['P1'][ind2,:].conjugate()
        self.Vf_stack['P22'] = self.A1.Ef_stack['P2'][ind1,:] * self.A2.Ef_stack['P2'][ind2,:].conjugate()

        self.f2t_on_stack()

    ############################################################################

    def flags_on_stack(self):
        
        """
        ------------------------------------------------------------------------
        Computes the visibility flags from the time-stacked electric fields for 
        the common timestamps between the pair of antennas. All four 
        cross-polarizations are computed.
        ------------------------------------------------------------------------
        """

        ts1 = NP.asarray(self.A1.timestamps)
        ts2 = NP.asarray(self.A2.timestamps)
        common_ts = NP.intersect1d(ts1, ts2, assume_unique=True)
        ind1 = NP.in1d(ts1, common_ts, assume_unique=True)
        ind2 = NP.in1d(ts2, common_ts, assume_unique=True)

        self.flag_stack['P11'] = NP.logical_or(self.A1.flag_stack['P1'][ind1],
                                               self.A2.flag_stack['P1'][ind2])
        self.flag_stack['P12'] = NP.logical_or(self.A1.flag_stack['P1'][ind1],
                                               self.A2.flag_stack['P2'][ind2])
        self.flag_stack['P21'] = NP.logical_or(self.A1.flag_stack['P2'][ind1],
                                               self.A2.flag_stack['P1'][ind2])
        self.flag_stack['P22'] = NP.logical_or(self.A1.flag_stack['P2'][ind1],
                                               self.A2.flag_stack['P2'][ind2])

    ############################################################################
    
    def XF_on_stack(self):

        """
        ------------------------------------------------------------------------
        Computes the visibility lags using an XF operation on the time-stacked 
        electric fields time-series in the individual antennas in the pair, 
        i.e., Cross-correlation (X) followed by Fourier transform (F). All four 
        cross-polarizations are computed. 

        THIS WILL NOT WORK IN ITS CURRENT FORM BECAUSE THE ENGINE OF THIS IS 
        THE CORRELATE FUNCTION OF NUMPY WRAPPED INSIDE XC() IN MY_DSP_MODULE 
        AND CURRENTLY IT CAN HANDLE ONLY 1D ARRAYS. NEEDS SERIOUS DEVELOPMENT!
        ------------------------------------------------------------------------
        """

        self.t = NP.hstack((self.A1.t.ravel(), self.A1.t.max()+self.A2.t.ravel()))
        self.f = self.f0 + self.channels()

        ts1 = NP.asarray(self.A1.timestamps)
        ts2 = NP.asarray(self.A2.timestamps)
        common_ts = NP.intersect1d(ts1, ts2, assume_unique=True)
        ind1 = NP.in1d(ts1, common_ts, assume_unique=True)
        ind2 = NP.in1d(ts2, common_ts, assume_unique=True)

        self.Vt_stack['P11'] = DSP.XC(self.A1.Et_stack['P1'], self.A2.Et_stack['P1'], shift=False)
        self.Vt_stack['P12'] = DSP.XC(self.A1.Et_stack['P1'], self.A2.Et_stack['P2'], shift=False)
        self.Vt_stack['P21'] = DSP.XC(self.A1.Et_stack['P2'], self.A2.Et_stack['P1'], shift=False)
        self.Vt_stack['P22'] = DSP.XC(self.A1.Et_stack['P2'], self.A2.Et_stack['P2'], shift=False)

        self.t2f_on_stack()

    ############################################################################

    def f2t_on_stack(self):

        """
        ------------------------------------------------------------------------
        Computes the visibility lags from the spectra for each cross-
        polarization from time-stacked visibilities
        ------------------------------------------------------------------------
        """
        for pol in ['P11', 'P12', 'P21', 'P22']:
            self.Vt_stack[pol] = DSP.FT1D(NP.fft.fftshift(self.Vf_stack[pol]),
                                          ax=1, inverse=True, shift=True,
                                          verbose=False)

    ############################################################################

    def t2f_on_stack(self):
        
        """
        ------------------------------------------------------------------------
        Computes the visibility spectra from the time-series for each cross-
        polarization from time-stacked visibility lags
        ------------------------------------------------------------------------
        """

        for pol in ['P11', 'P12', 'P21', 'P22']:
            self.Vf_stack[pol] = DSP.FT1D(NP.fft.ifftshift(self.Vt_stack[pol]),
                                          ax=1, shift=True, verbose=False)

    ############################################################################

    def flip_antenna_pair(self):
        
        """
        ------------------------------------------------------------------------
        Flip the antenna pair in the interferometer. This inverts the baseline
        vector and conjugates the visibility spectra
        ------------------------------------------------------------------------
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

    def refresh_antenna_pairs(self, A1=None, A2=None):

        """
        ------------------------------------------------------------------------
        Update the individual antenna instances of the antenna pair forming
        the interferometer with provided values

        Inputs:

        A1   [instance of class Antenna] first antenna instance in the 
             antenna pair corresponding to attribute A1. Default=None (no 
             update for attribute A1)

        A2   [instance of class Antenna] first antenna instance in the 
             antenna pair corresponding to attribute A2. Default=None (no 
             update for attribute A2)
        ------------------------------------------------------------------------
        """

        if isinstance(A1, Antenna):
            self.A1 = A1
        else:
            raise TypeError('Input A1 must be an instance of class Antenna')

        if isinstance(A2, Antenna):
            self.A2 = A2
        else:
            raise TypeError('Input A2 must be an instance of class Antenna')

    ############################################################################

    def get_visibilities(self, pol, flag=None, tselect=None, fselect=None,
                         datapool=None):

        """
        ------------------------------------------------------------------------
        Returns the visibilities based on selection criteria on timestamp 
        flags, timestamps and frequency channel indices and the type of data
        (most recent, stack or averaged visibilities)

        Inputs:

        pol      [string] select baselines of this polarization that are either 
                 flagged or unflagged as specified by input parameter flag. 
                 Allowed values are 'P11', 'P12', 'P21', and 'P22'. Only one of
                 these values must be specified.

        flag     [boolean] If False, return visibilities of unflagged 
                 timestamps, otherwise return flagged ones. Default=None means 
                 all visibilities independent of flagging are returned. This 
                 flagging refers to that along the timestamp axis under each
                 polarization
 
        tselect  [scalar, list, numpy array] timestamp index for visibilities
                 selection. For most recent visibility, it must be set to -1.
                 For all other selections, indices in tselect must be in the 
                 valid range of indices along time axis for stacked and 
                 averaged visibilities. Default=None means most recent data is
                 selected. 

        fselect  [scalar, list, numpy array] frequency channel index for 
                 visibilities selection. Indices must be in the valid range of
                 indices along the frequency axis for visibilities. 
                 Default=None selects all frequency channels

        datapool [string] denotes the data pool from which visibilities are to
                 be selected. Accepted values are 'current', 'stack', 'avg' and
                 None (default, same as 'current'). If set to None or 
                 'current', the value in tselect is ignored and only 
                 visibilities of the most recent timestamp are selected. If set
                 to None or 'current' the attribute Vf_stack is checked first 
                 and if unavailable, attribute crosspol.Vf is used. For 'stack'
                 and 'avg', attributes Vf_stack and Vf_avg are used 
                 respectively

        Output:

        outdict  [dictionary] consists of visibilities information under the 
                 following keys:

                 'label'        [tuple] interferometer label as a tuple of 
                                individual antenna labels
                 'pol'          [string] polarization string, one of 'P11', 
                                'P12', 'P21', or 'P22'
                 'visibilities' [numpy array] selected visibilities spectra
                                with dimensions n_ts x nchan which
                                are in time-frequency order. If no
                                visibilities are found satisfying the selection 
                                criteria, the value under this key is set to 
                                None.
                 'twts'         [numpy array] weights corresponding to the time
                                axis in the selected visibilities. These 
                                weights are determined by flagging of 
                                timestamps. A zero weight indicates unflagged 
                                visibilities were not found for that timestamp. 
                                A non-zero weight indicates how many unflagged
                                visibilities were found for that time bin (in
                                case of averaged visibilities) or timestamp. 
                                If no visibilities are found satisfying the 
                                selection criteria, the value under this key 
                                is set to None.
        ------------------------------------------------------------------------
        """

        try: 
            pol 
        except NameError:
            raise NameError('Input parameter pol must be specified.')

        if not isinstance(pol, str):
            raise TypeError('Input parameter must be a string')
        
        if not pol in ['P11', 'P12', 'P21', 'P22']:
            raise ValueError('Invalid specification for input parameter pol')

        if datapool is None:
            n_timestamps = 1
            datapool = 'current'
        elif datapool == 'stack':
            n_timestamps = len(self.timestamps)
        elif datapool == 'avg':
            n_timestamps = self.Vf_avg[pol].shape[0]
        elif datapool == 'current':
            n_timestamps = 1
        else:
            raise ValueError('Invalid datapool specified')

        if tselect is None:
            tsind = NP.asarray(-1).reshape(-1)  # Selects most recent data
        elif isinstance(tselect, (int, float, list, NP.ndarray)):
            tsind = NP.asarray(tselect).ravel()
            tsind = tsind.astype(NP.int)
            if tsind.size == 1:
                if (tsind < -1) or (tsind >= n_timestamps):
                    tsind = NP.asarray(-1).reshape(-1)
            else:
                if NP.any(tsind < 0) or NP.any(tsind >= n_timestamps):
                    raise IndexError('Timestamp indices outside available range for the specified datapool')
        else:
            raise TypeError('tselect must be None, integer, float, list or numpy array for visibilities selection')

        if fselect is None:
            chans = NP.arange(self.f.size)  # Selects all channels
        elif isinstance(fselect, (int, float, list, NP.ndarray)):
            chans = NP.asarray(fselect).ravel()
            chans = chans.astype(NP.int)
            if NP.any(chans < 0) or NP.any(chans >= self.f.size):
                raise IndexError('Channel indices outside available range')
        else:
            raise TypeError('fselect must be None, integer, float, list or numpy array for visibilities selection')

        select_ind = NP.ix_(tsind, chans)

        outdict = {}
        outdict['pol'] = pol
        outdict['twts'] = None
        outdict['label'] = self.label
        outdict['visibilities'] = None
        
        if datapool == 'current':
            if self.Vf_stack[pol] is not None:
                outdict['visibilities'] = self.Vf_stack[pol][-1,chans].reshape(1,chans.size)
                outdict['twts'] = NP.logical_not(NP.asarray(self.flag_stack[pol][-1]).astype(NP.bool).reshape(-1)).astype(NP.float)
            else:
                outdict['visibilities'] = self.crosspol.Vf[pol][chans].reshape(1,chans.size)
                outdict['twts'] = NP.logical_not(NP.asarray(self.crosspol.flag[pol]).astype(NP.bool).reshape(-1)).astype(NP.float)
        elif datapool == 'stack':
            if self.Vf_stack[pol] is not None:
                outdict['visibilities'] = self.Vf_stack[pol][select_ind].reshape(tsind.size,chans.size)
                outdict['twts'] = NP.logical_not(NP.asarray(self.flag_stack[pol][tsind]).astype(NP.bool).reshape(-1)).astype(NP.float)
            else:
                raise ValueError('Attribute Vf_stack has not been initialized to obtain visibilities from. Consider running method stack()')
        else:
            if self.Vf_avg[pol] is not None:
                outdict['visibilities'] = self.Vf_avg[pol][select_ind].reshape(tsind.size,chans.size)
                outdict['twts'] = NP.asarray(self.twts[pol][tsind]).reshape(-1)
            else:
                raise ValueError('Attribute Vf_avg has not been initialized to obtain visibilities from. Consider running methods stack() and accumulate()')

        if flag is not None:
            if not isinstance(flag, bool):
                raise TypeError('flag keyword has to be a Boolean value.')

            if flag:
                if NP.sum(outdict['twts'] == 0) == 0:
                    outdict['twts'] = None
                    outdict['visibilities'] = None
                else:
                    outdict['visibilities'] = outdict['visibilities'][outdict['twts']==0,:].reshape(-1,chans.size)
                    outdict['twts'] = outdict['twts'][outdict['twts']==0].reshape(-1,1)
            else:
                if NP.sum(outdict['twts'] > 0) == 0:
                    outdict['twts'] = None
                    outdict['visibilities'] = None
                else:
                    outdict['visibilities'] = outdict['visibilities'][outdict['twts']>0,:].reshape(-1,chans.size)
                    outdict['twts'] = outdict['twts'][outdict['twts']>0].reshape(-1,1)

        return outdict
                
    ############################################################################

    def update_flags(self, flags=None, stack=False, verify=True):

        """
        ------------------------------------------------------------------------
        Updates flags for cross-polarizations from component antenna
        polarization flags and also overrides with flags if provided as input
        parameters

        Inputs:

        flags  [dictionary] boolean flags for each of the 4 cross-polarizations 
               of the interferometer which are stored under keys 'P11', 'P12',
               'P21', and 'P22'. Default=None means no updates for flags.

        stack  [boolean] If True, appends the updated flag to the
               end of the stack of flags as a function of timestamp. If False,
               updates the last flag in the stack with the updated flag and 
               does not append. Default=False

        verify [boolean] If True, verify and update the flags, if necessary.
               Visibilities are checked for NaN values and if found, the
               flag in the corresponding polarization is set to True. Flags of
               individual antennas forming a pair are checked and transferred
               to the visibility flags. Default=True 
        ------------------------------------------------------------------------
        """

        # By default carry over the flags from previous timestamp
        # unless updated in this timestamp as below
        # Flags determined from interferometer level

        if flags is None:
            if self.crosspol._init_flags_on:  # begin with all flags set to False for first time update of flags
                flags = {pol: False for pol in ['P11', 'P12', 'P21', 'P22']}
            else:  # for non-first time updates carry over flags from last timestamp and process
                flags = copy.deepcopy(self.crosspol.flag)

            # now update flags based on current antenna flags
            if self.A1.antpol.flag['P1'] or self.A2.antpol.flag['P1']:
                flags['P11'] = True
            if self.A1.antpol.flag['P2'] or self.A2.antpol.flag['P1']:
                flags['P21'] = True
            if self.A1.antpol.flag['P1'] or self.A2.antpol.flag['P2']:
                flags['P12'] = True
            if self.A1.antpol.flag['P2'] or self.A2.antpol.flag['P2']:
                flags['P22'] = True

        if verify:  # Verify provided flags or default flags created above
            if self.A1.antpol.flag['P1'] or self.A2.antpol.flag['P1']:
                flags['P11'] = True
            if self.A1.antpol.flag['P2'] or self.A2.antpol.flag['P1']:
                flags['P21'] = True
            if self.A1.antpol.flag['P1'] or self.A2.antpol.flag['P2']:
                flags['P12'] = True
            if self.A1.antpol.flag['P2'] or self.A2.antpol.flag['P2']:
                flags['P22'] = True
                
        self.crosspol.update_flags(flags=flags, verify=verify)

        # Stack on to last value or update last value in stack
        for pol in ['P11', 'P12', 'P21', 'P22']: 
            if stack is True:
                self.flag_stack[pol] = NP.append(self.flag_stack[pol], self.crosspol.flag[pol])
            else:
                if self.flag_stack[pol].size > 0:
                    self.flag_stack[pol][-1] = self.crosspol.flag[pol]
                # else:
                #     self.flag_stack[pol] = NP.asarray(self.crosspol.flag[pol]).reshape(-1)
            self.flag_stack[pol] = self.flag_stack[pol].astype(NP.bool)

    ############################################################################

    def update_old(self, label=None, Vt=None, t=None, timestamp=None,
                   location=None, wtsinfo=None, flags=None, gridfunc_freq=None,
                   ref_freq=None, do_correlate=None, stack=False,
                   verify_flags=True, verbose=False):

        """
        ------------------------------------------------------------------------
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
                   polarizations which are stored under keys 'P11', 'P12', 
                   'P21', and 'P22'. Default=None means no updates for flags.

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
                               containing the wtspos and wts information above 
                               as columns (x-loc [float], y-loc [float], wts
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
                   the cross-polarization keys in wtsinfo have number of 
                   elements equal to the number of frequency channels.

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

        stack      [boolean] If True (default), appends the updated flag 
                   and data to the end of the stack as a function of 
                   timestamp. If False, updates the last flag and data in 
                   the stack and does not append

        verify_flags     
                   [boolean] If True, verify and update the flags, if necessary.
                   Visibilities are checked for NaN values and if found, the
                   flag in the corresponding polarization is set to True. Flags 
                   of individual antennas forming a pair are checked and 
                   transferred to the visibility flags. Default=True 

        verbose    [boolean] If True, prints diagnostic and progress messages. 
                   If False (default), suppress printing such messages.
        ------------------------------------------------------------------------
        """

        if label is not None: self.label = label
        if location is not None: self.location = location
        if timestamp is not None: self.timestamp = timestamp
        # if latitude is not None: self.latitude = latitude

        # Proceed with interferometer updates only if timestamps align
        if (self.timestamp != self.A1.timestamp) or (self.timestamp != self.A2.timestamp):
            if verbose:
                print 'Interferometer timestamp does not match with the component antenna timestamp(s). Update for interferometer {0} will be skipped.'.format(self.label)
        else:
            self.timestamps += [copy.deepcopy(self.timestamp)]
            if t is not None:
                self.t = t
                self.f = self.f0 + self.channels()     
    
            if (Vt is not None) or (flags is not None):
                self.crosspol.update(Vt=Vt, flags=flags, verify=verify_flags)
    
            if do_correlate is not None:
                if do_correlate == 'FX':
                    self.FX()
                elif do_correlate == 'XF':
                    self.XF()
                else:
                    raise ValueError('Invalid specification for input parameter do_correlate.')
    
            self.update_flags(flags=None, stack=stack, verify=True)  # Re-check flags and stack
            for pol in ['P11', 'P12', 'P21', 'P22']:
                if self.Vt_stack[pol] is None:
                    self.Vt_stack[pol] = copy.deepcopy(self.crosspol.Vt[pol].reshape(1,-1))
                    self.Vf_stack[pol] = copy.deepcopy(self.crosspol.Vf[pol].reshape(1,-1))
                else:
                    if stack:
                        self.Vt_stack[pol] = NP.vstack((self.Vt_stack[pol], self.crosspol.Vt[pol].reshape(1,-1)))
                        self.Vf_stack[pol] = NP.vstack((self.Vf_stack[pol], self.crosspol.Vf[pol].reshape(1,-1)))
                    else:
                        self.Vt_stack[pol][-1,:] = copy.deepcopy(self.crosspol.Vt[pol].reshape(1,-1))
                        self.Vf_stack[pol][-1,:] = copy.deepcopy(self.crosspol.Vf[pol].reshape(1,-1))

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

    ############################################################################

    def update(self, update_dict=None, verbose=False):

        """
        ------------------------------------------------------------------------
        Updates the interferometer instance with newer attribute values. Updates 
        the visibility spectrum and timeseries and applies FX or XF operation.

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
    
            t          [vector] The time axis for the visibility time series. 
                       Default=None means no update to apply
    
            flags      [dictionary] holds boolean flags for each of the 4 
                       cross-polarizations which are stored under keys 'P11', 
                       'P12', 'P21', and 'P22'. Default=None means no updates 
                       for flags.
    
            Vt         [dictionary] holds cross-correlation time series under 4 
                       cross-polarizations which are stored under keys 'P11', 
                       'P12', 'P21', and 'P22'. Default=None implies no updates 
                       for Vt.
    
            aperture   [instance of class APR.Aperture] aperture information for 
                       the interferometer. Read docstring of class Aperture for 
                       details

            wtsinfo    [dictionary] consists of weights information for each of 
                       the four cross-polarizations under keys 'P11', 'P12', 
                       'P21', and 'P22'. Each of the values under the keys is a 
                       list of dictionaries. Length of list is equal to the 
                       number of frequency channels or one (equivalent to 
                       setting wtspos_scale to 'scale'.). The list is indexed by 
                       the frequency channel number. Each element in the list
                       consists of a dictionary corresponding to that frequency
                       channel. Each dictionary consists of these items with the
                       following keys:
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
                       dictionaries under the cross-polarization keys in 
                       wtsinfo have number of elements equal to the number of 
                       frequency channels.
    
            ref_freq   [Scalar] Positive value (in Hz) of reference frequency 
                       (used if gridfunc_freq is set to None or 'scale') at 
                       which wtspos is provided. If set to None, ref_freq is 
                       assumed to be equal to the center frequency in the class 
                       Interferometer's attribute. 
    
            do_correlate
                       [string] Indicates whether correlation operation is to be
                       performed after updates. Accepted values are 'FX' (for FX
                       operation) and 'XF' (for XF operation). Default=None 
                       means no correlating operation is to be performed after 
                       updates.
    
            stack      [boolean] If True (default), appends the updated flag 
                       and data to the end of the stack as a function of 
                       timestamp. If False, updates the last flag and data in 
                       the stack and does not append

            verify_flags     
                       [boolean] If True, verify and update the flags, if 
                       necessary. Visibilities are checked for NaN values and if 
                       found, the flag in the corresponding polarization is set 
                       to True. Flags of individual antennas forming a pair are 
                       checked and transferred to the visibility flags. 
                       Default=True 

        verbose    [boolean] If True, prints diagnostic and progress messages. 
                   If False (default), suppress printing such messages.
        ------------------------------------------------------------------------
        """

        label = None
        location = None
        timestamp = None
        t = None
        flags = None
        stack = False
        verify_flags = True
        Vt = None
        do_correlate = None
        wtsinfo = None
        gridfunc_freq = None
        ref_freq = None
        aperture = None

        if update_dict is not None:
            if not isinstance(update_dict, dict):
                raise TypeError('Input parameter containing updates must be a dictionary')

            if 'label' in update_dict: label = update_dict['label']
            if 'location' in update_dict: location = update_dict['location']
            if 'timestamp' in update_dict: timestamp = update_dict['timestamp']
            if 't' in update_dict: t = update_dict['t']
            if 'Vt' in update_dict: Vt = update_dict['Vt']
            if 'flags' in update_dict: flags = update_dict['flags']
            if 'stack' in update_dict: stack = update_dict['stack']
            if 'verify_flags' in update_dict: verify_flags = update_dict['verify_flags']            
            if 'do_correlate' in update_dict: do_correlate = update_dict['do_correlate']
            if 'wtsinfo' in update_dict: wtsinfo = update_dict['wtsinfo']
            if 'gridfunc_freq' in update_dict: gridfunc_freq = update_dict['gridfunc_freq']
            if 'ref_freq' in update_dict: ref_freq = update_dict['ref_freq']
            if 'aperture' in update_dict: aperture = update_dict['aperture']

        if label is not None: self.label = label
        if location is not None: self.location = location
        if timestamp is not None: self.timestamp = timestamp
        # if latitude is not None: self.latitude = latitude

        # Proceed with interferometer updates only if timestamps align
        if (self.timestamp != self.A1.timestamp) or (self.timestamp != self.A2.timestamp):
            if verbose:
                print 'Interferometer timestamp does not match with the component antenna timestamp(s). Update for interferometer {0} will be skipped.'.format(self.label)
        else:
            self.timestamps += [copy.deepcopy(self.timestamp)]
            if t is not None:
                self.t = t
                self.f = self.f0 + self.channels()     
    
            self.crosspol.update(Vt=Vt, flags=flags, verify=verify_flags)
    
            if do_correlate is not None:
                if do_correlate == 'FX':
                    self.FX()
                elif do_correlate == 'XF':
                    self.XF()
                else:
                    raise ValueError('Invalid specification for input parameter do_correlate.')
                self.update_flags(flags=None, stack=stack, verify=False)  # Stack flags. Flag verification has already been performed inside FX() or XF()

            for pol in ['P11', 'P12', 'P21', 'P22']:
                if not self.crosspol._init_data_on:
                    if self.Vt_stack[pol] is None:
                        if stack:
                            self.Vt_stack[pol] = copy.deepcopy(self.crosspol.Vt[pol].reshape(1,-1))
                            self.Vf_stack[pol] = copy.deepcopy(self.crosspol.Vf[pol].reshape(1,-1))
                    else:
                        if stack:
                            self.Vt_stack[pol] = NP.vstack((self.Vt_stack[pol], self.crosspol.Vt[pol].reshape(1,-1)))
                            self.Vf_stack[pol] = NP.vstack((self.Vf_stack[pol], self.crosspol.Vf[pol].reshape(1,-1)))
                        else:
                            self.Vt_stack[pol][-1,:] = copy.deepcopy(self.crosspol.Vt[pol].reshape(1,-1))
                            self.Vf_stack[pol][-1,:] = copy.deepcopy(self.crosspol.Vf[pol].reshape(1,-1))

            blc_orig = NP.copy(self.blc)
            trc_orig = NP.copy(self.trc)
            eps = 1e-6
    
            if aperture is not None:
                if isinstance(aperture, APR.Aperture):
                    self.aperture = copy.deepcopy(aperture)
                else:
                    raise TypeError('Update for aperture must be an instance of class Aperture.')

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

    ############################################################################
    
    def update_pp_old(self, update_dict=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Updates the interferometer instance with newer attribute values. Updates 
        the visibility spectrum and timeseries and applies FX or XF operation.
        Used internally when parallel processing is used. Not to be used by the
        user directly.

        Inputs:

        update_dict [dictionary] contains the following keys and values:

            label      [Scalar] A unique identifier (preferably a string) for 
                       the interferometer. Default=None means no update to apply
            
            latitude   [Scalar] Latitude of the interferometer's location. 
                       Default=None means no update to apply
            
            location   [Instance of GEOM.Point class] The location of the 
                       interferometer in local East, North, Up (ENU) coordinate 
                       system. Default=None means no update to apply
            
            timestamp  [Scalar] String or float representing the timestamp for 
                       the current attributes. Default=None means no update to 
                       apply
            
            t          [vector] The time axis for the visibility time series. 
                       Default=None means no update to apply
            
            flags      [dictionary] holds boolean flags for each of the 4 cross-
                       polarizations which are stored under keys 'P11', 'P12', 
                       'P21', and 'P22'. Default=None means no updates for 
                       flags.
            
            Vt         [dictionary] holds cross-correlation time series under 4 
                       cross-polarizations which are stored under keys 'P11', 
                       'P12', 'P21', and 'P22'. Default=None implies no updates 
                       for Vt.
            
            wtsinfo    [dictionary] consists of weights information for each of 
                       the four cross-polarizations under keys 'P11', 'P12', 
                       'P21', and 'P22'. Each of the values under the keys is a 
                       list of dictionaries. Length of list is equal to the 
                       number of frequency channels or one (equivalent to 
                       setting wtspos_scale to 'scale'.). The list is indexed by 
                       the frequency channel number. Each element in the list
                       consists of a dictionary corresponding to that frequency
                       channel. Each dictionary consists of these items with the
                       following keys:
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
                       dictionaries under the cross-polarization keys in wtsinfo 
                       have number of elements equal to the number of frequency 
                       channels.
            
            ref_freq   [Scalar] Positive value (in Hz) of reference frequency 
                       (used if gridfunc_freq is set to None or 'scale') at 
                       which wtspos is provided. If set to None, ref_freq is 
                       assumed to be equal to the center frequency in the class 
                       Interferometer's attribute. 
            
            do_correlate
                       [string] Indicates whether correlation operation is to be
                       performed after updates. Accepted values are 'FX' (for FX
                       operation) and 'XF' (for XF operation). Default=None 
                       means no correlating operation is to be performed after 
                       updates.
            
           stack       [boolean] If True (default), appends the updated flag 
                       and data to the end of the stack as a function of 
                       timestamp. If False, updates the last flag and data in 
                       the stack and does not append

            verify_flags     
                   [boolean] If True, verify and update the flags, if necessary.
                   Visibilities are checked for NaN values and if found, the
                   flag in the corresponding polarization is set to True. Flags 
                   of individual antennas forming a pair are checked and 
                   transferred to the visibility flags. Default=True 

        verbose    [boolean] If True, prints diagnostic and progress messages. 
                   If False (default), suppress printing such messages.
        ------------------------------------------------------------------------
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
        stack = False
        verify_flags = True
            
        if update_dict is not None:
            if not isinstance(update_dict, dict):
                raise TypeError('Input parameter containing updates must be a dictionary')

            if 'label' in update_dict: label = update_dict['label']
            if 'location' in update_dict: location = update_dict['location']
            if 'timestamp' in update_dict: timestamp = update_dict['timestamp']
            if 't' in update_dict: t = update_dict['t']
            if 'Vt' in update_dict: Vt = update_dict['Vt']
            if 'flags' in update_dict: flags = update_dict['flags']
            if 'stack' in update_dict: stack = update_dict['stack']
            if 'verify_flags' in update_dict: verify_flags = update_dict['verify_flags']            
            if 'do_correlate' in update_dict: do_correlate = update_dict['do_correlate']
            if 'wtsinfo' in update_dict: wtsinfo = update_dict['wtsinfo']
            if 'gridfunc_freq' in update_dict: gridfunc_freq = update_dict['gridfunc_freq']
            if 'ref_freq' in update_dict: ref_freq = update_dict['ref_freq']

        if label is not None: self.label = label
        if location is not None: self.location = location
        if timestamp is not None: self.timestamp = timestamp

        # Proceed with interferometer updates only if timestamps align
        if (self.timestamp != self.A1.timestamp) or (self.timestamp != self.A2.timestamp):
            if verbose:
                print 'Interferometer timestamp does not match with the component antenna timestamp(s). Update for interferometer {0} will be skipped.'.format(self.label)
        else:
            self.timestamps += [copy.deepcopy(self.timestamp)]
            if t is not None:
                self.t = t
                self.f = self.f0 + self.channels()     
    
            if (Vt is not None) or (flags is not None):
                self.crosspol.update(Vt=Vt, flags=flags, verify=verify_flags)
    
            if do_correlate is not None:
                if do_correlate == 'FX':
                    self.FX()
                elif do_correlate == 'XF':
                    self.XF()
                else:
                    raise ValueError('Invalid specification for input parameter do_correlate.')
    
            self.update_flags(flags=None, stack=stack, verify=True)  # Re-check flags and stack
            for pol in ['P11', 'P12', 'P21', 'P22']:
                if self.Vt_stack[pol] is None:
                    self.Vt_stack[pol] = copy.deepcopy(self.crosspol.Vt[pol].reshape(1,-1))
                    self.Vf_stack[pol] = copy.deepcopy(self.crosspol.Vf[pol].reshape(1,-1))
                else:
                    if stack:
                        self.Vt_stack[pol] = NP.vstack((self.Vt_stack[pol], self.crosspol.Vt[pol].reshape(1,-1)))
                        self.Vf_stack[pol] = NP.vstack((self.Vf_stack[pol], self.crosspol.Vf[pol].reshape(1,-1)))
                    else:
                        self.Vt_stack[pol][-1,:] = copy.deepcopy(self.crosspol.Vt[pol].reshape(1,-1))
                        self.Vf_stack[pol][-1,:] = copy.deepcopy(self.crosspol.Vf[pol].reshape(1,-1))
    
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

    ############################################################################

    def update_pp(self, update_dict=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Updates the interferometer instance with newer attribute values. Updates 
        the visibility spectrum and timeseries and applies FX or XF operation.
        Used internally when parallel processing is used. Not to be used by the
        user directly.

        See member function update() for details on inputs.
        ------------------------------------------------------------------------
        """

        self.update(update_dict=update_dict, verbose=verbose)
        return self

    ############################################################################

    def stack(self, on_flags=True, on_data=True):

        """
        ------------------------------------------------------------------------
        Stacks and computes visibilities and flags from the individual antennas 
        in the pair.

        Inputs:

        on_flags  [boolean] if set to True (default), combines the time-stacked
                  electric field flags from individual antennas from the 
                  common timestamps into time-stacked visibility flags

        on_data   [boolean] if set to True (default), combines the time-stacked
                  electric fields from individual antennas from the common
                  timestamps into time-stacked visibilities
        ------------------------------------------------------------------------
        """

        ts1 = NP.asarray(self.A1.timestamps)
        ts2 = NP.asarray(self.A2.timestamps)
        common_ts = NP.intersect1d(ts1, ts2, assume_unique=True)
        ind1 = NP.in1d(ts1, common_ts, assume_unique=True)
        ind2 = NP.in1d(ts2, common_ts, assume_unique=True)

        self.timestamps = common_ts.tolist()

        if on_data:
            self.FX_on_stack()

        if on_flags:
            self.flags_on_stack()

    ############################################################################

    def stack_pp(self, on_flags=True, on_data=True):

        """
        ------------------------------------------------------------------------
        Stacks and computes visibilities and flags from the individual antennas 
        in the pair. To be used internally as a wrapper for stack() in case of
        parallel processing. Not to be used directly by the user.

        Inputs:

        on_flags  [boolean] if set to True (default), combines the time-stacked
                  electric field flags from individual antennas from the 
                  common timestamps into time-stacked visibility flags

        on_data   [boolean] if set to True (default), combines the time-stacked
                  electric fields from individual antennas from the common
                  timestamps into time-stacked visibilities
        ------------------------------------------------------------------------
        """

        self.stack(on_flags=on_flags, on_data=on_data)
        return self

    ############################################################################

    def accumulate(self, tbinsize=None):

        """
        ------------------------------------------------------------------------
        Accumulate and average visibility spectra across timestamps under 
        different polarizations depending on the time bin size for the 
        corresponding polarization.

        Inputs:

        tbinsize [scalar or dictionary] Contains bin size of timestamps while
                 stacking. Default = None means all visibility spectra over all
                 timestamps are averaged. If scalar, the same (positive) value 
                 applies to all polarizations. If dictionary, timestamp bin size
                 (positive) is provided under each key 'P11', 'P12', 'P21', 
                 'P22'. If any of the keys is missing the visibilities for that 
                 polarization are averaged over all timestamps.
        ------------------------------------------------------------------------
        """

        timestamps = NP.asarray(self.timestamps).astype(NP.float)
        Vf_acc = {}
        twts = {}
        Vf_avg = {}
        for pol in ['P11', 'P12', 'P21', 'P22']:
            Vf_acc[pol] = None
            Vf_avg[pol] = None
            twts[pol] = []

        if tbinsize is None:   # Average visibilities across all timestamps
            for pol in ['P11', 'P12', 'P21', 'P22']:
                unflagged_ind = NP.logical_not(self.flag_stack[pol])
                Vf_acc[pol] = NP.nansum(self.Vf_stack[pol][unflagged_ind,:], axis=0, keepdims=True)
                twts[pol] = NP.sum(unflagged_ind).astype(NP.float).reshape(-1,1)
                # twts[pol] = NP.asarray(len(self.timestamps) - NP.sum(self.flag_stack[pol])).reshape(-1,1)
            self.tbinsize = tbinsize
        elif isinstance(tbinsize, (int, float)): # Apply same time bin size to all polarizations 
            eps = 1e-10
            tbins = NP.arange(timestamps.min(), timestamps.max(), tbinsize)
            tbins = NP.append(tbins, timestamps.max()+eps)
            for pol in ['P11', 'P12', 'P21', 'P22']:
                counts, tbin_edges, tbinnum, ri = OPS.binned_statistic(timestamps, statistic='count', bins=tbins)
                for binnum in range(counts.size):
                    ind = ri[ri[binnum]:ri[binnum+1]]
                    unflagged_ind = NP.logical_not(self.flag_stack[pol][ind])
                    twts[pol] += [NP.sum(unflagged_ind)]
                    # twts[pol] += [counts[binnum] - NP.sum(self.flag_stack[pol][ind])]
                    if Vf_acc[pol] is None:
                        Vf_acc[pol] = NP.nansum(self.Vf_stack[pol][ind[unflagged_ind],:], axis=0, keepdims=True)
                    else:
                        Vf_acc[pol] = NP.vstack((Vf_acc[pol], NP.nansum(self.Vf_stack[pol][ind[unflagged_ind],:], axis=0, keepdims=True)))
                twts[pol] = NP.asarray(twts[pol]).astype(NP.float).reshape(-1,1)
            self.tbinsize = tbinsize
        elif isinstance(tbinsize, dict): # Apply different time binsizes to corresponding polarizations
            tbsize = {}
            for pol in ['P11', 'P12', 'P21', 'P22']:
                if pol not in tbinsize:
                    unflagged_ind = NP.logical_not(self.flag_stack[pol])
                    Vf_acc[pol] = NP.nansum(self.Vf_stack[pol][unflagged_ind,:], axis=0, keepdims=True)
                    twts[pol] = NP.sum(unflagged_ind).astype(NP.float).reshape(-1,1)
                    # twts[pol] = NP.asarray(len(self.timestamps) - NP.sum(self.flag_stack[pol])).reshape(-1,1)
                    tbsize[pol] = None
                elif isinstance(tbinsize[pol], (int,float)):
                    eps = 1e-10
                    tbins = NP.arange(timestamps.min(), timestamps.max(), tbinsize[pol])
                    tbins = NP.append(tbins, timestamps.max()+eps)
                    
                    counts, tbin_edges, tbinnum, ri = OPS.binned_statistic(timestamps, statistic='count', bins=tbins)
                    for binnum in range(counts.size):
                        ind = ri[ri[binnum]:ri[binnum+1]]
                        unflagged_ind = NP.logical_not(self.flag_stack[pol][ind])
                        twts[pol] += [NP.sum(unflagged_ind)]
                        # twts[pol] += [counts[binnum] - NP.sum(self.flag_stack[pol][ind])]
                        if Vf_acc[pol] is None:
                            Vf_acc[pol] = NP.nansum(self.Vf_stack[pol][ind[unflagged_ind],:], axis=0, keepdims=True)
                        else:
                            Vf_acc[pol] = NP.vstack((Vf_acc[pol], NP.nansum(self.Vf_stack[pol][ind[unflagged_ind],:], axis=0, keepdims=True)))
                    twts[pol] = NP.asarray(twts[pol]).astype(NP.float).reshape(-1,1)
                    tbsize[pol] = tbinsize[pol]
                else:
                    unflagged_ind = NP.logical_not(self.flag_stack[pol])
                    Vf_acc[pol] = NP.nansum(self.Vf_stack[pol][unflagged_ind,:], axis=0, keepdims=True)
                    twts[pol] = NP.sum(unflagged_ind).astype(NP.float).reshape(-1,1)
                    # twts[pol] = NP.asarray(len(self.timestamps) - NP.sum(self.flag_stack[pol])).reshape(-1,1)
                    tbsize[pol] = None
            self.tbinsize = tbsize

        # Compute the average from the accumulated visibilities
        for pol in ['P11', 'P12', 'P21', 'P22']:
            Vf_avg[pol] = Vf_acc[pol] / twts[pol]

        self.Vf_avg = Vf_avg
        self.twts = twts

################################################################################

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
                              'twts'         [scalar] if positive, indicates
                                             the number of timestamps that 
                                             have gone into the measurement of  
                                             complex Vf made by the 
                                             interferometer under the 
                                             specific polarization. If zero, it
                                             indicates no unflagged timestamp 
                                             data was found for the 
                                             interferometer and will not 
                                             contribute to the complex grid 
                                             illumination and visibilities
                              'twts'         [scalar] denotes the number of 
                                             timestamps for which the 
                                             interferometer data was not flagged
                                             which were used in stacking and 
                                             averaging
                              'gridind'      [numpy vector] one-dimensional 
                                             index into the three-dimensional 
                                             grid locations where the 
                                             interferometer contributes 
                                             illumination and visibilities. The 
                                             one-dimensional indices are 
                                             obtained using numpy's 
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
                  'per_bl2grid'
                              [list] each element in the list is a dictionary
                              corresponding to an interferometer with information 
                              on its mapping and contribution to the grid. Each 
                              dictionary has the following keys and values:
                              'label'        [tuple of two strings] 
                                             interferometer label
                              'f_gridind'    [numpy array] mapping information 
                                             with indices to the frequency axis
                                             of the grid
                              'u_gridind'    [numpy array] mapping information 
                                             with indices to the u-axis
                                             of the grid. Must be of same size 
                                             as array under 'f_gridind'
                              'v_gridind'    [numpy array] mapping information 
                                             with indices to the v-axis
                                             of the grid. Must be of same size 
                                             as array under 'f_gridind'
                              'per_bl_per_freq_norm_wts'
                                             [numpy array] mapping information 
                                             on the (complex) normalizing 
                                             multiplicative factor required to 
                                             make the sum of illumination/weights 
                                             per interferometer per frequency on 
                                             the grid equal to unity. Must be of 
                                             same size as array under 'f_gridind'
                              'illumination' [numpy array] Complex aperture 
                                             illumination/weights contributed
                                             by the interferometer onto the grid. 
                                             The grid pixels to which it 
                                             contributes is given by 'f_gridind', 
                                             'u_gridind', 'v_gridind'. Must be of 
                                             same size as array under 'f_gridind'
                              'Vf'           [numpy array] Complex visibilities 
                                             contributed by the 
                                             interferometer onto the grid. The 
                                             grid pixels to which it contributes 
                                             is given by 'f_gridind', 
                                             'u_gridind', 'v_gridind'. Must be of 
                                             same size as array under 'f_gridind'
                  'all_bl2grid'
                              [dictionary] contains the combined information of
                              mapping of all interferometers to the grid. It 
                              consists of the following keys and values:
                              'blind'        [numpy array] all interferometer 
                                             indices (to attribute ordered 
                                             labels) that map to the uvf-grid
                              'u_gridind'    [numpy array] all indices to the 
                                             u-axis of the uvf-grid mapped to by 
                                             all interferometers whose indices 
                                             are given in key 'blind'. Must be 
                                             of same size as the array under key 
                                             'blind'
                              'v_gridind'    [numpy array] all indices to the 
                                             v-axis of the uvf-grid mapped to by 
                                             all interferometers whose indices 
                                             are given in key 'blind'. Must be 
                                             of same size as the array under key 
                                             'blind'
                              'f_gridind'    [numpy array] all indices to the 
                                             f-axis of the uvf-grid mapped to by 
                                             all interferometers whose indices 
                                             are given in key 'blind'. Must be 
                                             of same size as the array under key 
                                             'blind'
                              'indNN_list'   [list of lists] Each item in the 
                                             top level list corresponds to an 
                                             interferometer in the same order as 
                                             in the attribute ordered_labels. 
                                             Each of these items is another list 
                                             consisting of the unraveled grid 
                                             indices it contributes to. The 
                                             unraveled indices are what are used 
                                             to obtain the u-, v- and f-indices 
                                             in the grid using a conversion 
                                             assuming f is the first axis, v is 
                                             the second and u is the third
                              'illumination' [numpy array] complex values of 
                                             aperture illumination contributed 
                                             by all interferometers to the grid. 
                                             The interferometer indices are in 
                                             'blind' and the grid indices are 
                                             in 'u_gridind', 'v_gridind' and 
                                             'f_gridind'. Must be of same size as 
                                             these indices
                              'per_bl_per_freq_norm_wts'
                                             [numpy array] mapping information 
                                             on the (complex) normalizing 
                                             multiplicative factor required to 
                                             make the sum of illumination or 
                                             weights per interferometer per 
                                             frequency on the grid equal to 
                                             unity. This is appended for all 
                                             interferometers together. Must be of 
                                             same size as array under 
                                             'illumination'
                              'Vf'           [numpy array] Complex visibilities 
                                             contributed by all 
                                             interferometers onto the grid. The 
                                             grid pixels to which it contributes 
                                             is given by 'f_gridind', 
                                             'u_gridind', 'v_gridind'. Must be of 
                                             same size as array under 'f_gridind' 
                                             and 'illumination'
                                             
    bl2grid_mapper
                  [sparse matrix] contains the interferometer array to grid 
                  mapping information in sparse matrix format. When converted 
                  to a dense array, it will have dimensions nrows equal to size 
                  of the 3D cube and ncols equal to number of visibility spectra 
                  of all interferometers over all channels. In other words, 
                  nrows = nu x nv x nchan and ncols = n_bl x nchan. Dot product
                  of this matrix with flattened visibility spectra or 
                  interferometer weights will give the 3D cubes of gridded 
                  visibilities and interferometer array illumination 
                  respectively

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

    refresh_antenna_pairs()
                    Refresh the individual antennas in the interferometer(s) 
                    with the information in the Antenna instances in the 
                    attribute antenna_array which is an instance of class 
                    AntennaArray

    FX()            Computes the Fourier transform of the cross-correlated time 
                    series of the interferometer pairs in the interferometer 
                    array to compute the visibility spectra

    XF()            Computes the visibility spectra by cross-multiplying the 
                    electric field spectra for all the interferometer pairs in 
                    the interferometer array

    get_visibilities()
                    Routine to return the interferometer labels, time-based 
                    weights and visibilities (sorted by interferometer label 
                    if specified) based on selection criteria specified by 
                    flags, timestamps, frequency channels, labels and data pool 
                    (most recent, stack, averaged, etc.)

    stack()         Stacks and computes visibilities and flags for all the 
                    interferometers in the interferometer array from the 
                    individual antennas in the pair.

    accumulate()    Accumulate and average visibility spectra across timestamps 
                    under different polarizations depending on the time bin 
                    size for the corresponding polarization for all 
                    interferometers in the interferometer array

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

    grid_convolve_new() 
                    Routine to project the complex illumination power pattern 
                    and the visibilities on the grid from the interferometer 
                    array

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

    quick_beam_synthesis()
                    A quick generator of synthesized beam using interferometer 
                    array grid illumination pattern using the center frequency. 
                    Not intended to be used rigorously but rather for comparison 
                    purposes and making quick plots

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
        self.bl2grid_mapper = {}  # contains the sparse mapping matrix

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

            self.grid_mapper[pol]['per_bl2grid'] = []
            self.grid_mapper[pol]['all_bl2grid'] = {}

            self.grid_illumination[pol] = None
            self.grid_Vf[pol] = None
            self._bl_contribution[pol] = {}

            self.bl2grid_mapper[pol] = None

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

    ############################################################################

    def __str__(self):
        printstr = '\n-----------------------------------------------------------------'
        printstr += '\n Instance of class "{0}" in module "{1}".\n Holds the following "Interferometer" class instances with labels:\n '.format(self.__class__.__name__, self.__module__)
        printstr += str(self.interferometers.keys()).strip('[]')
        # printstr += '  '.join(sorted(self.interferometers.keys()))
        printstr += '\n Interferometer array bounds: blc = [{0[0]}, {0[1]}],\n\ttrc = [{1[0]}, {1[1]}]'.format(self.blc, self.trc)
        printstr += '\n Grid bounds: blc = [{0[0]}, {0[1]}],\n\ttrc = [{1[0]}, {1[1]}]'.format(self.grid_blc, self.grid_trc)
        printstr += '\n-----------------------------------------------------------------'
        return printstr

    ############################################################################

    def __add__(self, others):

        """
        ------------------------------------------------------------------------
        Operator overloading for adding interferometer(s)
    
        Inputs:
    
        others     [Instance of class InterferometerArray, dictionary holding
                   instance(s) of class Interferometer, list of instances of 
                   class Interferometer, or a single instance of class 
                   Interferometer] If a dictionary is provided, the keys should 
                   be the antenna labels and the values should be instances  of 
                   class Interferometer. If a list is provided, it should be a 
                   list of valid instances of class Interferometer. These 
                   instance(s) of class Interferometer will be added to the 
                   existing instance of InterferometerArray class.
        ------------------------------------------------------------------------
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

    ############################################################################

    def __radd__(self, others):

        """
        ------------------------------------------------------------------------
        Operator overloading for adding interferometer(s)
    
        Inputs:
    
        others     [Instance of class InterferometerArray, dictionary holding 
                   instance(s) of class Interferometer, list of instances of 
                   class Interferometer, or a single instance of class 
                   Interferometer] If a dictionary is provided, the keys should 
                   be the interferometer labels and the values should be 
                   instances of class Interferometer. If a list is provided, it 
                   should be a list of valid instances of class Interferometer. 
                   These instance(s) of class Interferometer will be added to 
                   the existing instance of InterferometerArray class.
        ------------------------------------------------------------------------
        """

        return self.__add__(others)

    ############################################################################

    def __sub__(self, others):

        """
        ------------------------------------------------------------------------
        Operator overloading for removing interferometer(s)
    
        Inputs:
    
        others     [Instance of class InterferometerArray, dictionary holding 
                   instance(s) of class Interferometer, list of instances of 
                   class Interferometer, list of strings containing 
                   interferometer labels or a single instance of class 
                   Interferometer] If a dictionary is provided, the keys should 
                   be the interferometer labels and the values should be 
                   instances of class Interferometer. If a list is provided, it 
                   should be a list of valid instances of class Interferometer. 
                   These instance(s) of class Interferometer will be removed 
                   from the existing instance of InterferometerArray class.
        ------------------------------------------------------------------------
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

    ############################################################################

    def add_interferometers(self, A=None):

        """
        ------------------------------------------------------------------------
        Routine to add interferometer(s) to the interferometer array instance. 
        A wrapper for operator overloading __add__() and __radd__()
    
        Inputs:
    
        A          [Instance of class InterferometerArray, dictionary holding 
                   instance(s) of class Interferometer, list of instances of 
                   class Interferometer, or a single instance of class 
                   Interferometer] If a dictionary is provided, the keys should 
                   be the interferometer labels and the values should be 
                   instances of class Interferometer. If a list is provided, it 
                   should be a list of valid instances of class Interferometer. 
                   These instance(s) of class Interferometer will be added to 
                   the existing instance of InterferometerArray class.
        ------------------------------------------------------------------------
        """

        if A is None:
            print 'No interferometer(s) supplied.'
        elif isinstance(A, (list, Interferometer)):
            self = self.__add__(A)
        else:
            print 'Input(s) is/are not instance(s) of class Interferometer.'

    ############################################################################

    def remove_interferometers(self, A=None):

        """
        ------------------------------------------------------------------------
        Routine to remove interferometer(s) from the interferometer array 
        instance. A wrapper for operator overloading __sub__()
    
        Inputs:
    
        A          [Instance of class InterferometerArray, dictionary holding 
                   instance(s) of class Interferometer, list of instances of 
                   class Interferometer, or a single instance of class 
                   Interferometer] If a dictionary is provided, the keys should 
                   be the interferometer labels and the values should be 
                   instances of class Interferometer. If a list is provided, it 
                   should be a list of valid instances of class Interferometer. 
                   These instance(s) of class Interferometer will be removed 
                   from the existing instance of InterferometerArray class.
        ------------------------------------------------------------------------
        """

        if A is None:
            print 'No interferometer specified for removal.'
        else:
            self = self.__sub__(A)

    ############################################################################

    def interferometers_containing_antenna(self, antenna_label):

        """
        ------------------------------------------------------------------------
        Find interferometer pairs which contain the specified antenna labels

        Inputs:

        antenna_label [list] List of antenna labels which will be searched for 
                      in the interferometer pairs in the interferometer array.

        Outputs:

        ant_pair_labels
                      [list] List of interferometer pair labels containing one 
                      of more of the specified antenna labels

        ant_order     [list] List of antenna order of antenna labels found in 
                      the interferometer pairs of the interferometer array. If 
                      the antenna label appears as the first antenna in the 
                      antenna pair, ant_order is assigned to 1 and if it is 
                      the second antenna in the pair, it is assigned to 2.
        ------------------------------------------------------------------------
        """

        ant_pair_labels = [ant_pair_label for ant_pair_label in self.interferometers if antenna_label in ant_pair_label]
        ant_order = [1 if ant_pair_label[0] == antenna_label else 2 for ant_pair_label in ant_pair_labels]

        return (ant_pair_labels, ant_order)

    ############################################################################

    def baseline_vectors(self, pol=None, flag=False, sort=True):
        
        """
        ------------------------------------------------------------------------
        Routine to return the interferometer label and baseline vectors (sorted 
        by interferometer label if specified)

        Keyword Inputs:

        pol      [string] select baselines of this polarization that are either 
                 flagged or unflagged as specified by input parameter flag. 
                 Allowed values are 'P11', 'P12', 'P21', and 'P22'. 
                 Default=None. This means all baselines are returned 
                 irrespective of the flags

        flag     [boolean] If False, return unflagged baselines, otherwise 
                 return flagged ones. Default=None means return all baselines
                 independent of flagging or polarization

        sort     [boolean] If True, returned interferometer information is 
                 sorted by interferometer's first antenna label. Default = True.

        Output:

        outdict  [dictionary] Output consists of a dictionary with the following 
                 keys and information:
                 'labels':    list of tuples of strings of interferometer labels
                 'baselines': baseline vectors of interferometers (3-column 
                              array)
        ------------------------------------------------------------------------
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

    ############################################################################

    def refresh_antenna_pairs(self, interferometer_labels=None,
                               antenna_labels=None):

        """
        ------------------------------------------------------------------------
        Refresh the individual antennas in the interferometer(s) with the 
        information in the Antenna instances in the attribute antenna_array
        which is an instance of class AntennaArray

        Inputs:

        interferometer_labels
                  [list] list of interferometer labels each given as a tuple 
                  of antenna labels. The antennas in these pairs are refreshed 
                  using the corresponding antenna instances in the attribute
                  antenna_array. Default = None.

        antenna_labels 
                  [list] list of antenna labels to determine which 
                  interferometers they contribute to. The antenna pairs in 
                  these interferometers are refreshed based on the current 
                  antenna instances in the attribute antenna_array. 
                  Default = None.

        If both input keywords interferometer_labels and antenna_labels are 
        set to None, all the interferometer instances are refreshed.
        ------------------------------------------------------------------------
        """

        ilabels = []
        if interferometer_labels is not None:
            if not isinstance(interferometer_labels, list):
                raise TypeError('Input keyword interferometer_labels must be a list')
            ilabels = antenna_labels
        if antenna_labels is not None:
            if not isinstance(interferometer_labels, list):
                raise TypeError('Input keyword interferometer_labels must be a list')
            ant_pair_labels, = self.interferometers_containing_antenna(antenna_labels)
            ilabels += ant_pair_labels

        if len(ilabels) == 0:
            ilabels = self.interferometers.keys()

        for antpair_label in ilabels:
            if antpair_label in self.interferometers:
                self.interferometers[antpair_label].refresh_antenna_pairs(A1=self.antenna_array.antennas[antpair_label[0]], A2=self.antenna_array.antennas[antpair_label[1]])

    ############################################################################

    def FX(self, parallel=False, nproc=None):

        """
        ------------------------------------------------------------------------
        Computes the Fourier transform of the cross-correlated time series of 
        the interferometer pairs in the interferometer array to compute the 
        visibility spectra

        Inputs:

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
        ------------------------------------------------------------------------
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

    ############################################################################

    def XF(self):

        """
        ------------------------------------------------------------------------
        Computes the visibility spectra by cross-multiplying the electric field
        spectra for all the interferometer pairs in the interferometer array
        ------------------------------------------------------------------------
        """
        
        if self.t is None:
            self.t = self.interferometers.itervalues().next().t

        if self.f is None:
            self.f = self.interferometers.itervalues().next().f

        if self.f0 is None:
            self.f0 = self.interferometers.itervalues().next().f0

        for label in self.interferometers:
            self.interferometers[label].XF()

        
    ############################################################################

    def get_visibilities_old(self, pol, flag=None, sort=True):

        """
        ------------------------------------------------------------------------
        Routine to return the interferometer label and visibilities (sorted by
        interferometer label if specified)

        Keyword Inputs:

        pol      [string] select baselines of this polarization that are either 
                 flagged or unflagged as specified by input parameter flag. 
                 Allowed values are 'P11', 'P12', 'P21', and 'P22'. Only one of
                 these values must be specified.

        flag     [boolean] If False, return visibilities of unflagged baselines,
                 otherwise return flagged ones. Default=None means all 
                 visibilities independent of flagging are returned.

        sort     [boolean] If True, returned interferometer information is 
                 sorted by interferometer's first antenna label. Default = True.

        Output:

        outdict  [dictionary] Output consists of a dictionary with the following 
                 keys and information:
                 'labels':    Contains a numpy array of strings of 
                              interferometer labels
                 'visibilities': 
                              interferometer visibilities (n_bl x nchan array)
        ------------------------------------------------------------------------
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

    ############################################################################

    def get_visibilities(self, pol, flag=None, tselect=None, fselect=None,
                             bselect=None, datapool=None, sort=True):

        """
        ------------------------------------------------------------------------
        Routine to return the interferometer labels, time-based weights and 
        visibilities (sorted by interferometer label if specified) based on 
        selection criteria specified by flags, timestamps, frequency channels,
        labels and data pool (most recent, stack, averaged, etc.)

        Keyword Inputs:

        pol      [string] select baselines of this polarization that are either 
                 flagged or unflagged as specified by input parameter flag. 
                 Allowed values are 'P11', 'P12', 'P21', and 'P22'. Only one of
                 these values must be specified.

        flag     [boolean] If False, return visibilities of unflagged baselines,
                 otherwise return flagged ones. Default=None means all 
                 visibilities independent of flagging are returned.

        tselect  [scalar, list, numpy array] timestamp index for visibilities
                 selection. For most recent visibility, it must be set to -1.
                 For all other selections, indices in tselect must be in the 
                 valid range of indices along time axis for stacked and 
                 averaged visibilities. Default=None means most recent data is
                 selected. 

        fselect  [scalar, list, numpy array] frequency channel index for 
                 visibilities selection. Indices must be in the valid range of
                 indices along the frequency axis for visibilities. 
                 Default=None selects all frequency channels

        bselect  [list of tuples] labels of interferometers to select. If set 
                 to None (default) all interferometers are selected. 

        datapool [string] denotes the data pool from which visibilities are to
                 be selected. Accepted values are 'current', 'stack', 'avg' and
                 None (default, same as 'current'). If set to None or 
                 'current', the value in tselect is ignored and only 
                 visibilities of the most recent timestamp are selected. If set
                 to None or 'current' the attribute Vf_stack is checked first 
                 and if unavailable, attribute crosspol.Vf is used. For 'stack'
                 and 'avg', attributes Vf_stack and Vf_avg are used 
                 respectively

        sort     [boolean] If True, returned interferometer information is 
                 sorted by interferometer's first antenna label. Default=True.

        Output:

        outdict  [dictionary] Output consists of a dictionary with the following 
                 keys and information:
                 'labels'        [list of tuples] Contains a list of 
                                 interferometer labels
                 'visibilities'  [list or numpy array] interferometer 
                                 visibilities under the specified polarization. 
                                 In general, it is a list of 
                                 numpy arrays where each array in the list 
                                 corresponds to 
                                 an individual interferometer and the size of
                                 each numpy array is n_ts x nchan. If input 
                                 keyword flag is set to None, the visibilities 
                                 are rearranged into a numpy array of size
                                 n_ts x n_bl x nchan. 
                 'twts'          [list or numpy array] weights based on flags
                                 along time axis under the specified 
                                 polarization. In general it is a list of numpy
                                 arrays where each array in the list corresponds 
                                 to an individual interferometer and the size 
                                 of each array is n_ts x 1. If input 
                                 keyword flag is set to None, the time weights 
                                 are rearranged into a numpy array of size
                                 n_ts x n_bl x 1
        ------------------------------------------------------------------------
        """

        if not isinstance(sort, bool):
            raise TypeError('sort keyword has to be a Boolean value.')

        if bselect is None:
            labels = self.interferometers.keys()
        elif isinstance(bselect, list):
            labels = [label for label in bselect if label in self.interferometers]
            
        if sort:
            labels_orig = copy.deepcopy(labels)
            labels = [label for label in sorted(labels_orig, key=lambda tup: tup[0])]

        visinfo = [self.interferometers[label].get_visibilities(pol, flag=flag, tselect=tselect, fselect=fselect, datapool=datapool) for label in labels]
      
        outdict = {}
        outdict['labels'] = labels
        outdict['twts'] = [vinfo['twts'] for vinfo in visinfo]
        outdict['visibilities'] = [vinfo['visibilities'] for vinfo in visinfo]
        if flag is None:
            outdict['visibilities'] = NP.swapaxes(NP.asarray(outdict['visibilities']), 0, 1)
            outdict['twts'] = NP.swapaxes(NP.asarray(outdict['twts']), 0, 1)
            outdict['twts'] = outdict['twts'][:,:,NP.newaxis]

        return outdict

    ############################################################################
    
    def stack(self, on_flags=True, on_data=True, parallel=False, nproc=None):

        """
        ------------------------------------------------------------------------
        Stacks and computes visibilities and flags for all the interferometers 
        in the interferometer array from the individual antennas in the pair.

        Inputs:

        on_flags  [boolean] if set to True (default), combines the time-stacked
                  electric field flags from individual antennas from the 
                  common timestamps into time-stacked visibility flags

        on_data   [boolean] if set to True (default), combines the time-stacked
                  electric fields from individual antennas from the common
                  timestamps into time-stacked visibilities

        parallel  [boolean] specifies if parallelization is to be invoked. 
                  False (default) means only serial processing

        nproc     [integer] specifies number of independent processes to spawn.
                  Default = None, means automatically determines the number of 
                  process cores in the system and use one less than that to 
                  avoid locking the system for other processes. Applies only 
                  if input parameter 'parallel' (see above) is set to True. 
                  If nproc is set to a value more than the number of process
                  cores in the system, it will be reset to number of process 
                  cores in the system minus one to avoid locking the system out 
                  for other processes
        ------------------------------------------------------------------------
        """

        if parallel:
            if nproc is None:
                nproc = max(MP.cpu_count()-1, 1) 
            else:
                nproc = min(nproc, max(MP.cpu_count()-1, 1))

            list_of_perform_flag_stack = [on_flags] * len(self.interferometers)
            list_of_perform_data_stack = [on_data] * len(self.interferometers)

            pool = MP.Pool(processes=nproc)
            updated_interferometers = pool.map(unwrap_interferometer_stack, IT.izip(self.interferometers.values(), list_of_perform_flag_stack, list_of_perform_data_stack))
            pool.close()
            pool.join()
            for interferometer in updated_interferometers: 
                self.interferometers[interferometer.label] = interferometer
            del updated_interferometers
        else:
            for label in self.interferometers:
                self.interferometers[label].stack(on_flags=on_flags, on_data=on_data)

    ############################################################################

    def accumulate(self, tbinsize=None):

        """
        ------------------------------------------------------------------------
        Accumulate and average visibility spectra across timestamps under 
        different polarizations depending on the time bin size for the 
        corresponding polarization for all interferometers in the 
        interferometer array

        Inputs:

        tbinsize [scalar or dictionary] Contains bin size of timestamps while
                 stacking. Default = None means all visibility spectra over all
                 timestamps are averaged. If scalar, the same (positive) value 
                 applies to all polarizations. If dictionary, timestamp bin size
                 (positive) is provided under each key 'P11', 'P12', 'P21', 
                 'P22'. If any of the keys is missing the visibilities for that 
                 polarization are averaged over all timestamps.
        ------------------------------------------------------------------------
        """

        for label in self.interferometers:
            self.interferometers[label].accumulate(tbinsize=tbinsize)

    ############################################################################

    def grid(self, uvspacing=0.5, uvpad=None, pow2=True):
        
        """
        ------------------------------------------------------------------------
        Routine to produce a grid based on the interferometer array 

        Inputs:

        uvspacing   [Scalar] Positive value indicating the maximum uv-spacing
                    desirable at the lowest wavelength (max frequency). 
                    Default = 0.5

        xypad       [List] Padding to be applied around the interferometer 
                    locations before forming a grid. List elements should be 
                    positive. If it is a one-element list, the element is 
                    applicable to both x and y axes. If list contains three or 
                    more elements, only the first two elements are considered 
                    one for each axis. Default = None.

        pow2        [Boolean] If set to True, the grid is forced to have a size 
                    a next power of 2 relative to the actual sie required. If 
                    False, gridding is done with the appropriate size as 
                    determined by uvspacing. Default = True.
        ------------------------------------------------------------------------
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

    ############################################################################

    def grid_convolve(self, pol=None, antpairs=None, unconvolve_existing=False,
                      normalize=False, method='NN', distNN=NP.inf, tol=None,
                      maxmatch=None, identical_interferometers=True,
                      gridfunc_freq=None, mapping='weighted', wts_change=False,
                      parallel=False, nproc=None, pp_method='pool', verbose=True): 

        """
        ------------------------------------------------------------------------
        Routine to project the complex illumination power pattern and the 
        visibilities on the grid. It can operate on the entire interferometer 
        array or incrementally project the visibilities and complex illumination 
        power patterns from specific interferometers on to an already existing 
        grid. (The latter is not implemented yet)

        Inputs:

        pol         [String] The polarization to be gridded. Can be set to 
                    'P11', 'P12', 'P21' or 'P22'. If set to None, gridding for 
                    all the polarizations is performed. Default = None

        antpairs    [instance of class InterferometerArray, single instance or 
                    list of instances of class Interferometer, or a dictionary 
                    holding instances of class Interferometer] If a dictionary 
                    is provided, the keys should be the interferometer labels 
                    and the values should be instances of class Interferometer. 
                    If a list is provided, it should be a list of valid 
                    instances of class Interferometer. These instance(s) of 
                    class Interferometer will be merged to the existing grid 
                    contained in the instance of InterferometerArray class. If 
                    ants is not provided (set to None), the gridding operations 
                    will be performed on the entire set of interferometers 
                    contained in the instance of class InterferometerArray. 
                    Default=None.

        unconvolve_existing
                   [Boolean] Default = False. If set to True, the effects of
                   gridding convolution contributed by the interferometer(s) 
                   specified will be undone before updating the interferometer 
                   measurements on the grid, if the interferometer(s) is/are 
                   already found to in the set of interferometers held by the 
                   instance of InterferometerArray. If False and if one or more 
                   interferometer instances specified are already found to be 
                   held in the instance of class InterferometerArray, the code 
                   will stop raising an error indicating the gridding oepration 
                   cannot proceed. 

        normalize  [Boolean] Default = False. If set to True, the gridded 
                   weights are divided by the sum of weights so that the 
                   gridded weights add up to unity. (Need to work on 
                   normaliation)

        method     [string] The gridding method to be used in applying the 
                   interferometer weights on to the interferometer array grid. 
                   Accepted values are 'NN' (nearest neighbour - default), 'CS' 
                   (cubic spline), or 'BL' (Bi-linear). In case of applying grid 
                   weights by 'NN' method, an optional distance upper bound for 
                   the nearest neighbour can be provided in the parameter distNN 
                   to prune the search and make it efficient. Currently, only 
                   the nearest neighbour method is operational.

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
                   considered for nearest neighbour lookup. Default = None 
                   implies all lookup values will be considered for nearest 
                   neighbour determination. tol is to be interpreted as a 
                   minimum value considered as significant in the lookup table. 

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
                   in this attribute under the specific polarization are the 
                   same as the number of frequency channels.

        mapping    [string] indicates the type of mapping between baseline 
                   locations and the grid locations. Allowed values are 
                   'sampled' and 'weighted' (default). 'sampled' means only the 
                   baseline measurement closest ot a grid location contributes 
                   to that grid location, whereas, 'weighted' means that all the 
                   baselines contribute in a weighted fashion to their nearest 
                   grid location. The former is faster but possibly discards 
                   baseline data whereas the latter is slower but includes all 
                   data along with their weights.

        wts_change [boolean] indicates if weights and/or their lcoations have 
                   changed from the previous intergration or snapshot. 
                   Default=False means they have not changed. In such a case the 
                   baseline-to-grid mapping and grid illumination pattern do not 
                   have to be determined, and mapping and values from the 
                   previous snapshot can be used. If True, a new mapping has to 
                   be determined.

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
        ------------------------------------------------------------------------
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

                        vis_dict = self.get_visibilities_old(cpol, flag=False, sort=True)
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

                    # Vf_dict = self.get_visibilities_old(cpol, flag=None, sort=True)
                    # Vf = Vf_dict['visibilities'].astype(NP.complex64)  # n_bl x nchan

                    Vf_dict = self.get_visibilities(cpol, flag=None, tselect=-1, fselect=None, bselect=None, datapool='avg', sort=True)
                    Vf = Vf_dict['visibilities'].astype(NP.complex64)  #  (n_ts=1) x n_bl x nchan
                    Vf = NP.squeeze(Vf, axis=0)  # n_bl x nchan
                    if Vf.shape[0] != n_bl:
                        raise ValueError('Encountered unexpected behavior. Need to debug.')
                    bl_labels = Vf_dict['labels']
                    twts = Vf_dict['twts']  # (n_ts=1) x n_bl x (nchan=1)
                    twts = NP.squeeze(twts, axis=(0,2))  # n_bl

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
                                            self.grid_mapper[cpol]['labels'][label]['twts'] = twts[bl_labels.index(label)]
                                            # self.grid_mapper[cpol]['labels'][label]['flag'] = self.interferometers[label].crosspol.flag[cpol]
        
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
                                        self.grid_mapper[cpol]['labels'][label]['twts'] = twts[bl_labels.index(label)]
                                        # self.grid_mapper[cpol]['labels'][label]['flag'] = self.interferometers[label].crosspol.flag[cpol]
    
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
    
                                if nproc is not None:
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
                                    self.grid_mapper[cpol]['labels'][label]['twts'] = twts[bl_labels.index(label)]
                                    # self.grid_mapper[cpol]['labels'][label]['flag'] = self.interferometers[label].crosspol.flag[cpol]
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
                                        self.grid_mapper[cpol]['labels'][label]['twts'] = twts[bl_labels.index(label)]    
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
                                        self.grid_mapper[cpol]['labels'][label]['twts'] = twts[bl_labels.index(label)]    
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
                                    self.grid_mapper[cpol]['labels'][label]['twts'] = twts[bl_labels.index(label)]                                        
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

    ############################################################################

    def grid_convolve_new(self, pol=None, normalize=False, method='NN',
                          distNN=NP.inf, identical_interferometers=True,
                          cal_loop=False, gridfunc_freq=None, wts_change=False,
                          parallel=False, nproc=None, pp_method='pool',
                          verbose=True): 

        """
        ------------------------------------------------------------------------
        Routine to project the complex illumination power pattern and the 
        visibilities on the grid from the interferometer array

        Inputs:

        pol        [String] The polarization to be gridded. Can be set to 'P1' 
                   or 'P2'. If set to None, gridding for all the polarizations 
                   is performed. Default = None

        normalize  [Boolean] Default = False. If set to True, the gridded 
                   weights are divided by the sum of weights so that the gridded 
                   weights add up to unity. (Need to work on normaliation)

        method     [string] The gridding method to be used in applying the 
                   interferometer weights on to the interferometer array grid. 
                   Accepted values are 'NN' (nearest neighbour - default), 'CS' 
                   (cubic spline), or 'BL' (Bi-linear). In case of applying grid 
                   weights by 'NN' method, an optional distance upper bound for 
                   the nearest neighbour can be provided in the parameter distNN 
                   to prune the search and make it efficient. Currently, only 
                   the nearest neighbour method is operational.

        distNN     [scalar] A positive value indicating the upper bound on 
                   distance to the nearest neighbour in the gridding process. It 
                   has units of distance, the same units as the interferometer 
                   attribute location and interferometer array attribute gridx 
                   and gridy. Default is NP.inf (infinite distance). It will be 
                   internally converted to have same units as interferometer 
                   attributes wtspos (units in number of wavelengths). To ensure
                   all relevant pixels in the grid, the search distance used 
                   internally will be a fraction more than distNN

        identical_interferometers
                   [boolean] indicates if all interferometer elements are to be
                   treated as identical. If True (default), they are identical
                   and their gridding kernels are identical. If False, they are
                   not identical and each one has its own gridding kernel.

        cal_loop   [boolean] If True, the calibration loop is assumed to be ON 
                   and hence the calibrated electric fields are set in the 
                   calibration loop. If False (default), the calibration loop is
                   assumed to be OFF and the current electric fields are assumed 
                   to be the calibrated data to be mapped to the grid 
                   via gridding convolution.

        gridfunc_freq
                   [String scalar] If set to None (not provided) or to 'scale'
                   assumes that attribute wtspos is given for a
                   reference frequency which need to be scaled for the frequency
                   channels. Will be ignored if the number of elements of list 
                   in this attribute under the specific polarization are the 
                   same as the number of frequency channels.

        wts_change [boolean] indicates if weights and/or their lcoations have 
                   changed from the previous intergration or snapshot. 
                   Default=False means they have not changed. In such a case the 
                   interferometer-to-grid mapping and grid illumination pattern do not 
                   have to be determined, and mapping and values from the 
                   previous snapshot can be used. If True, a new mapping has to 
                   be determined.

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
        ------------------------------------------------------------------------
        """

        eps = 1.0e-10
        if pol is None:
            pol = ['P1', 'P2']
        elif not isinstance(pol, list):
            pol = [pol]

        if not self.grid_ready:
            self.grid()

        du = self.gridu[0,1] - self.gridu[0,0]
        dv = self.gridv[1,0] - self.gridv[0,0]
        wavelength = FCNST.c / self.f
        min_lambda = NP.abs(wavelength).min()
        rmaxNN = 0.5 * NP.sqrt(du**2 + dv**2) * min_lambda
 
        krn = {}
        crosspol = ['P11', 'P12', 'P21', 'P22']
        for cpol in crosspol:
            krn[cpol] = None
            if cpol in pol:

                bl_dict = self.baseline_vectors(pol=cpol, flag=None, sort=True)
                self.ordered_labels = bl_dict['labels']
                bl_xy = bl_dict['baselines'][:,:2] # n_bl x 2
                n_bl = bl_xy.shape[0]

                Vf_dict = self.get_visibilities(cpol, flag=None, tselect=-1, fselect=None, bselect=None, datapool='avg', sort=True)
                Vf = Vf_dict['visibilities'].astype(NP.complex64)  #  (n_ts=1) x n_bl x nchan
                Vf = NP.squeeze(Vf, axis=0)  # n_bl x nchan
                if Vf.shape[0] != n_bl:
                    raise ValueError('Encountered unexpected behavior. Need to debug.')
                bl_labels = Vf_dict['labels']
                twts = Vf_dict['twts']  # (n_ts=1) x n_bl x (nchan=1)
                twts = NP.squeeze(twts, axis=(0,2))  # n_bl

                if verbose:
                    print 'Gathered interferometer data for gridding convolution for timestamp {0}'.format(self.timestamp)

                if wts_change or (not self.grid_mapper[cpol]['all_bl2grid']):
                    self.grid_mapper[cpol]['per_bl2grid'] = []
                    self.grid_mapper[cpol]['all_bl2grid'] = {}
                    gridlocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                    if gridfunc_freq == 'scale':
                        grid_xy = gridlocs[NP.newaxis,:,:] * wavelength.reshape(-1,1,1)   # nchan x nv x nu
                        wl = NP.ones(gridlocs.shape[0])[NP.newaxis,:] * wavelength.reshape(-1,1)
                        grid_xy = grid_xy.reshape(-1,2)
                        wl = wl.reshape(-1)
                        indNN_list, blind, fvu_gridind = LKP.find_NN(bl_xy, grid_xy, distance_ULIM=2.0*distNN, flatten=True, parallel=False)
                        dxy = grid_xy[fvu_gridind,:] - bl_xy[blind,:]
                        fvu_gridind_unraveled = NP.unravel_index(fvu_gridind, (self.f.size,)+self.gridu.shape)   # f-v-u order since temporary grid was created as nchan x nv x nu
                        self.grid_mapper[cpol]['all_bl2grid']['blind'] = NP.copy(blind)
                        self.grid_mapper[cpol]['all_bl2grid']['u_gridind'] = NP.copy(fvu_gridind_unraveled[2])
                        self.grid_mapper[cpol]['all_bl2grid']['v_gridind'] = NP.copy(fvu_gridind_unraveled[1])                            
                        self.grid_mapper[cpol]['all_bl2grid']['f_gridind'] = NP.copy(fvu_gridind_unraveled[0])
                        self.grid_mapper[cpol]['all_bl2grid']['indNN_list'] = copy.deepcopy(indNN_list)
                        self.grid_mapper[cpol]['all_bl2grid']['twts'] = copy.deepcopy(twts)

                        if identical_interferometers:
                            arbitrary_interferometer_aperture = self.interferometers.itervalues().next().aperture
                            krn = arbitrary_interferometer_aperture.compute(dxy, wavelength=wl[fvu_gridind], pol=cpol, rmaxNN=rmaxNN, load_lookup=False)
                        else:
                            # This block #1 is one way to go about per interferometer
                            for bi,gi in enumerate(indNN_list):
                                if len(gi) > 0:
                                    label = self.ordered_labels[bi]
                                    ind = NP.asarray(gi)
                                    diffxy = grid_xy[ind,:].reshape(-1,2) - bl_xy[bi,:].reshape(-1,2)
                                    krndict = self.interferometers[label].aperture.compute(diffxy, wavelength=wl[ind], pol=cpol, rmaxNN=rmaxNN, load_lookup=False)
                                    if krn[cpol] is None:
                                        krn[cpol] = NP.copy(krndict[cpol])
                                    else:
                                        krn[cpol] = NP.append(krn[cpol], krndict[cpol])
                                    
                            # # This block #2 is another way equivalent to above block #1
                            # uniq_blind = NP.unique(blind)
                            # blhist, blbe, blbn, blri = OPS.binned_statistic(blind, statistic='count', bins=NP.append(uniq_blind, uniq_blind.max()+1))
                            # for i,ublind in enumerate(uniq_blind):
                            #     label = self.ordered_labels[ublind]
                            #     ind = blri[blri[i]:blri[i+1]]
                            #     krndict = self.interferometers[label].aperture.compute(dxy[ind,:], wavelength=wl[ind], pol=cpol, rmaxNN=rmaxNN, load_lookup=False)
                            #     if krn[cpol] is None:
                            #         krn[cpol] = NP.copy(krndict[cpol])
                            #     else:
                            #         krn[cpol] = NP.append(krn[cpol], krndict[cpol])

                        self.grid_mapper[cpol]['all_bl2grid']['illumination'] = NP.copy(krn[cpol])
                    else: # Weights do not scale with frequency (needs serious development)
                        pass
                        
                    # Determine weights that can normalize sum of kernel per interferometer per frequency to unity
                    # per_bl_per_freq_norm_wts = NP.ones(blind.size, dtype=NP.complex64)
                    per_bl_per_freq_norm_wts = NP.zeros(blind.size, dtype=NP.complex64)
                    
                    runsum = 0
                    for bi,gi in enumerate(indNN_list):
                        if len(gi) > 0:
                            fvu_ind = NP.asarray(gi)
                            unraveled_fvu_ind = NP.unravel_index(fvu_ind, (self.f.size,)+self.gridu.shape)
                            f_ind = unraveled_fvu_ind[0]
                            v_ind = unraveled_fvu_ind[1]
                            u_ind = unraveled_fvu_ind[2]
                            chanhist, chanbe, chanbn, chanri = OPS.binned_statistic(f_ind, statistic='count', bins=NP.arange(self.f.size+1))
                            for ci in xrange(self.f.size):
                                if chanhist[ci] > 0.0:
                                    select_chan_ind = chanri[chanri[ci]:chanri[ci+1]]
                                    per_bl_per_freq_kernel_sum = NP.sum(krn[cpol][runsum:runsum+len(gi)][select_chan_ind])
                                    per_bl_per_freq_norm_wts[runsum:runsum+len(gi)][select_chan_ind] = 1.0 / per_bl_per_freq_kernel_sum

                        per_bl2grid_info = {}
                        per_bl2grid_info['label'] = self.ordered_labels[bi]
                        per_bl2grid_info['twts'] = twts[bi]
                        per_bl2grid_info['f_gridind'] = NP.copy(f_ind)
                        per_bl2grid_info['u_gridind'] = NP.copy(u_ind)
                        per_bl2grid_info['v_gridind'] = NP.copy(v_ind)
                        # per_bl2grid_info['fvu_gridind'] = NP.copy(gi)
                        per_bl2grid_info['per_bl_per_freq_norm_wts'] = per_bl_per_freq_norm_wts[runsum:runsum+len(gi)]
                        per_bl2grid_info['illumination'] = krn[cpol][runsum:runsum+len(gi)]
                        self.grid_mapper[cpol]['per_bl2grid'] += [copy.deepcopy(per_bl2grid_info)]
                        runsum += len(gi)

                    self.grid_mapper[cpol]['all_bl2grid']['per_bl_per_freq_norm_wts'] = NP.copy(per_bl_per_freq_norm_wts)

                # Determine the gridded electric fields
                Vf_on_grid = Vf[(self.grid_mapper[cpol]['all_bl2grid']['blind'], self.grid_mapper[cpol]['all_bl2grid']['f_gridind'])]
                self.grid_mapper[cpol]['all_bl2grid']['Vf'] = copy.deepcopy(Vf_on_grid)
                runsum = 0
                for bi,gi in enumerate(self.grid_mapper[cpol]['all_bl2grid']['indNN_list']):
                    if len(gi) > 0:
                        self.grid_mapper[cpol]['per_bl2grid'][bi]['Vf'] = Vf_on_grid[runsum:runsum+len(gi)]
                        runsum += len(gi)

    ############################################################################

    def genMappingMatrix(self, pol=None, normalize=True, method='NN',
                         distNN=NP.inf, identical_interferometers=True,
                         gridfunc_freq=None, wts_change=False, parallel=False,
                         nproc=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Routine to construct sparse interferometer-to-grid mapping matrix that 
        will be used in projecting illumination and visibilities from the 
        array of interferometers onto the grid. It has elements very common to 
        grid_convolve_new()

        Inputs:

        pol        [String] The polarization to be gridded. Can be set to 'P11', 
                   'P12', 'P21', or 'P2'. If set to None, gridding for all the 
                   polarizations is performed. Default = None

        normalize  [Boolean] Default = False. If set to True, the gridded 
                   weights are divided by the sum of weights so that the gridded 
                   weights add up to unity. (Need to work on normaliation)

        method     [string] The gridding method to be used in applying the 
                   interferometer weights on to the interferometer array grid. 
                   Accepted values are 'NN' (nearest neighbour - default), 'CS' 
                   (cubic spline), or 'BL' (Bi-linear). In case of applying grid 
                   weights by 'NN' method, an optional distance upper bound for 
                   the nearest neighbour can be provided in the parameter distNN 
                   to prune the search and make it efficient. Currently, only 
                   the nearest neighbour method is operational.

        distNN     [scalar] A positive value indicating the upper bound on 
                   distance to the nearest neighbour in the gridding process. It 
                   has units of distance, the same units as the interferometer 
                   attribute location and interferometer array attribute gridx 
                   and gridy. Default is NP.inf (infinite distance). It will be 
                   internally converted to have same units as interferometer 
                   attributes wtspos (units in number of wavelengths). To ensure
                   all relevant pixels in the grid, the search distance used 
                   internally will be a fraction more than distNN

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
                   in this attribute under the specific polarization are the 
                   same as the number of frequency channels.

        wts_change [boolean] indicates if weights and/or their lcoations have 
                   changed from the previous intergration or snapshot. 
                   Default=False means they have not changed. In such a case the 
                   interferometer-to-grid mapping and grid illumination pattern 
                   do not have to be determined, and mapping and values from the 
                   previous snapshot can be used. If True, a new mapping has to 
                   be determined.

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
        verbose    [boolean] If True, prints diagnostic and progress messages. 
                   If False (default), suppress printing such messages.

        NOTE: Although certain portions are parallelizable, the overheads in 
        these processes seem to make it worse than serial processing. It is 
        advisable to stick to serialized version unless testing with larger
        data sets clearly indicates otherwise.
        ------------------------------------------------------------------------
        """

        if pol is None:
            pol = ['P1', 'P2']
        elif not isinstance(pol, list):
            pol = [pol]

        if not self.grid_ready:
            self.grid()

        du = self.gridu[0,1] - self.gridu[0,0]
        dv = self.gridv[1,0] - self.gridv[0,0]
        wavelength = FCNST.c / self.f
        min_lambda = NP.abs(wavelength).min()
        rmaxNN = 0.5 * NP.sqrt(du**2 + dv**2) * min_lambda
 
        krn = {}
        self.bl2grid_mapper = {}
        crosspol = ['P11', 'P12', 'P21', 'P22']
        for cpol in crosspol:
            krn[cpol] = None
            self.bl2grid_mapper[cpol] = None
            if cpol in pol:
                bl_dict = self.baseline_vectors(pol=cpol, flag=None, sort=True)
                self.ordered_labels = bl_dict['labels']
                bl_xy = bl_dict['baselines'][:,:2] # n_bl x 2
                n_bl = bl_xy.shape[0]

                if verbose:
                    print 'Gathered interferometer data for gridding convolution for timestamp {0}'.format(self.timestamp)

                if wts_change or (not self.grid_mapper[cpol]['all_bl2grid']):
                    self.grid_mapper[cpol]['per_bl2grid'] = []
                    self.grid_mapper[cpol]['all_bl2grid'] = {}
                    gridlocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                    if gridfunc_freq == 'scale':
                        grid_xy = gridlocs[NP.newaxis,:,:] * wavelength.reshape(-1,1,1)   # nchan x nv x nu
                        wl = NP.ones(gridlocs.shape[0])[NP.newaxis,:] * wavelength.reshape(-1,1)
                        grid_xy = grid_xy.reshape(-1,2)
                        wl = wl.reshape(-1)
                        indNN_list, blind, fvu_gridind = LKP.find_NN(bl_xy, grid_xy, distance_ULIM=2.0*distNN, flatten=True, parallel=False)
                        dxy = grid_xy[fvu_gridind,:] - bl_xy[blind,:]
                        fvu_gridind_unraveled = NP.unravel_index(fvu_gridind, (self.f.size,)+self.gridu.shape)   # f-v-u order since temporary grid was created as nchan x nv x nu
                        self.grid_mapper[cpol]['all_bl2grid']['blind'] = NP.copy(blind)
                        self.grid_mapper[cpol]['all_bl2grid']['u_gridind'] = NP.copy(fvu_gridind_unraveled[2])
                        self.grid_mapper[cpol]['all_bl2grid']['v_gridind'] = NP.copy(fvu_gridind_unraveled[1])                            
                        self.grid_mapper[cpol]['all_bl2grid']['f_gridind'] = NP.copy(fvu_gridind_unraveled[0])
                        # self.grid_mapper[cpol]['all_bl2grid']['indNN_list'] = copy.deepcopy(indNN_list)

                        if identical_interferometers:
                            arbitrary_interferometer_aperture = self.interferometers.itervalues().next().aperture
                            krn = arbitrary_interferometer_aperture.compute(dxy, wavelength=wl[fvu_gridind], pol=cpol, rmaxNN=rmaxNN, load_lookup=False)
                        else:
                            # This block #1 is one way to go about per interferometer
                            for ai,gi in enumerate(indNN_list):
                                if len(gi) > 0:
                                    label = self.ordered_labels[ai]
                                    ind = NP.asarray(gi)
                                    diffxy = grid_xy[ind,:].reshape(-1,2) - bl_xy[ai,:].reshape(-1,2)
                                    krndict = self.interferometers[label].aperture.compute(diffxy, wavelength=wl[ind], pol=cpol, rmaxNN=rmaxNN, load_lookup=False)
                                    if krn[cpol] is None:
                                        krn[cpol] = NP.copy(krndict[cpol])
                                    else:
                                        krn[cpol] = NP.append(krn[cpol], krndict[cpol])
                                    
                            # # This block #2 is another way equivalent to above block #1
                            # uniq_blind = NP.unique(blind)
                            # blhist, blbe, blbn, blri = OPS.binned_statistic(blind, statistic='count', bins=NP.append(uniq_blind, uniq_blind.max()+1))
                            # for i,ublind in enumerate(uniq_blind):
                            #     label = self.ordered_labels[ublind]
                            #     ind = blri[blri[i]:blri[i+1]]
                            #     krndict = self.interferometers[label].aperture.compute(dxy[ind,:], wavelength=wl[ind], pol=cpol, rmaxNN=rmaxNN, load_lookup=False)
                            #     if krn[cpol] is None:
                            #         krn[cpol] = NP.copy(krndict[cpol])
                            #     else:
                            #         krn[cpol] = NP.append(krn[cpol], krndict[cpol])

                        self.grid_mapper[cpol]['all_bl2grid']['illumination'] = NP.copy(krn[cpol])
                    else: # Weights do not scale with frequency (needs serious development)
                        pass
                        
                    # Determine weights that can normalize sum of kernel per interferometer per frequency to unity
                    per_bl_per_freq_norm_wts = NP.zeros(blind.size, dtype=NP.complex64)
                    # per_bl_per_freq_norm_wts = NP.ones(blind.size, dtype=NP.complex64)                    
                    
                    if parallel or (nproc is not None):
                        list_of_val = []
                        list_of_rowcol_tuple = []
                    else:
                        spval = []
                        sprow = []
                        spcol = []
                        
                    runsum = 0
                    if verbose:
                        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Baselines '.format(n_bl), PGB.ETA()], maxval=n_bl).start()

                    for bi,gi in enumerate(indNN_list):
                        if len(gi) > 0:
                            fvu_ind = NP.asarray(gi)
                            unraveled_fvu_ind = NP.unravel_index(fvu_ind, (self.f.size,)+self.gridu.shape)
                            f_ind = unraveled_fvu_ind[0]
                            v_ind = unraveled_fvu_ind[1]
                            u_ind = unraveled_fvu_ind[2]
                            chanhist, chanbe, chanbn, chanri = OPS.binned_statistic(f_ind, statistic='count', bins=NP.arange(self.f.size+1))
                            for ci in xrange(self.f.size):
                                if chanhist[ci] > 0.0:
                                    select_chan_ind = chanri[chanri[ci]:chanri[ci+1]]
                                    per_bl_per_freq_kernel_sum = NP.sum(krn[cpol][runsum:runsum+len(gi)][select_chan_ind])
                                    per_bl_per_freq_norm_wts[runsum:runsum+len(gi)][select_chan_ind] = 1.0 / per_bl_per_freq_kernel_sum

                        per_bl2grid_info = {}
                        per_bl2grid_info['label'] = self.ordered_labels[bi]
                        per_bl2grid_info['f_gridind'] = NP.copy(f_ind)
                        per_bl2grid_info['u_gridind'] = NP.copy(u_ind)
                        per_bl2grid_info['v_gridind'] = NP.copy(v_ind)
                        # per_bl2grid_info['fvu_gridind'] = NP.copy(gi)
                        per_bl2grid_info['per_bl_per_freq_norm_wts'] = per_bl_per_freq_norm_wts[runsum:runsum+len(gi)]
                        per_bl2grid_info['illumination'] = krn[cpol][runsum:runsum+len(gi)]
                        self.grid_mapper[cpol]['per_bl2grid'] += [copy.deepcopy(per_bl2grid_info)]
                        runsum += len(gi)

                        # determine the sparse interferometer-to-grid mapping matrix pre-requisites

                        val = per_bl2grid_info['per_bl_per_freq_norm_wts']*per_bl2grid_info['illumination']
                        vuf_gridind_unraveled = (per_bl2grid_info['v_gridind'],per_bl2grid_info['u_gridind'],per_bl2grid_info['f_gridind'])
                        vuf_gridind_raveled = NP.ravel_multi_index(vuf_gridind_unraveled, (self.gridu.shape+(self.f.size,)))
                        
                        if (not parallel) and (nproc is None):
                            spval += val.tolist()
                            sprow += vuf_gridind_raveled.tolist()
                            spcol += (per_bl2grid_info['f_gridind'] + bi*self.f.size).tolist()
                        else:
                            list_of_val += [per_bl2grid_info['per_bl_per_freq_norm_wts']*per_bl2grid_info['illumination']]
                            list_of_rowcol_tuple += [(vuf_gridind_raveled, per_bl2grid_info['f_gridind'])]
                    
                        if verbose:
                            progress.update(bi+1)

                    if verbose:
                        progress.finish()

                    # determine the sparse interferometer-to-grid mapping matrix
                    if parallel or (nproc is not None):
                        list_of_shapes = [(self.gridu.size*self.f.size, self.f.size)] * n_bl
                        if nproc is None:
                            nproc = max(MP.cpu_count()-1, 1) 
                        else:
                            nproc = min(nproc, max(MP.cpu_count()-1, 1))
                        pool = MP.Pool(processes=nproc)
                        list_of_spmat = pool.map(genMatrixMapper_arg_splitter, IT.izip(list_of_val, list_of_rowcol_tuple, list_of_shapes))
                        self.bl2grid_mapper[cpol] = SpM.hstack(list_of_spmat, format='csr')
                    else:
                        spval = NP.asarray(spval)
                        sprowcol = (NP.asarray(sprow), NP.asarray(spcol))
                        self.bl2grid_mapper[cpol] = SpM.csr_matrix((spval, sprowcol), shape=(self.gridu.size*self.f.size, n_bl*self.f.size))
                    
                    self.grid_mapper[cpol]['all_bl2grid']['per_bl_per_freq_norm_wts'] = NP.copy(per_bl_per_freq_norm_wts)

    ############################################################################

    def applyMappingMatrix(self, pol=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Constructs the grid of complex illumination and visibilities
        using the sparse baseline-to-grid mapping matrix. Intended to serve as a 
        "matrix" alternative to make_grid_cube_new() 

        Inputs:

        pol     [String] The polarization to be gridded. Can be set to 'P11', 
                'P12', 'P21', or 'P22'. If set to None, gridding for all the 
                polarizations is performed. Default=None
        
        verbose [boolean] If True, prints diagnostic and progress messages. 
                If False (default), suppress printing such messages.
        ------------------------------------------------------------------------
        """
        
        if pol is None:
            pol = ['P11', 'P12', 'P21', 'P22']

        pol = NP.unique(NP.asarray(pol))
        
        for cpol in pol:

            if verbose:
                print 'Gridding aperture illumination and visibilities for polarization {0} ...'.format(cpol)

            if cpol not in ['P11', 'P12', 'P21', 'P22']:
                raise ValueError('Invalid specification for input parameter pol')

            Vf_dict = self.get_visibilities(cpol, flag=None, tselect=-1, fselect=None, bselect=None, datapool='avg', sort=True)
            Vf = Vf_dict['visibilities'].astype(NP.complex64)  #  (n_ts=1) x n_bl x nchan
            Vf = NP.squeeze(Vf, axis=0)  # n_bl x nchan

            twts = Vf_dict['twts']  # (n_ts=1) x n_ant x 1
            twts = NP.squeeze(twts, axis=0)  # n_ant x 1
            unflagged = twts > 0.0
            unflagged = unflagged.astype(int)

            Vf = Vf * unflagged    # applies antenna flagging, n_ant x nchan
            wts = unflagged * NP.ones(self.f.size).reshape(1,-1)  # n_ant x nchan

            wts[NP.isnan(Vf)] = 0.0
            Vf[NP.isnan(Vf)] = 0.0

            Vf = Vf.ravel()
            wts = wts.ravel()

            sparse_Vf = SpM.csr_matrix(Vf)
            sparse_wts = SpM.csr_matrix(wts)

            # Store as sparse matrices
            self.grid_illumination[cpol] = self.bl2grid_mapper[cpol].dot(sparse_wts.T)
            self.grid_Vf[cpol] = self.bl2grid_mapper[cpol].dot(sparse_Vf.T)

            # # Store as dense matrices
            # self.grid_illumination[cpol] = self.bl2grid_mapper[cpol].dot(wts).reshape(self.gridu.shape+(self.f.size,))
            # self.grid_Vf[cpol] = self.bl2grid_mapper[cpol].dot(Vf).reshape(self.gridu.shape+(self.f.size,))   
            
            if verbose:
                print 'Gridded aperture illumination and electric fields for polarization {0} from {1:0d} unflagged contributing antennas'.format(cpol, NP.sum(unflagged).astype(int))

    ############################################################################

    def make_grid_cube(self, pol=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Constructs the grid of complex power illumination and visibilities using 
        the gridding information determined for every baseline. Flags are taken
        into account while constructing this grid.

        Inputs:

        pol     [String] The polarization to be gridded. Can be set to 'P11', 
                'P12', 'P21' or 'P22'. If set to None, gridding for all the
                polarizations is performed. Default = None
        
        verbose [boolean] If True, prints diagnostic and progress messages. 
                If False (default), suppress printing such messages.
        ------------------------------------------------------------------------
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
            sum_twts = 0.0
            for bllabel, blinfo in self.grid_mapper[cpol]['labels'].iteritems():
                # if not self.interferometers[bllabel].crosspol.flag[cpol]:
                if blinfo['twts'] > 0.0:
                    num_unflagged += 1
                    sum_twts += blinfo['twts']
                    gridind_unraveled = NP.unravel_index(blinfo['gridind'], self.gridu.shape+(self.f.size,))
                    # self.grid_illumination[cpol][gridind_unraveled] += blinfo['illumination'] * blinfo['twts']
                    # self.grid_Vf[cpol][gridind_unraveled] += blinfo['Vf'] * blinfo['twts']
                    self.grid_illumination[cpol][gridind_unraveled] += blinfo['illumination']                    
                    self.grid_Vf[cpol][gridind_unraveled] += blinfo['Vf']

                progress.update(loopcount+1)
                loopcount += 1
            progress.finish()

            # self.grid_Vf[cpol] *= num_unflagged/sum_twts
                
            if verbose:
                print 'Gridded aperture illumination and visibilities for polarization {0} from {1:0d} unflagged contributing baselines'.format(cpol, num_unflagged)

    ############################################################################

    def make_grid_cube_new(self, pol=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Constructs the grid of complex power illumination and visibilities using 
        the gridding information determined for every baseline. Flags are taken
        into account while constructing this grid.

        Inputs:

        pol     [String] The polarization to be gridded. Can be set to 'P11', 
                'P12', 'P21' or 'P22'. If set to None, gridding for all the
                polarizations is performed. Default = None
        
        verbose [boolean] If True, prints diagnostic and progress messages. 
                If False (default), suppress printing such messages.
        ------------------------------------------------------------------------
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
    
            nlabels = len(self.grid_mapper[cpol]['per_bl2grid'])
            loopcount = 0
            num_unflagged = 0
            sum_twts = 0.0
            if verbose:
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(nlabels), PGB.ETA()], maxval=nlabels).start()

            for bi, per_bl2grid_info in enumerate(self.grid_mapper[cpol]['per_bl2grid']):
                bllabel = per_bl2grid_info['label']
                if per_bl2grid_info['twts'] > 0.0:
                    num_unflagged += 1
                    sum_twts += per_bl2grid_info['twts']
                    vuf_gridind_unraveled = (per_bl2grid_info['v_gridind'],per_bl2grid_info['u_gridind'],per_bl2grid_info['f_gridind'])
                    self.grid_illumination[cpol][vuf_gridind_unraveled] += per_bl2grid_info['per_bl_per_freq_norm_wts'] * per_bl2grid_info['illumination']
                    self.grid_Vf[cpol][vuf_gridind_unraveled] += per_bl2grid_info['per_bl_per_freq_norm_wts'] * per_bl2grid_info['Vf'] * per_bl2grid_info['illumination']
                    # self.grid_illumination[cpol][vuf_gridind_unraveled] += per_bl2grid_info['per_bl_per_freq_norm_wts'] * per_bl2grid_info['illumination'] * per_bl2grid_info['twts']
                    # self.grid_Vf[cpol][vuf_gridind_unraveled] += per_bl2grid_info['per_bl_per_freq_norm_wts'] * per_bl2grid_info['Vf'] * per_bl2grid_info['twts']

                if verbose:
                    progress.update(loopcount+1)
                    loopcount += 1
            if verbose:
                progress.finish()
            # self.grid_illumination[cpol] *= num_unflagged/sum_twts
            # self.grid_Vf[cpol] *= num_unflagged/sum_twts

            if verbose:
                print 'Gridded aperture illumination and visibilities for polarization {0} from {1:0d} unflagged contributing baselines'.format(cpol, num_unflagged)

    ############################################################################

    def quick_beam_synthesis(self, pol=None):
        
        """
        ------------------------------------------------------------------------
        A quick generator of synthesized beam using interferometer array grid 
        illumination pattern using the center frequency. Not intended to be used
        rigorously but rather for comparison purposes and making quick plots

        Inputs:

        pol     [String] The polarization of the synthesized beam. Can be set 
                to 'P11', 'P12', 'P21' or 'P2'. If set to None, synthesized beam 
                for all the polarizations are generated. Default=None

        Outputs:

        Dictionary with the following keys and information:

        'syn_beam'  [numpy array] synthesized beam of same size as that of the 
                    interferometer array grid. It is FFT-shifted to place the 
                    origin at the center of the array. The peak value of the 
                    synthesized beam is fixed at unity

        'grid_power_illumination'
                    [numpy array] complex grid illumination obtained from 
                    inverse fourier transform of the synthesized beam in 
                    'syn_beam' and has size same as that of the interferometer 
                    array grid. It is FFT-shifted to have the origin at the 
                    center. The sum of this array is set to unity to match the 
                    peak of the synthesized beam

        'l'         [numpy vector] x-values of the direction cosine grid 
                    corresponding to x-axis (axis=1) of the synthesized beam

        'm'         [numpy vector] y-values of the direction cosine grid 
                    corresponding to y-axis (axis=0) of the synthesized beam
        ------------------------------------------------------------------------
        """

        if not self.grid_ready:
            raise ValueError('Need to perform gridding of the antenna array before an equivalent UV grid can be simulated')

        if pol is None:
            pol = ['P11', 'P12', 'P21', 'P22']
        elif isinstance(pol, str):
            if pol in ['P11', 'P12', 'P21', 'P22']:
                pol = [pol]
            else:
                raise ValueError('Invalid polarization specified')
        elif isinstance(pol, list):
            p = [cpol for cpol in pol if cpol in ['P11', 'P12', 'P21', 'P22']]
            if len(p) == 0:
                raise ValueError('Invalid polarization specified')
            pol = p
        else:
            raise TypeError('Input keyword pol must be string, list or set to None')

        pol = sorted(pol)

        for cpol in pol:
            if self.grid_illumination[cpol] is None:
                raise ValueError('Grid illumination for the specified polarization is not determined yet. Must use make_grid_cube()')

        chan = NP.argmin(NP.abs(self.f - self.f0))
        orig_syn_beam_in_uv = NP.empty(self.gridu.shape+(len(pol),), dtype=NP.complex)
        for pind, cpol in enumerate(pol):
            orig_syn_beam_in_uv[:,:,pind] = self.grid_illumination[cpol][:,:,chan]

        # # Pad it with zeros to be twice the size
        # padded_syn_beam_in_uv = NP.pad(orig_syn_beam_in_uv, ((0,orig_syn_beam_in_uv.shape[0]),(0,orig_syn_beam_in_uv.shape[1]),(0,0)), mode='constant', constant_values=0)
        # # The NP.roll statements emulate a fftshift but by 1/4 of the size of the padded array
        # padded_syn_beam_in_uv = NP.roll(padded_syn_beam_in_uv, -orig_syn_beam_in_uv.shape[0]/2, axis=0)
        # padded_syn_beam_in_uv = NP.roll(padded_syn_beam_in_uv, -orig_syn_beam_in_uv.shape[1]/2, axis=1)

        # Pad it with zeros on either side to be twice the size
        padded_syn_beam_in_uv = NP.pad(orig_syn_beam_in_uv, ((orig_syn_beam_in_uv.shape[0]/2,orig_syn_beam_in_uv.shape[0]/2),(orig_syn_beam_in_uv.shape[1]/2,orig_syn_beam_in_uv.shape[1]/2),(0,0)), mode='constant', constant_values=0)

        # Shift to be centered
        padded_syn_beam_in_uv = NP.fft.ifftshift(padded_syn_beam_in_uv)

        # Compute the synthesized beam. It is at a finer resolution due to padding
        syn_beam = NP.fft.fft2(padded_syn_beam_in_uv, axes=(0,1))

        # Select only the real part, equivalent to adding conjugate baselines
        syn_beam = 2 * syn_beam.real
        syn_beam /= syn_beam.max()

        # Inverse Fourier Transform to obtain real and symmetric uv-grid illumination
        syn_beam_in_uv = NP.fft.ifft2(syn_beam, axes=(0,1))

        # shift the array to be centered
        syn_beam_in_uv = NP.fft.ifftshift(syn_beam_in_uv, axes=(0,1))

        # Discard pads at either end and select only the central values of original size
        syn_beam_in_uv = syn_beam_in_uv[orig_syn_beam_in_uv.shape[0]/2:orig_syn_beam_in_uv.shape[0]/2+orig_syn_beam_in_uv.shape[0],orig_syn_beam_in_uv.shape[1]/2:orig_syn_beam_in_uv.shape[1]/2+orig_syn_beam_in_uv.shape[1],:]
        syn_beam = NP.fft.fftshift(syn_beam[::2,::2,:], axes=(0,1))  # Downsample by factor 2 to get native resolution and shift to be centered
        
        du = self.gridu[0,1] - self.gridu[0,0]
        dv = self.gridv[1,0] - self.gridv[0,0]
        l = DSP.spectax(self.gridu.shape[1], resolution=du, shift=True)
        m = DSP.spectax(self.gridv.shape[0], resolution=dv, shift=True)        

        return {'syn_beam': syn_beam, 'grid_power_illumination': syn_beam_in_uv, 'l': l, 'm': m}

    ############################################################################ 

    def grid_convolve_old(self, pol=None, antpairs=None, unconvolve_existing=False,
                          normalize=False, method='NN', distNN=NP.inf, tol=None,
                          maxmatch=None): 

        """
        ------------------------------------------------------------------------
        Routine to project the visibility illumination pattern and the 
        visibilities on the grid. It can operate on the entire antenna array or
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
                   neighbour determination. tol is to be interpreted as a 
                   minimum value considered as significant in the lookup table. 
        ------------------------------------------------------------------------
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

    ############################################################################

    def grid_unconvolve(self, antpairs, pol=None):

        """
        ------------------------------------------------------------------------
        [Needs to be re-written]

        Routine to de-project the visibility illumination pattern and the
        visibilities on the grid. It can operate on the entire interferometer 
        array or incrementally de-project the visibilities and illumination 
        patterns of specific antenna pairs from an already existing grid.

        Inputs:

        antpairs    [instance of class InterferometerArray, single instance or 
                    list of instances of class Interferometer, or a dictionary 
                    holding instances of class Interferometer] If a dictionary 
                    is provided, the keys should be the interferometer labels 
                    and the values should be instances of class Interferometer. 
                    If a list is provided, it should be a list of valid 
                    instances of class Interferometer. These instance(s) of 
                    class Interferometer will be merged to the existing grid 
                    contained in the instance of InterferometerArray class. If 
                    any of the interferoemters are not found to be in the 
                    already existing set of interferometers, an exception is 
                    raised accordingly and code execution stops.

        pol         [String] The polarization to be gridded. Can be set to 
                    'P11', 'P12', 'P21', or 'P22'. If set to None, gridding for 
                    all polarizations is performed. Default=None.
        ------------------------------------------------------------------------
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

    ############################################################################

    def update_flags(self, dictflags=None, stack=True, verify=False):

        """
        ------------------------------------------------------------------------
        Updates all flags in the interferometer array followed by any flags that
        need overriding through inputs of specific flag information

        Inputs:

        dictflags  [dictionary] contains flag information overriding after 
                   default flag updates are determined. Baseline based flags 
                   are given as further dictionaries with each under under a 
                   key which is the same as the interferometer label. Flags for 
                   each baseline are specified as a dictionary holding boolean 
                   flags for each of the four cross-polarizations which are 
                   stored under keys 'P11', 'P12', 'P21', and 'P22'. An absent 
                   key just means it is not a part of the update. Flag 
                   information under each baseline must be of same type as 
                   input parameter flags in member function update_flags() of 
                   class CrossPolInfo

        stack      [boolean] If True (default), appends the updated flag to the
                   end of the stack of flags as a function of timestamp. If 
                   False, updates the last flag in the stack with the updated 
                   flag and does not append

        verify     [boolean] If True, verify and update the flags, if necessary.
                   Visibilities are checked for NaN values and if found, the
                   flag in the corresponding polarization is set to True. 
                   Default=False. 
        ------------------------------------------------------------------------
        """

        for label in self.interferometers:
            self.interferometers[label].update_flags(stack=stack, verify=verify)

        if dictflags is not None: # Performs flag overriding. Use stack=False
            if not isinstance(dictflags, dict):
                raise TypeError('Input parameter dictflags must be a dictionary')
            
            for label in dictflags:
                if label in self.interferometers:
                    self.interferometers[label].update_flags(flags=dictflags[label], stack=False, verify=True)

    ############################################################################

    def update(self, interferometer_level_updates=None,
               antenna_level_updates=None, do_correlate=None, parallel=False,
               nproc=None, verbose=False):

        """
        ------------------------------------------------------------------------
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
                                'do_grid'     [boolean] If set to True, create 
                                              or recreate a grid. To be 
                                              specified when the antenna 
                                              locations are updated.
                    'antennas': Holds a list of dictionaries consisting of 
                                updates for individual antennas. Each element 
                                in the list contains update for one antenna. 
                                For each of these dictionaries, one of the keys 
                                is 'label' which indicates an antenna label. If 
                                absent, the code execution stops by throwing an 
                                exception. The other optional keys and the 
                                information they hold are listed below:
                                'action'      [String scalar] Indicates the type 
                                              of update operation. 'add' adds 
                                              the Antenna instance to the 
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
                                              False, gridding effects will 
                                              remain unchanged. Default=None
                                              (=False).
                                'antenna'     [instance of class Antenna] 
                                              Updated Antenna class instance. 
                                              Can work for action key 'remove' 
                                              even if not set (=None) or set to 
                                              an empty string '' as long as 
                                              'label' key is specified. 
                                'gridpol'     [Optional. String scalar] 
                                              Initiates the specified action on 
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
                                'stack'       [boolean] If True (default), 
                                              appends the updated flag and data 
                                              to the end of the stack as a 
                                              function of timestamp. If False, 
                                              updates the last flag and data in 
                                              the stack and does not append
                                't'           [Optional. Numpy array] Time axis 
                                              of the time series. Is used only 
                                              if set and if 'action' key value 
                                              is set to 'modify'. Default=None.
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
                                'aperture'    [instance of class 
                                              APR.Aperture] aperture 
                                              information for the antenna. Read 
                                              docstring of class 
                                              Aperture for details
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
                                              the size of wtspos_P1 and 
                                              wtspos_P2. It is applicable only 
                                              when 'action' key is set to 
                                              'modify'. Default = None.
                                'delaydict'   [Dictionary] contains information 
                                              on delay compensation to be 
                                              applied to the fourier transformed 
                                              electric fields under each 
                                              polarization which are stored 
                                              under keys 'P1' and 'P2'. Default 
                                              is None (no delay compensation to 
                                              be applied). Refer to the 
                                              docstring of member function 
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
                                              If set to True, the gridded 
                                              weights are divided by the sum of 
                                              weights so that the gridded 
                                              weights add up to unity. This is 
                                              used only when grid_action keyword 
                                              is set when action keyword is set 
                                              to 'add' or 'modify'
                                'gridmethod'  [Optional. String] Indicates 
                                              gridding method. It accepts the 
                                              following values 'NN' (nearest 
                                              neighbour), 'BL' (Bi-linear
                                              interpolation), and'CS' (Cubic
                                              Spline interpolation). 
                                              Default='NN'
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
                    interferoemters and interferometer array as a whole under 
                    the following principal keys:
                    'interferometer_array': Consists of updates for the
                                InterferometerArray instance. This is a
                                dictionary which consists of the following keys:
                                'timestamp': Unique identifier of the time
                                       series. It is optional to set this to a
                                       scalar. If not given, no change is made 
                                       to the existing timestamp attribute
                    'interferometers': Holds a list of dictionaries where 
                                element consists of updates for individual 
                                interferometers. Each dictionary must contain a 
                                key 'label' which indicates an interferometer 
                                label. If absent, the code execution stops by 
                                throwing an exception. The other optional keys 
                                and the information they hold are listed below:
                                'action'      [String scalar] Indicates the type 
                                              of update operation. 'add' adds 
                                              the Interferometer instance to the 
                                              InterferometerArray instance. 
                                              'remove' removes the 
                                              interferometer from the 
                                              interferometer array instance. 
                                              'modify' modifies the 
                                              interferometer attributes in the 
                                              interferometer array instance. 
                                              This key has to be set. No default
                                'grid_action' [Boolean] If set to True, will 
                                              apply the grdding operations 
                                              (grid(), grid_convolve(), and 
                                              grid_unconvolve()) appropriately 
                                              according to the value of the 
                                              'action' key. If set to None or 
                                              False, gridding effects will 
                                              remain unchanged. Default=None
                                              (=False).
                                'interferometer' 
                                              [instance of class Interferometer] 
                                              Updated Interferometer class 
                                              instance. Can work for action key
                                              'remove' even if not set (=None) 
                                              or set to an empty string '' as 
                                              long as 'label' key is specified. 
                                'gridpol'     [Optional. String scalar] 
                                              Initiates the specified action on 
                                              polarization 'P11' or 'P22'. Can 
                                              be set to 'P11' or 'P22'. If not 
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
                                              if set and if 'action' key value 
                                              is set to 'modify'. Default=None.
                                'timestamp'   [Optional. Scalar] Unique 
                                              identifier of the time series. Is 
                                              used only if set and if 'action' 
                                              key value is set to 'modify'.
                                              Default = None.
                                'stack'       [boolean] If True (default), 
                                              appends the updated flag and data 
                                              to the end of the stack as a 
                                              function of timestamp. If False, 
                                              updates the last flag and data in 
                                              the stack and does not append
                                'location'    [Optional. instance of GEOM.Point
                                              class] Interferometer location in 
                                              the local ENU coordinate system. 
                                              Used only if set and if 'action' 
                                              key value is set to 'modify'.
                                              Default=None.
                                'aperture'    [instance of class 
                                              APR.Aperture] aperture 
                                              information for the 
                                              interferometer. Read docstring of 
                                              class Aperture for details
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
                                              If not set (=None), the previous 
                                              or default flag status will 
                                              continue to apply. If set to 
                                              False, the antenna status will be 
                                              updated to become unflagged.
                                'gridfunc_freq'
                                              [Optional. String scalar] Read the 
                                              description of inputs to 
                                              Interferometer class member 
                                              function update(). If set to None 
                                              (not provided), this attribute is 
                                              determined based on the size of 
                                              wtspos under each polarization. 
                                              It is applicable only when 
                                              'action' key is set to 'modify'. 
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
                                              If set to True, the gridded 
                                              weights are divided by the sum of 
                                              weights so that the gridded 
                                              weights add up to unity. This is 
                                              used only when grid_action keyword 
                                              is set when action keyword is set 
                                              to 'add' or 'modify'
                                'gridmethod'  [Optional. String] Indicates 
                                              gridding method. It accepts the 
                                              following values 'NN' (nearest 
                                              neighbour), 'BL' (Bi-linear
                                              interpolation), and'CS' (Cubic
                                              Spline interpolation). 
                                              Default='NN'
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
                                              per interferometer, use maxmatch=1 
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

        parallel    [boolean] specifies if parallelization is to be invoked. 
                    False (default) means only serial processing

        nproc       [integer] specifies number of independent processes to 
                    spawn. Default = None, means automatically determines the 
                    number of process cores in the system and use one less than 
                    that to avoid locking the system for other processes. 
                    Applies only if input parameter 'parallel' (see above) is 
                    set to True. If nproc is set to a value more than the number 
                    of process cores in the system, it will be reset to number 
                    of process cores in the system minus one to avoid locking 
                    the system out for other processes

        verbose     [Boolean] Default = False. If set to True, prints some 
                    diagnotic or progress messages.
        ------------------------------------------------------------------------
        """

        if antenna_level_updates is not None:
            if verbose:
                print 'Updating antenna array...'
            self.antenna_array.update(updates=antenna_level_updates)
            if verbose:
                print 'Updated antenna array. Refreshing interferometer flags from antenna flags...'
            self.update_flags(dictflags=None, stack=False, verify=False)  # Update interferometer flags using antenna level flags
            if verbose:
                print 'Refreshed interferometer flags. Refreshing antenna pairs...'
            self.refresh_antenna_pairs()
            if verbose:
                print 'Refreshed antenna pairs...'

        if verbose:
            print 'Updating interferometer array ...'

        self.timestamp = self.antenna_array.timestamp
        self.t = self.antenna_array.t

        if interferometer_level_updates is not None:
            if not isinstance(interferometer_level_updates, dict):
                raise TypeError('Input parameter interferometer_level_updates must be a dictionary')

            if 'interferometers' in interferometer_level_updates:
                if not isinstance(interferometer_level_updates['interferometers'], list):
                    interferometer_level_updates['interferometers'] = [interferometer_level_updates['interferometers']]
                if parallel:
                    list_of_interferometer_updates = []
                    list_of_interferometers = []

                if verbose:
                    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Interferometers '.format(len(interferometer_level_updates['interferometers'])), PGB.ETA()], maxval=len(interferometer_level_updates['interferometers'])).start()
                loopcount = 0
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
                            if 'stack' not in dictitem: dictitem['stack']=False
                            if 'gridfunc_freq' not in dictitem: dictitem['gridfunc_freq']=None
                            if 'ref_freq' not in dictitem: dictitem['ref_freq']=None
                            if 'pol_type' not in dictitem: dictitem['pol_type']=None
                            if 'norm_wts' not in dictitem: dictitem['norm_wts']=False
                            if 'gridmethod' not in dictitem: dictitem['gridmethod']='NN'
                            if 'distNN' not in dictitem: dictitem['distNN']=NP.inf
                            if 'maxmatch' not in dictitem: dictitem['maxmatch']=None
                            if 'tol' not in dictitem: dictitem['tol']=None
                            if 'do_correlate' not in dictitem: dictitem['do_correlate']=None
                            if 'aperture' not in dictitem: dictitem['aperture']=None

                            if not parallel:
                                # self.interferometers[dictitem['label']].update_old(dictitem['label'], dictitem['Vt'], dictitem['t'], dictitem['timestamp'], dictitem['location'], dictitem['wtsinfo'], dictitem['flags'], dictitem['gridfunc_freq'], dictitem['ref_freq'], dictitem['do_correlate'], verbose)
                                self.interferometers[dictitem['label']].update(dictitem, verbose)                                
                            else:
                                list_of_interferometers += [self.interferometers[dictitem['label']]]
                                list_of_interferometer_updates += [dictitem]

                            if 'gric_action' in dictitem:
                                self.grid_convolve(pol=dictitem['gridpol'], antpairs=dictitem['interferometer'], unconvolve_existing=True, normalize=dictitem['norm_wts'], method=dictitem['gridmethod'], distNN=dictitem['distNN'], tol=dictitem['tol'], maxmatch=dictitem['maxmatch'])
                    else:
                        raise ValueError('Update action should be set to "add", "remove" or "modify".')

                    if verbose:
                        progress.update(loopcount+1)
                        loopcount += 1
                if verbose:
                    progress.finish()
                    
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

################################################################################
        
class NewImage:

    """
    ----------------------------------------------------------------------------
    Class to manage image information and processing pertaining to the class 
    holding antenna array or interferometer array information.

    [Docstring is outdated. Needs to be updated definitely]

    Attributes:

    timestamp:   [Scalar] String or float representing the timestamp for the 
                 current attributes
                 
    f:           [vector] Frequency channels (in Hz)
                 
    f0:          [Scalar] Positive value for the center frequency in Hz.

    autocorr_wts_vuf
                 [dictionary] dictionary with polarization keys 'P1' and 'P2. 
                 Under each key is a matrix of size nt x nv x nu x nchan

    autocorr_data_vuf 
                 [dictionary] dictionary with polarization keys 'P1' and 'P2. 
                 Under each key is a matrix of size nt x nv x nu x nchan 
                 where nt=1, nt=n_timestamps, or nt=n_tavg if datapool is set 
                 to 'current', 'stack' or 'avg' respectively

    gridx_P1     [Numpy array] x-locations of the grid lattice for P1 
                 polarization

    gridy_P1     [Numpy array] y-locations of the grid lattice for P1 
                 polarization

    gridx_P2     [Numpy array] x-locations of the grid lattice for P2 
                 polarization

    gridy_P2     [Numpy array] y-locations of the grid lattice for P2 
                 polarization

    grid_illumination_P1
                 [Numpy array] Electric field illumination for P1 polarization 
                 on the grid. Could be complex. Same size as the grid

    grid_illumination_P2
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

    holograph_P1 [Numpy array] Complex holographic image cube for polarization 
                 P1 obtained by inverse fourier transforming Ef_P1

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

    holograph_P2 [Numpy array] Complex holographic image cube for polarization 
                 P2 obtained by inverse fourier transforming Ef_P2

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

    __init__()   Initializes an instance of class Image which manages 
                 information and processing of images from data obtained by an 
                 antenna array. It can be initialized either by values in an 
                 instance of class AntennaArray, by values in a fits file 
                 containing information about the antenna array, or to defaults.

    imagr()      Imaging engine that performs inverse fourier transforms of 
                 appropriate electric field quantities associated with the 
                 antenna array.

    evalAutoCorr()
                 Evaluate sum of auto-correlations of all antenna weights on 
                 the UV-plane. 

    evalPowerPattern()
                 Evaluate power pattern for the antenna from its zero-centered 
                 cross-correlated footprint

    evalPowerPatternSkypos()
                 Evaluate power pattern at the given sky positions for the 
                 antenna from its zero-centered cross-correlated footprint

    getStats()   Get statistics from images from inside specified boxes

    save()       Saves the image information to disk

    Read the member function docstrings for more details
    ----------------------------------------------------------------------------
    """

    def __init__(self, f0=None, f=None, pol=None, antenna_array=None,
                 interferometer_array=None, infile=None, timestamp=None,
                 verbose=True):
        
        """
        ------------------------------------------------------------------------
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
        holograph_PB_P1, img_P1, PB_P1, lf_P1, mf_P1, autocorr_wts_vuf, 
        autocorr_data_vuf

        Read docstring of class Image for details on these attributes.
        ------------------------------------------------------------------------
        """

        if verbose:
            print '\nInitializing an instance of class Image...\n'
            print '\tVerifying for compatible arguments...'

        if timestamp is not None:
            self.timestamp = timestamp
            if verbose:
                print '\t\tInitialized time stamp.'

        self.timestamps = []
        self.tbinsize = None

        if f0 is not None:
            self.f0 = f0
            if verbose:
                print '\t\tInitialized center frequency.'

        if f is not None:
            self.f = NP.asarray(f)
            if verbose:
                print '\t\tInitialized frequency channels.'

        self.measured_type = None
        self.antenna_array = None
        self.interferometer_array = None
        self.autocorr_set = False
        self.autocorr_removed = False

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

        self.grid_illumination = {}
        self.holimg = {}
        self.holbeam = {}
        self.img = {}
        self.beam = {}
        self.pbeam = {}
        self.gridl = {}
        self.gridm = {}
        self.grid_wts = {}
        self.grid_Ef = {}
        self.grid_Vf = {}
        self.holimg_stack = {}
        self.holbeam_stack = {}
        self.img_stack = {}
        self.beam_stack = {}
        self.grid_illumination_stack = {}
        self.grid_vis_stack = {}
        self.img_avg = {}
        self.beam_avg = {}
        self.grid_vis_avg = {}
        self.grid_illumination_avg = {}
        self.wts_vuf = {}
        self.vis_vuf = {}
        self.twts = {}
        self.autocorr_wts_vuf = {}
        self.autocorr_data_vuf = {}
        self.nzsp_grid_vis_avg = {}
        self.nzsp_grid_illumination_avg = {}
        self.nzsp_wts_vuf = {}
        self.nzsp_vis_vuf = {}
        self.nzsp_img_avg = {}
        self.nzsp_beam_avg = {}
        self.nzsp_img = {}
        self.nzsp_beam = {}

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
            
                if pol is None:
                    pol = ['P1', 'P2']
                pol = NP.unique(NP.asarray(pol))

                self.gridu, self.gridv = antenna_array.gridu, antenna_array.gridv
                for apol in ['P1', 'P2']:
                    self.holimg[apol] = None
                    self.holbeam[apol] = None
                    self.img[apol] = None
                    self.beam[apol] = None
                    self.grid_illumination[apol] = None
                    self.grid_Ef[apol] = None
                    self.grid_wts[apol] = None
                    self.holimg_stack[apol] = None
                    self.holbeam_stack[apol] = None
                    self.img_stack[apol] = None
                    self.beam_stack[apol] = None
                    self.grid_illumination_stack[apol] = None
                    self.grid_vis_stack[apol] = None
                    self.grid_vis_avg[apol] = None
                    self.grid_illumination_avg[apol] = None
                    self.img_avg[apol] = None
                    self.beam_avg[apol] = None
                    self.twts[apol] = None
                    self.wts_vuf[apol] = None
                    self.vis_vuf[apol] = None
                    self.autocorr_wts_vuf[apol] = None
                    self.autocorr_data_vuf[apol] = None
                    self.nzsp_grid_vis_avg[apol] = None
                    self.nzsp_grid_illumination_avg[apol] = None
                    self.nzsp_wts_vuf[apol] = None
                    self.nzsp_vis_vuf[apol] = None
                    self.nzsp_img_avg[apol] = None
                    self.nzsp_beam_avg[apol] = None
                    self.nzsp_img[apol] = None
                    self.nzsp_beam[apol] = None
                    self.pbeam[apol] = None

                self.antenna_array = antenna_array
                self.measured_type = 'E-field'

                if verbose:
                    print '\t\tInitialized gridded attributes for image object'
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
            
                if pol is None:
                    pol = ['P11', 'P12', 'P21', 'P22']
                pol = NP.unique(NP.asarray(pol))

                self.gridu, self.gridv = interferometer_array.gridu, interferometer_array.gridv
                for cpol in ['P11', 'P12', 'P21', 'P22']:
                    self.holimg[cpol] = None
                    self.holbeam[cpol] = None
                    self.img[cpol] = None
                    self.beam[cpol] = None
                    self.grid_illumination[cpol] = None
                    self.grid_Vf[cpol] = None
                    self.grid_wts[cpol] = None
                    self.holimg_stack[cpol] = None
                    self.holbeam_stack[cpol] = None
                    self.img_stack[cpol] = None
                    self.beam_stack[cpol] = None
                    self.grid_illumination_stack[cpol] = None
                    self.grid_vis_stack[cpol] = None
                    self.grid_vis_avg[cpol] = None
                    self.grid_illumination_avg[cpol] = None
                    self.img_avg[cpol] = None
                    self.beam_avg[cpol] = None
                    self.twts[cpol] = None
                    self.wts_vuf[cpol] = None
                    self.vis_vuf[cpol] = None
                    self.autocorr_wts_vuf[cpol] = None                    
                    self.nzsp_grid_vis_avg[cpol] = None
                    self.nzsp_grid_illumination_avg[cpol] = None
                    self.nzsp_wts_vuf[cpol] = None
                    self.nzsp_vis_vuf[cpol] = None
                    self.nzsp_img_avg[cpol] = None
                    self.nzsp_beam_avg[cpol] = None
                    self.nzsp_img[cpol] = None
                    self.nzsp_beam[cpol] = None
                    self.pbeam[cpol] = None
    
                self.interferometer_array = interferometer_array
                self.measured_type = 'visibility'

                if verbose:
                    print '\t\tInitialized gridded attributes for image object'
            else:
                raise TypeError('interferometer_array is not an instance of class InterferometerArray. Cannot initiate instance of class Image.')

        if verbose:
            print '\nSuccessfully initialized an instance of class Image\n'

    ############################################################################

    def reset(self, verbose=True):

        """
        ------------------------------------------------------------------------
        Reset some grid level attributes of image object to init values

        Inputs:

        verbose   [boolean] If True (default), prints diagnostic and progress
                  messages. If False, suppress printing such messages.

        The attributes reset to init values are grid_illumination, holbeam, 
        grid_Vf, grid_Ef, interferometer_array, antenna_array, holimg, gridl, 
        gridm, img, beam, grid_wts
        ------------------------------------------------------------------------
        """
        
        if verbose:
            print 'Resetting grid level attributes of image object...'

        self.antenna_array = None
        self.interferometer_array = None
        self.timestamp = None
        self.grid_illumination = {}
        self.holimg = {}
        self.holbeam = {}
        self.img = {}
        self.beam = {}
        self.gridl = {}
        self.gridm = {}
        self.grid_wts = {}
        self.grid_Ef = {}
        self.grid_Vf = {}
        self.wts_vuf = {}
        self.vis_vuf = {}

        if self.measured_type == 'E-field':
            for apol in ['P1', 'P2']:
                self.holimg[apol] = None
                self.holbeam[apol] = None
                self.img[apol] = None
                self.beam[apol] = None
                self.grid_illumination[apol] = None
                self.grid_Ef[apol] = None
                self.grid_wts[apol] = None
                self.wts_vuf[apol] = None
                self.vis_vuf[apol] = None
        else:
            for cpol in ['P11', 'P12', 'P21', 'P22']:
                self.holimg[cpol] = None
                self.holbeam[cpol] = None
                self.img[cpol] = None
                self.beam[cpol] = None
                self.grid_illumination[cpol] = None
                self.grid_Vf[cpol] = None
                self.grid_wts[cpol] = None
                self.wts_vuf[cpol] = None
                self.vis_vuf[cpol] = None

    ############################################################################

    def update(self, antenna_array=None, interferometer_array=None, reset=True, 
               verbose=True):

        """
        ------------------------------------------------------------------------
        Updates the image object with newer instance of class AntennaArray or
        InterferometerArray

        Inputs:

        antenna_array [instance of class AntennaArray] Update the image object 
                      with this new instance of class AntennaArray (if attribute
                      measured_type is 'E-field')

        interferometer_array 
                      [instance of class InterferometerArray] Update the image 
                      object with this new instance of class InterferometerArray 
                      (if attribute measured_type is 'visibility')

        reset         [boolean] if set to True (default), resets some of the
                      image object attribtues by calling member function reset()

        verbose       [boolean] If True (default), prints diagnostic and progress
                      messages. If False, suppress printing such messages.    
        ------------------------------------------------------------------------
        """

        if not isinstance(reset, bool):
            raise TypeError('reset keyword must be of boolean type')

        if not isinstance(verbose, bool):
            raise TypeError('verbose keyword must be of boolean type')

        if self.measured_type == 'E-field':
            if antenna_array is not None:
                if isinstance(antenna_array, AntennaArray):
                    if reset:
                        self.reset(verbose=verbose)
                        self.gridu, self.gridv = antenna_array.gridu, antenna_array.gridv
                        self.antenna_array = antenna_array
                else:
                    raise TypeError('Input antenna_array must be an instance of class AntennaArray')
                self.timestamp = antenna_array.timestamp
                if verbose:
                    print 'Updated antenna array attributes of the image instance'
        else:
            if interferometer_array is not None:
                if isinstance(interferometer_array, InterferometerArray):
                    if reset:
                        self.reset(verbose=verbose)
                        self.gridu, self.gridv = interferometer_array.gridu, interferometer_array.gridv
                        self.interferometer_array = interferometer_array
                else:
                    raise TypeError('Input interferometer_array must be an instance of class InterferometerArray')
                self.timestamp = interferometer_array.timestamp
                if verbose:
                    print 'Updated interferometer array attributes of the image instance'

    ############################################################################

    def imagr(self, pol=None, weighting='natural', pad=0, stack=True,
              grid_map_method='sparse', cal_loop=False, verbose=True):

        """
        ------------------------------------------------------------------------
        Imaging engine that performs inverse fourier transforms of appropriate
        electric fields or visibilities associated with the antenna array or
        interferometer array respectively.

        Keyword Inputs:

        pol       [string] indicates which polarization information to be 
                  imaged. Allowed values are 'P1', 'P2' or None (default). If 
                  None, both polarizations are imaged.
        
        weighting [string] indicates weighting scheme. Default='natural'. 
                  Accepted values are 'natural' and 'uniform'

        pad       [integer] indicates the amount of padding before imaging. 
                  In case of MOFF imaging the output image will be of size 
                  2**(pad+1) times the size of the antenna array grid along u- 
                  and v-axes. In case of FX imaging, the output image will be of
                  size 2**pad times the size of interferometer array grid along
                  u- and v-axes. Value must not be negative. Default=0 (implies 
                  padding by factor 2 along u- and v-axes for MOFF, and no 
                  padding for FX)

        stack     [boolean] If True (default), stacks the imaged and uv-gridded
                  data to the stack for batch processing later

        grid_map_method
                  [string] Accepted values are 'regular' and 'sparse' (default).
                  If 'regular' it applies the regular grid mapping while 
                  'sparse' applies the grid mapping based on sparse matrix 
                  methods

        cal_loop  [boolean] Applicable only in case when attribute 
                  measured_type is set to 'E-field' (MOFF imaging) and 
                  grid_map_method is set to 'sparse'. If True, the calibration 
                  loop is assumed to be ON and hence the calibrated electric 
                  fields are used in imaging. If False (default), the 
                  calibration loop is assumed to be OFF and the current stream 
                  of electric fields are assumed to be the calibrated data to 
                  be mapped to the grid 

        verbose   [boolean] If True (default), prints diagnostic and progress
                  messages. If False, suppress printing such messages.
        ------------------------------------------------------------------------
        """

        if verbose:
            print '\nPreparing to image...\n'

        if self.f is None:
            raise ValueError('Frequency channels have not been initialized. Cannot proceed with imaging.')

        if self.measured_type is None:
            raise ValueError('Measured type is unknown.')

        if not isinstance(pad, int):
            raise TypeError('Input keyword pad must be an integer')
        elif pad < 0:
            raise ValueError('Input keyword pad must not be negative')

        # if not isinstance(pad, str):
        #     raise TypeError('Input keyword pad must be a string')
        # else:
        #     if pad not in ['on', 'off']:
        #         raise ValueError('Invalid value specified for pad')

        du = self.gridu[0,1] - self.gridu[0,0]
        dv = self.gridv[1,0] - self.gridv[0,0]
        grid_shape = self.gridu.shape

        if self.measured_type == 'E-field':
            if pol is None: pol = ['P1', 'P2']
            pol = NP.unique(NP.asarray(pol)).tolist()
            for apol in pol:
                if apol in ['P1', 'P2']:
                    if grid_map_method == 'regular':
                        self.antenna_array.make_grid_cube_new(pol=apol, verbose=verbose)
                    elif grid_map_method == 'sparse':
                        self.antenna_array.applyMappingMatrix(pol=apol, cal_loop=cal_loop, verbose=verbose)
                    else:
                        raise ValueError('Invalid value specified for input parameter grid_map_method')

                    self.grid_wts[apol] = NP.zeros(self.gridu.shape+(self.f.size,))
                    if apol in self.antenna_array.grid_illumination:
                        if SpM.issparse(self.antenna_array.grid_illumination[apol]):
                            self.grid_illumination[apol] = self.antenna_array.grid_illumination[apol].A.reshape(self.gridu.shape+(self.f.size,))
                            self.grid_Ef[apol] = self.antenna_array.grid_Ef[apol].A.reshape(self.gridu.shape+(self.f.size,))
                        else:
                            self.grid_illumination[apol] = self.antenna_array.grid_illumination[apol]
                            self.grid_Ef[apol] = self.antenna_array.grid_Ef[apol]
                    
                    if verbose: print 'Preparing to Inverse Fourier Transform...'
                    if weighting == 'uniform':
                        self.grid_wts[apol][NP.abs(self.grid_illumination[apol]) > 0.0] = 1.0/NP.abs(self.grid_illumination[apol][NP.abs(self.grid_illumination[apol]) > 0.0])
                    else:
                        self.grid_wts[apol][NP.abs(self.grid_illumination[apol]) > 0.0] = 1.0

                    sum_wts = NP.sum(NP.abs(self.grid_wts[apol] * self.grid_illumination[apol]), axis=(0,1), keepdims=True)

                    syn_beam = NP.fft.fft2(self.grid_wts[apol]*self.grid_illumination[apol], s=[2**(pad+1) * self.gridu.shape[0], 2**(pad+1) * self.gridv.shape[1]], axes=(0,1))
                    dirty_image = NP.fft.fft2(self.grid_wts[apol]*self.grid_Ef[apol], s=[2**(pad+1) * self.gridu.shape[0], 2**(pad+1) * self.gridv.shape[1]], axes=(0,1))
                    self.gridl, self.gridm = NP.meshgrid(NP.fft.fftshift(NP.fft.fftfreq(2**(pad+1) * self.gridu.shape[1], du)), NP.fft.fftshift(NP.fft.fftfreq(2**(pad+1) * self.gridv.shape[0], dv)))

                    # if pad == 'on':
                    #     syn_beam = NP.fft.fft2(self.grid_wts[apol]*self.grid_illumination[apol], s=[4*self.gridu.shape[0], 4*self.gridv.shape[1]], axes=(0,1))
                    #     dirty_image = NP.fft.fft2(self.grid_wts[apol]*self.grid_Ef[apol], s=[4*self.gridu.shape[0], 4*self.gridv.shape[1]], axes=(0,1))
                    #     self.gridl, self.gridm = NP.meshgrid(NP.fft.fftshift(NP.fft.fftfreq(4*grid_shape[1], du)), NP.fft.fftshift(NP.fft.fftfreq(4*grid_shape[0], dv)))
                    # else:
                    #     syn_beam = NP.fft.fft2(self.grid_wts[apol]*self.grid_illumination[apol], axes=(0,1))
                    #     dirty_image = NP.fft.fft2(self.grid_wts[apol]*self.grid_Ef[apol], axes=(0,1))
                    #     self.gridl, self.gridm = NP.meshgrid(NP.fft.fftshift(NP.fft.fftfreq(grid_shape[1], du)), NP.fft.fftshift(NP.fft.fftfreq(grid_shape[0], dv)))

                    self.holbeam[apol] = NP.fft.fftshift(syn_beam/sum_wts, axes=(0,1))
                    self.holimg[apol] = NP.fft.fftshift(dirty_image/sum_wts, axes=(0,1))
                    syn_beam = NP.abs(syn_beam)**2
                    sum_wts2 = sum_wts**2
                    dirty_image = NP.abs(dirty_image)**2
                    self.beam[apol] = NP.fft.fftshift(syn_beam/sum_wts2, axes=(0,1))
                    self.img[apol] = NP.fft.fftshift(dirty_image/sum_wts2, axes=(0,1))
                    qty_vuf = NP.fft.ifft2(syn_beam/sum_wts2, axes=(0,1)) # Inverse FT
                    qty_vuf = NP.fft.ifftshift(qty_vuf, axes=(0,1)) # Shift array to be centered
                    # self.wts_vuf[apol] = qty_vuf[self.gridv.shape[0]:3*self.gridv.shape[0],self.gridu.shape[1]:3*self.gridu.shape[1],:]
                    self.wts_vuf[apol] = qty_vuf[qty_vuf.shape[0]/2-self.gridv.shape[0]:qty_vuf.shape[0]/2+self.gridv.shape[0], qty_vuf.shape[1]/2-self.gridu.shape[1]:qty_vuf.shape[1]/2+self.gridu.shape[1], :]
                    qty_vuf = NP.fft.ifft2(dirty_image/sum_wts2, axes=(0,1)) # Inverse FT
                    qty_vuf = NP.fft.ifftshift(qty_vuf, axes=(0,1)) # Shift array to be centered
                    self.vis_vuf[apol] = qty_vuf[qty_vuf.shape[0]/2-self.gridv.shape[0]:qty_vuf.shape[0]/2+self.gridv.shape[0], qty_vuf.shape[1]/2-self.gridu.shape[1]:qty_vuf.shape[1]/2+self.gridu.shape[1], :]
                       
        if self.measured_type == 'visibility':
            if pol is None: pol = ['P11', 'P12', 'P21', 'P22']
            pol = NP.unique(NP.asarray(pol)).tolist()
            for cpol in pol:
                if cpol in ['P11', 'P12', 'P21', 'P22']:
                    if grid_map_method == 'regular':
                        self.interferometer_array.make_grid_cube_new(verbose=verbose, pol=cpol)
                    elif grid_map_method == 'sparse':
                        self.interferometer_array.applyMappingMatrix(pol=cpol, verbose=verbose)
                    else:
                        raise ValueError('Invalid value specified for input parameter grid_map_method')

                    self.grid_wts[cpol] = NP.zeros(self.gridu.shape+(self.f.size,))
                    if cpol in self.interferometer_array.grid_illumination:
                        if SpM.issparse(self.interferometer_array.grid_illumination[cpol]):
                            self.grid_illumination[cpol] = self.interferometer_array.grid_illumination[cpol].A.reshape(self.gridu.shape+(self.f.size,))
                            self.grid_Vf[cpol] = self.interferometer_array.grid_Vf[cpol].A.reshape(self.gridu.shape+(self.f.size,))
                        else:
                            self.grid_illumination[cpol] = self.interferometer_array.grid_illumination[cpol]
                            self.grid_Vf[cpol] = self.interferometer_array.grid_Vf[cpol]

                    if verbose: print 'Preparing to Inverse Fourier Transform...'
                    if weighting == 'uniform':
                        self.grid_wts[cpol][NP.abs(self.grid_illumination[cpol]) > 0.0] = 1.0/NP.abs(self.grid_illumination[cpol][NP.abs(self.grid_illumination[cpol]) > 0.0])
                    else:
                        self.grid_wts[cpol][NP.abs(self.grid_illumination[cpol]) > 0.0] = 1.0

                    sum_wts = NP.sum(NP.abs(self.grid_wts[cpol] * self.grid_illumination[cpol]), axis=(0,1), keepdims=True)

                    padded_syn_beam_in_uv = NP.pad(self.grid_wts[cpol]*self.grid_illumination[cpol], (((2**pad-1)*self.gridv.shape[0]/2,(2**pad-1)*self.gridv.shape[0]/2),((2**pad-1)*self.gridu.shape[1]/2,(2**pad-1)*self.gridu.shape[1]/2),(0,0)), mode='constant', constant_values=0)
                    padded_grid_Vf = NP.pad(self.grid_wts[cpol]*self.grid_Vf[cpol], (((2**pad-1)*self.gridv.shape[0]/2,(2**pad-1)*self.gridv.shape[0]/2),((2**pad-1)*self.gridu.shape[1]/2,(2**pad-1)*self.gridu.shape[1]/2),(0,0)), mode='constant', constant_values=0)
                    self.gridl, self.gridm = NP.meshgrid(NP.fft.fftshift(NP.fft.fftfreq(2**pad * grid_shape[1], du)), NP.fft.fftshift(NP.fft.fftfreq(2**pad * grid_shape[0], dv)))

                    # if pad == 'on': # Pad it with zeros on either side to be twice the size
                    #     padded_syn_beam_in_uv = NP.pad(self.grid_wts[cpol]*self.grid_illumination[cpol], ((self.gridv.shape[0]/2,self.gridv.shape[0]/2),(self.gridu.shape[1]/2,self.gridu.shape[1]/2),(0,0)), mode='constant', constant_values=0)
                    #     padded_grid_Vf = NP.pad(self.grid_wts[cpol]*self.grid_Vf[cpol], ((self.gridv.shape[0]/2,self.gridv.shape[0]/2),(self.gridu.shape[1]/2,self.gridu.shape[1]/2),(0,0)), mode='constant', constant_values=0)
                    #     self.gridl, self.gridm = NP.meshgrid(NP.fft.fftshift(NP.fft.fftfreq(2*grid_shape[1], du)), NP.fft.fftshift(NP.fft.fftfreq(2*grid_shape[0], dv)))
                    # else:  # No padding
                    #     padded_syn_beam_in_uv = self.grid_wts[cpol]*self.grid_illumination[cpol]
                    #     padded_grid_Vf = self.grid_wts[cpol]*self.grid_Vf[cpol]
                    #     self.gridl, self.gridm = NP.meshgrid(NP.fft.fftshift(NP.fft.fftfreq(grid_shape[1], du)), NP.fft.fftshift(NP.fft.fftfreq(grid_shape[0], dv)))

                    # Shift to be centered
                    padded_syn_beam_in_uv = NP.fft.ifftshift(padded_syn_beam_in_uv, axes=(0,1))
                    padded_grid_Vf = NP.fft.ifftshift(padded_grid_Vf, axes=(0,1))

                    # Compute the synthesized beam. It is at a finer resolution due to padding
                    syn_beam = NP.fft.fft2(padded_syn_beam_in_uv, axes=(0,1))
                    dirty_image = NP.fft.fft2(padded_grid_Vf, axes=(0,1))
        
                    # Select only the real part, equivalent to adding conjugate baselines
                    dirty_image = dirty_image.real
                    syn_beam = syn_beam.real

                    self.beam[cpol] = NP.fft.fftshift(syn_beam/sum_wts, axes=(0,1))
                    self.img[cpol] = NP.fft.fftshift(dirty_image/sum_wts, axes=(0,1))
                    qty_vuf = NP.fft.ifft2(syn_beam/sum_wts, axes=(0,1)) # Inverse FT
                    qty_vuf = NP.fft.ifftshift(qty_vuf, axes=(0,1)) # Shift array to be centered
                    # self.wts_vuf[cpol] = qty_vuf[self.gridv.shape[0]/2:3*self.gridv.shape[0]/2,self.gridu.shape[1]/2:3*self.gridu.shape[1]/2,:]
                    self.wts_vuf[cpol] = qty_vuf[qty_vuf.shape[0]/2-self.gridv.shape[0]/2:qty_vuf.shape[0]/2+self.gridv.shape[0]/2, qty_vuf.shape[1]/2-self.gridu.shape[1]/2:qty_vuf.shape[1]/2+self.gridu.shape[1]/2,:]
                    qty_vuf = NP.fft.ifft2(dirty_image/sum_wts, axes=(0,1)) # Inverse FT
                    qty_vuf = NP.fft.ifftshift(qty_vuf, axes=(0,1)) # Shift array to be centered
                    # self.vis_vuf[cpol] = qty_vuf[self.gridv.shape[0]/2:3*self.gridv.shape[0]/2,self.gridu.shape[1]/2:3*self.gridu.shape[1]/2,:]
                    self.vis_vuf[cpol] = qty_vuf[qty_vuf.shape[0]/2-self.gridv.shape[0]/2:qty_vuf.shape[0]/2+self.gridv.shape[0]/2, qty_vuf.shape[1]/2-self.gridu.shape[1]/2:qty_vuf.shape[1]/2+self.gridu.shape[1]/2,:]

        nan_ind = NP.where(self.gridl**2 + self.gridm**2 > 1.0)
        # nan_ind_unraveled = NP.unravel_index(nan_ind, self.gridl.shape)
        # self.beam[cpol][nan_ind_unraveled,:] = NP.nan
        # self.img[cpol][nan_ind_unraveled,:] = NP.nan    

        if verbose:
            print 'Successfully imaged.'

        # Call stack() if required
        if stack:
            self.stack(pol=pol)

    ############################################################################
        
    def stack(self, pol=None):

        """
        ------------------------------------------------------------------------
        Stacks current images and UV-grid information onto a stack

        Inputs:

        pol     [string] indicates which polarization information to be saved. 
                Allowed values are 'P1', 'P2' in case of MOFF or 'P11', 'P12', 
                'P21', 'P22' in case of FX or None (default). If None, 
                information on all polarizations appropriate for MOFF or FX 
                are stacked
        ------------------------------------------------------------------------
        """

        if self.timestamp not in self.timestamps:
            if pol is None:
                if self.measured_type == 'E-field':
                    pol = ['P1', 'P2']
                else:
                    pol = ['P11', 'P12', 'P21', 'P22']
            elif isinstance(pol, str):
                pol = [pol]
            elif isinstance(pol, list):
                p = [item for item in pol if item in ['P1', 'P2', 'P11', 'P12', 'P21', 'P22']]
                pol = p
            else:
                raise TypeError('Input pol must be a string or list specifying polarization(s)')
    
            for p in pol:
                if self.img_stack[p] is None:
                    self.img_stack[p] = self.img[p][NP.newaxis,:,:,:]
                    self.beam_stack[p] = self.beam[p][NP.newaxis,:,:,:]
                    self.grid_illumination_stack[p] = self.wts_vuf[p][NP.newaxis,:,:,:]
                    self.grid_vis_stack[p] = self.vis_vuf[p][NP.newaxis,:,:,:]
                else:
                    self.img_stack[p] = NP.concatenate((self.img_stack[p], self.img[p][NP.newaxis,:,:,:]), axis=0)
                    self.beam_stack[p] = NP.concatenate((self.beam_stack[p], self.beam[p][NP.newaxis,:,:,:]), axis=0)
                    self.grid_illumination_stack[p] = NP.concatenate((self.grid_illumination_stack[p], self.wts_vuf[p][NP.newaxis,:,:,:]), axis=0)
                    self.grid_vis_stack[p] = NP.concatenate((self.grid_vis_stack[p], self.vis_vuf[p][NP.newaxis,:,:,:]), axis=0)
    
                if self.measured_type == 'E-field':
                    if self.holimg_stack[p] is None:
                        self.holimg_stack[p] = self.holimg[p][NP.newaxis,:,:,:]
                        self.holbeam_stack[p] = self.holbeam[p][NP.newaxis,:,:,:]
                    else:
                        self.holimg_stack[p] = NP.concatenate((self.holimg_stack[p], self.holimg[p][NP.newaxis,:,:,:]), axis=0)
                        self.holbeam_stack[p] = NP.concatenate((self.holbeam_stack[p], self.holbeam[p][NP.newaxis,:,:,:]), axis=0)

            self.timestamps += [self.timestamp]

    ############################################################################

    def accumulate(self, tbinsize=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Accumulates and averages gridded quantities that are statistically
        stationary such as images and visibilities

        Input:

        tbinsize [scalar or dictionary] Contains bin size of timestamps while
                 averaging. Default = None means gridded quantities over all
                 timestamps are averaged. If scalar, the same (positive) value 
                 applies to all polarizations. If dictionary, timestamp bin size
                 (positive) is provided under each key 'P11', 'P12', 'P21', 
                 'P22'. If any of the keys is missing the gridded quantities 
                 for that polarization are averaged over all timestamps.

        verbose  [boolean] If True (default), prints diagnostic and progress
                 messages. If False, suppress printing such messages.
        ------------------------------------------------------------------------
        """
        
        if self.measured_type == 'E-field':
            pol = ['P1', 'P2']
        else:
            pol = ['P11', 'P12', 'P21', 'P22']

        timestamps = NP.asarray(self.timestamps).astype(NP.float)
        twts = {}
        img_acc = {}
        beam_acc = {}
        grid_vis_acc = {}
        grid_illumination_acc = {}
        for p in pol:
            img_acc[p] = None
            beam_acc[p] = None
            grid_vis_acc[p] = None
            grid_illumination_acc[p] = None
            twts[p] = []

        if tbinsize is None:   # Average across all timestamps
            for p in pol:
                if self.img_stack[p] is not None:
                    img_acc[p] = NP.nansum(self.img_stack[p], axis=0, keepdims=True)
                    beam_acc[p] = NP.nansum(self.beam_stack[p], axis=0, keepdims=True)
                    grid_vis_acc[p] = NP.nansum(self.grid_vis_stack[p], axis=0, keepdims=True)
                    grid_illumination_acc[p] = NP.nansum(self.grid_illumination_stack[p], axis=0, keepdims=True)
                twts[p] = NP.asarray(len(self.timestamps)).reshape(-1,1,1,1)
            self.tbinsize = tbinsize
        elif isinstance(tbinsize, (int, float)): # Apply same time bin size to all polarizations 
            eps = 1e-10
            tbins = NP.arange(timestamps.min(), timestamps.max(), tbinsize)
            tbins = NP.append(tbins, timestamps.max()+eps)
            for p in pol:
                counts, tbin_edges, tbinnum, ri = OPS.binned_statistic(timestamps, statistic='count', bins=tbins)
                for binnum in range(counts.size):
                    ind = ri[ri[binnum]:ri[binnum+1]]
                    twts[p] += [counts]
                    if img_acc[p] is None:
                        if self.img_stack[p] is not None:
                            img_acc[p] = NP.nansum(self.img_stack[p][ind,:,:,:], axis=0, keepdims=True)
                            beam_acc[p] = NP.nansum(self.beam_stack[p][ind,:,:,:], axis=0, keepdims=True)
                            grid_vis_acc[p] = NP.nansum(self.grid_vis_stack[p][ind,:,:,:], axis=0, keepdims=True)
                            grid_illumination_acc[p] = NP.nansum(self.grid_illumination_stack[p][ind,:,:,:], axis=0, keepdims=True)
                    else:
                        if self.img_stack[p] is not None:
                            img_acc[p] = NP.vstack((img_acc[p], NP.nansum(self.img_stack[p][ind,:,:,:], axis=0, keepdims=True)))
                            beam_acc[p] = NP.vstack((beam_acc[p], NP.nansum(self.beam_stack[p][ind,:,:,:], axis=0, keepdims=True)))
                            grid_vis_acc[p] = NP.vstack((grid_vis_acc[p], NP.nansum(self.grid_vis_stack[p][ind,:,:,:], axis=0, keepdims=True)))
                            grid_illumination_acc[p] = NP.vstack((grid_illumination_acc[p], NP.nansum(self.grid_illumination_stack[p][ind,:,:,:], axis=0, keepdims=True)))
                twts[p] = NP.asarray(twts[p]).astype(NP.float).reshape(-1,1,1,1)
            self.tbinsize = tbinsize
        elif isinstance(tbinsize, dict): # Apply different time binsizes to corresponding polarizations
            tbsize = {}
            for p in pol:
                if p not in tbinsize:
                    if self.img_stack[p] is not None:
                        img_acc[p] = NP.nansum(self.img_stack[p], axis=0, keepdims=True)
                        beam_acc[p] = NP.nansum(self.beam_stack[p], axis=0, keepdims=True)
                        grid_vis_acc[p] = NP.nansum(self.grid_vis_stack[p], axis=0, keepdims=True)
                        grid_illumination_acc[p] = NP.nansum(self.grid_illumination_stack[p], axis=0, keepdims=True)
                    twts[p] = NP.asarray(len(self.timestamps)).reshape(-1,1,1,1)
                    tbsize[p] = None
                elif isinstance(tbinsize[p], (int,float)):
                    eps = 1e-10
                    tbins = NP.arange(timestamps.min(), timestamps.max(), tbinsize[p])
                    tbins = NP.append(tbins, timestamps.max()+eps)
                    
                    counts, tbin_edges, tbinnum, ri = OPS.binned_statistic(timestamps, statistic='count', bins=tbins)
                    for binnum in range(counts.size):
                        ind = ri[ri[binnum]:ri[binnum+1]]
                        twts[p] += [counts]
                        if img_acc[p] is None:
                            if self.img_stack[p] is not None:
                                img_acc[p] = NP.nansum(self.img_stack[p][ind,:,:,:], axis=0, keepdims=True)
                                beam_acc[p] = NP.nansum(self.beam_stack[p][ind,:,:,:], axis=0, keepdims=True)
                                grid_vis_acc[p] = NP.nansum(self.grid_vis_stack[p][ind,:,:,:], axis=0, keepdims=True)
                                grid_illumination_acc[p] = NP.nansum(self.grid_illumination_stack[p][ind,:,:,:], axis=0, keepdims=True)
                        else:
                            if self.img_stack[p] is not None:
                                img_acc[p] = NP.vstack((img_acc[p], NP.nansum(self.img_stack[p][ind,:,:,:], axis=0, keepdims=True)))
                                beam_acc[p] = NP.vstack((beam_acc[p], NP.nansum(self.beam_stack[p][ind,:,:,:], axis=0, keepdims=True)))
                                grid_vis_acc[p] = NP.vstack((grid_vis_acc[p], NP.nansum(self.grid_vis_stack[p][ind,:,:,:], axis=0, keepdims=True)))
                                grid_illumination_acc[p] = NP.vstack((grid_illumination_acc[p], NP.nansum(self.grid_illumination_stack[p][ind,:,:,:], axis=0, keepdims=True)))
                    twts[p] = NP.asarray(twts[p]).astype(NP.float).reshape(-1,1,1,1)
                    tbsize[p] = tbinsize[p]
                else:
                    if self.img_stack[p] is not None:
                        img_acc[p] = NP.nansum(self.img_stack[p], axis=0, keepdims=True)
                        beam_acc[p] = NP.nansum(self.beam_stack[p], axis=0, keepdims=True)
                        grid_vis_acc[p] = NP.nansum(self.grid_vis_stack[p], axis=0, keepdims=True)
                        grid_illumination_acc[p] = NP.nansum(self.grid_illumination_stack[p], axis=0, keepdims=True)
                    twts[p] = NP.asarray(len(self.timestamps)).reshape(-1,1,1,1)
                    tbsize[p] = None

            self.tbinsize = tbsize

        # Compute the averaged grid quantities from the accumulated versions
        for p in pol:
            if img_acc[p] is not None:
                self.img_avg[p] = img_acc[p] / twts[p]
                self.beam_avg[p] = beam_acc[p] / twts[p]
                self.grid_vis_avg[p] = grid_vis_acc[p] / twts[p]
                self.grid_illumination_avg[p] = grid_illumination_acc[p] / twts[p]

        self.twts = twts

    ############################################################################

    # def evalAutoCorr(self, lkpinfo=None, forceeval=False):

    #     """
    #     ------------------------------------------------------------------------
    #     Evaluate auto-correlation of single antenna weights with itself on the
    #     UV-plane. 

    #     Inputs:

    #     lkpinfo   [dictionary] consists of weights information for each of 
    #               the polarizations under polarization keys. Each of 
    #               the values under the keys is a string containing the full
    #               path to a filename that contains the positions and 
    #               weights for the aperture illumination in the form of 
    #               a lookup table as columns (x-loc [float], y-loc 
    #               [float], wts[real], wts[imag if any]). In this case, the 
    #               lookup is for auto-corrlation of antenna weights. It only 
    #               applies when the antenna aperture class is set to 
    #               lookup-based kernel estimation instead of a functional form

    #     forceeval [boolean] When set to False (default) the auto-correlation in
    #               the UV plane is not evaluated if it was already evaluated 
    #               earlier. If set to True, it will be forcibly evaluated 
    #               independent of whether they were already evaluated or not
    #     ------------------------------------------------------------------------
    #     """

    #     if forceeval or (not self.autocorr_set):
    #         if self.measured_type == 'E-field':
    
    #             pol = ['P1', 'P2']
    
    #             # Assume all antenna apertures are identical and make a copy of the
    #             # antenna aperture
    #             # Need serious development for non-identical apertures
    
    #             ant_aprtr = copy.deepcopy(self.antenna_array.antennas.itervalues().next().aperture)
    #             pol_type = 'dual'
    #             kerntype = ant_aprtr.kernel_type
    #             shape = ant_aprtr.shape
    #             # kernshapeparms = {'xmax': {p: ant_aprtr.xmax[p] for p in pol}, 'ymax': {p: ant_aprtr.ymax[p] for p in pol}, 'rmin': {p: ant_aprtr.rmin[p] for p in pol}, 'rmax': {p: ant_aprtr.rmax[p] for p in pol}, 'rotangle': {p: ant_aprtr.rotangle[p] for p in pol}}
    #             kernshapeparms = {p: {'xmax': ant_aprtr.xmax[p], 'ymax': ant_aprtr.ymax[p], 'rmax': ant_aprtr.rmax[p], 'rmin': ant_aprtr.rmin[p], 'rotangle': ant_aprtr.rotangle[p]} for p in pol}
    
    #             for p in pol:
    #                 if kerntype[p] == 'func':
    #                     if shape[p] == 'rect':
    #                         shape[p] = 'auto_convolved_rect'
    #                     elif shape[p] == 'square':
    #                         shape[p] = 'auto_convolved_square'
    #                     elif shape[p] == 'circular':
    #                         shape[p] = 'auto_convolved_circular'
    #                     else:
    #                         raise ValueError('Aperture kernel footprint shape - {0} - currently unsupported'.format(shape[p]))
                        
    #             aprtr = APR.Aperture(pol_type=pol_type, kernel_type=kerntype,
    #                                  shape=shape, parms=kernshapeparms,
    #                                  lkpinfo=lkpinfo, load_lookup=True)
                
    #             du = self.gridu[0,1] - self.gridu[0,0]
    #             dv = self.gridv[1,0] - self.gridv[0,0]
    #             if self.measured_type == 'E-field':
    #                 gridu, gridv = NP.meshgrid(du*(NP.arange(2*self.gridu.shape[1])-self.gridu.shape[1]), dv*(NP.arange(2*self.gridu.shape[0])-self.gridu.shape[0]))
    #             else:
    #                 gridu, gridv = self.gridu, self.gridv
    
    #             wavelength = FCNST.c / self.f
    #             min_lambda = NP.abs(wavelength).min()
    #             rmaxNN = 0.5 * NP.sqrt(du**2 + dv**2) * min_lambda 
    
    #             gridx = gridu[:,:,NP.newaxis] * wavelength.reshape(1,1,-1)
    #             gridy = gridv[:,:,NP.newaxis] * wavelength.reshape(1,1,-1)
    #             gridxy = NP.hstack((gridx.reshape(-1,1), gridy.reshape(-1,1)))
    #             wl = NP.ones(gridu.shape)[:,:,NP.newaxis] * wavelength.reshape(1,1,-1)
    #             wl = wl.reshape(-1)
    #             distNN = 2.0 * max([NP.sqrt(aprtr.xmax['P1']**2 + NP.sqrt(aprtr.ymax['P1']**2)), NP.sqrt(aprtr.xmax['P2']**2 + NP.sqrt(aprtr.ymax['P2']**2)), aprtr.rmax['P1'], aprtr.rmax['P2']]) # factor in the front is to safely estimate kernel around some extra grid pixels
    #             indNN_list, blind, vuf_gridind = LKP.find_NN(NP.zeros((1,2)), gridxy, distance_ULIM=distNN, flatten=True, parallel=False)
    #             dxy = gridxy[vuf_gridind,:]
    #             unraveled_vuf_ind = NP.unravel_index(vuf_gridind, gridu.shape+(self.f.size,))
    
    #             self.autocorr_wts_vuf = {p: NP.zeros(gridu.shape+(self.f.size,), dtype=NP.complex64) for p in pol}
    #             # self.pbeam = {p: NP.zeros((2*gridv.shape[0],2*gridu.shape[1],self.f.size), dtype=NP.complex64) for p in pol}                
    #             for p in pol:
    #                 krn = aprtr.compute(dxy, wavelength=wl[vuf_gridind], pol=p, rmaxNN=rmaxNN, load_lookup=False)
    #                 self.autocorr_wts_vuf[p][unraveled_vuf_ind] = krn[p]
    #                 self.autocorr_wts_vuf[p] = self.autocorr_wts_vuf[p] / NP.sum(self.autocorr_wts_vuf[p], axis=(0,1), keepdims=True)
    #                 # sum_wts = NP.sum(self.autocorr_wts_vuf[p], axis=(0,1), keepdims=True)
    #                 # padded_wts_vuf = NP.pad(self.autocorr_wts_vuf[p], ((self.gridv.shape[0],self.gridv.shape[0]),(self.gridu.shape[1],self.gridu.shape[1]),(0,0)), mode='constant', constant_values=0)
    #                 # padded_wts_vuf = NP.fft.ifftshift(padded_wts_vuf, axes=(0,1))
    #                 # wts_lmf = NP.fft.fft2(padded_wts_vuf, axes=(0,1)) / sum_wts
    #                 # if NP.abs(wts_lmf.imag).max() < 1e-10:
    #                 #     self.pbeam[p] = NP.fft.fftshift(wts_lmf.real, axes=(0,1))
    #                 # else:
    #                 #     raise ValueError('Significant imaginary component found in the power pattern')
                    
    #             self.autocorr_set = True

    ############################################################################

    def evalAutoCorr(self, datapool='avg', forceeval=False):

        """
        ------------------------------------------------------------------------
        Evaluate sum of auto-correlations of all antenna weights on the
        UV-plane. 

        Inputs:

        datapool  [string] Specifies whether data to be used in determining the
                  auto-correlation the E-fields to be used come from
                  'stack', 'current', or 'avg' (default). Squared electric 
                  fields will be used if set to 'current' or 'stack', and 
                  averaged squared electric fields if set to 'avg'

        forceeval [boolean] When set to False (default) the auto-correlation in
                  the UV plane is not evaluated if it was already evaluated 
                  earlier. If set to True, it will be forcibly evaluated 
                  independent of whether they were already evaluated or not
        ------------------------------------------------------------------------
        """

        if forceeval or (not self.autocorr_set):
            self.autocorr_wts_vuf, self.autocorr_data_vuf = self.antenna_array.makeAutoCorrCube(datapool=datapool, tbinsize=self.tbinsize)
            self.autocorr_set = True
            
    ############################################################################
    
    def evalPowerPattern(self, pad=0, skypos=None, datapool='avg'):

        """
        ------------------------------------------------------------------------
        Evaluate power pattern for the antenna from its zero-centered 
        cross-correlated footprint

        Input:

        datapool
                [string] Specifies whether weights to be used in determining 
                the power pattern come from 'stack', 'current', or 'avg' 
                (default). 

        skypos  [numpy array] Positions on sky at which power pattern is 
                to be esimated. It is a 2- or 3-column numpy array in 
                direction cosine coordinates. It must be of size nsrc x 2 
                or nsrc x 3. If set to None (default), the power pattern is 
                estimated over a grid on the sky. If a numpy array is
                specified, then power pattern at the given locations is 
                estimated.

        pad     [integer] indicates the amount of padding before estimating
                power pattern image. Applicable only when attribute 
                measured_type is set to 'E-field' (MOFF imaging). The output 
                image of the power pattern will be of size 2**pad-1 times the 
                size of the antenna array grid along u- and v-axes. Value must 
                not be negative. Default=0 (implies no padding). pad=1 implies 
                padding by factor 2 along u- and v-axes for MOFF

        Outputs:

        pbinfo is a dictionary with the following keys and values:
        'pb'    [dictionary] Dictionary with keys 'P1' and 'P2' for 
                polarization. Under each key is a numpy array of estimated 
                power patterns. If skypos was set to None, the numpy array is 
                3D masked array of size nm x nl x nchan. The mask is based on 
                which parts of the grid are valid direction cosine coordinates 
                on the sky. If skypos was a numpy array denoting specific sky 
                locations, the value in this key is a 2D numpy array of size 
                nsrc x nchan
        'llocs' [None or numpy array] If the power pattern estimated is a grid
                (if input skypos was set to None), it contains the l-locations
                of the grid on the sky. If input skypos was not set to None, 
                the value under this key is set to None
        'mlocs' [None or numpy array] If the power pattern estimated is a grid
                (if input skypos was set to None), it contains the m-locations
                of the grid on the sky. If input skypos was not set to None, 
                the value under this key is set to None
        ------------------------------------------------------------------------
        """

        if not isinstance(pad, int):
            raise TypeError('Input keyword pad must be an integer')
        
        if datapool not in ['recent', 'stack', 'avg']:
            raise ValueError('Invalid value specified for input datapool')

        self.antenna_array.evalAllAntennaPairCorrWts()
        centered_crosscorr_wts_vuf = self.antenna_array.makeCrossCorrWtsCube()
        du = self.antenna_array.gridu[0,1] - self.antenna_array.gridu[0,0]
        dv = self.antenna_array.gridv[1,0] - self.antenna_array.gridv[0,0]
        ulocs = du*(NP.arange(2*self.antenna_array.gridu.shape[1])-self.antenna_array.gridu.shape[1])
        vlocs = dv*(NP.arange(2*self.antenna_array.gridv.shape[0])-self.antenna_array.gridv.shape[0])
        pol = ['P1', 'P2']
        pbinfo = {'pb': {}}
        for p in pol:
            pb = evalCorrWts2PB(centered_crosscorr_wts_vuf[p], ulocs, vlocs, pad=pad, skypos=skypos)
            pbinfo['pb'][p] = pb['pb']
            pbinfo['llocs'] = pb['llocs']
            pbinfo['mlocs'] = pb['mlocs']

        return pbinfo

        # shape_tuple = tuple(2 * NP.asarray(self.antenna_array.gridu.shape)) + (self.f.size,)
        # if self.measured_type == 'E-field':
        #     self.pbeam = {p: None for p in pol}                
        #     for p in pol:
        #         sum_wts = centered_crosscorr_wts_vuf[p].sum(axis=0).A # 1 x nchan
        #         sum_wts = sum_wts[NP.newaxis,:,:] # 1 x 1 x nchan
        #         padded_wts_vuf = NP.pad(centered_crosscorr_wts_vuf[p].toarray().reshape(shape_tuple), (((2**pad-1)*self.gridv.shape[0],(2**pad-1)*self.gridv.shape[0]),((2**pad-1)*self.gridu.shape[1],(2**pad-1)*self.gridu.shape[1]),(0,0)), mode='constant', constant_values=0)
        #         padded_wts_vuf = NP.fft.ifftshift(padded_wts_vuf, axes=(0,1))
        #         wts_lmf = NP.fft.fft2(padded_wts_vuf, axes=(0,1)) / sum_wts
        #         self.pbeam[p] = NP.fft.fftshift(wts_lmf.real, axes=(0,1))

    ############################################################################
    
    # def evalPowerPattern(self, pad=0):

    #     """
    #     ------------------------------------------------------------------------
    #     Evaluate power pattern for the antenna from its auto-correlated 
    #     footprint

    #     Input:

    #     pad     [integer] indicates the amount of padding before estimating
    #             power pattern image. Applicable only when attribute 
    #             measured_type is set to 'E-field' (MOFF imaging). The output 
    #             image of the pwoer pattern will be of size 2**pad-1 times the 
    #             size of the antenna array grid along u- and v-axes. Value must 
    #             not be negative. Default=0 (implies no padding of the 
    #             auto-correlated footprint). pad=1 implies padding by factor 2 
    #             along u- and v-axes for MOFF
    #     ------------------------------------------------------------------------
    #     """

    #     if not isinstance(pad, int):
    #         raise TypeError('Input keyword pad must be an integer')
        
    #     if not self.autocorr_set:
    #         self.evalAutoCorr()

    #     pol = ['P1', 'P2']
    #     if self.measured_type == 'E-field':
    #         self.pbeam = {p: None for p in pol}                
    #         for p in pol:
    #             sum_wts = NP.sum(self.autocorr_wts_vuf[p], axis=(0,1), keepdims=True)
    #             padded_wts_vuf = NP.pad(self.autocorr_wts_vuf[p], (((2**pad-1)*self.gridv.shape[0],(2**pad-1)*self.gridv.shape[0]),((2**pad-1)*self.gridu.shape[1],(2**pad-1)*self.gridu.shape[1]),(0,0)), mode='constant', constant_values=0)
    #             padded_wts_vuf = NP.fft.ifftshift(padded_wts_vuf, axes=(0,1))
    #             wts_lmf = NP.fft.fft2(padded_wts_vuf, axes=(0,1)) / sum_wts
    #             if NP.abs(wts_lmf.imag).max() < 1e-10:
    #                 self.pbeam[p] = NP.fft.fftshift(wts_lmf.real, axes=(0,1))
    #             else:
    #                 raise ValueError('Significant imaginary component found in the power pattern')

    ############################################################################

    def removeAutoCorr(self, lkpinfo=None, forceeval=False, datapool='avg',
                       pad=0):

        """
        ------------------------------------------------------------------------
        Remove auto-correlation of single antenna weights with itself from the
        UV-plane. 

        Inputs:

        lkpinfo   [dictionary] consists of weights information for each of 
                  the polarizations under polarization keys. Each of 
                  the values under the keys is a string containing the full
                  path to a filename that contains the positions and 
                  weights for the aperture illumination in the form of 
                  a lookup table as columns (x-loc [float], y-loc 
                  [float], wts[real], wts[imag if any]). In this case, the 
                  lookup is for auto-corrlation of antenna weights. It only 
                  applies when the antenna aperture class is set to 
                  lookup-based kernel estimation instead of a functional form

        forceeval [boolean] When set to False (default) the auto-correlation in
                  the UV plane is not evaluated if it was already evaluated 
                  earlier. If set to True, it will be forcibly evaluated 
                  independent of whether they were already evaluated or not

        datapool  [string] When set to 'avg' (or None) (default), 
                  auto-correlations from antennas (zero-spacing with a width) 
                  are removed from the averaged data set. If set to 'current',
                  the latest timestamp is used in subtracting the zero-spacing
                  visibilities information

        pad       [integer] indicates the amount of padding before imaging.
                  Applicable only when attribute measured_type is set to 
                  'E-field' (MOFF imaging). The output image will be of size 
                  2**pad-1 times the size of the antenna array grid along u- 
                  and v-axes. Value must not be negative. Default=0 (implies no 
                  padding of the auto-correlated footprint). pad=1 implies 
                  padding by factor 2 along u- and v-axes for MOFF
        ------------------------------------------------------------------------
        """

        if self.measured_type == 'E-field':
            if forceeval or (not self.autocorr_removed):
                if isinstance(datapool, str):
                    if datapool is None: datapool = 'avg'
                    if datapool not in ['avg', 'current']:
                        raise ValueError('Input keywrod datapool must be set to "avg" or "current"')
                else:
                    raise TypeError('Input keyword data pool must be a string')
        
                if forceeval or (not self.autocorr_set):
                    self.evalAutoCorr(forceeval=forceeval)
                    # self.evalAutoCorr(lkpinfo=lkpinfo, forceeval=forceeval)
        
                autocorr_wts_vuf = copy.deepcopy(self.autocorr_wts_vuf)
                autocorr_data_vuf = copy.deepcopy(self.autocorr_data_vuf)
                pol = ['P1', 'P2']
                for p in pol:
                    if datapool == 'avg':
                        if self.grid_illumination_avg[p] is not None:
                            vis_vuf = NP.copy(self.grid_vis_avg[p])
                            wts_vuf = NP.copy(self.grid_illumination_avg[p])
    
                            # autocorr_wts_vuf[p] = autocorr_wts_vuf[p][NP.newaxis,:,:,:]
                            vis_vuf = vis_vuf - (vis_vuf[:,self.gridv.shape[0],self.gridu.shape[1],:][:,NP.newaxis,NP.newaxis,:] / autocorr_data_vuf[p][:,self.gridv.shape[0],self.gridu.shape[1],:][:,NP.newaxis,NP.newaxis,:]) * autocorr_data_vuf[p]
                            wts_vuf = wts_vuf - (wts_vuf[:,self.gridv.shape[0],self.gridu.shape[1],:][:,NP.newaxis,NP.newaxis,:] / autocorr_wts_vuf[p][:,self.gridv.shape[0],self.gridu.shape[1],:][:,NP.newaxis,NP.newaxis,:]) * autocorr_wts_vuf[p]
                            sum_wts = NP.sum(wts_vuf, axis=(1,2), keepdims=True)
                            padded_wts_vuf = NP.pad(wts_vuf, ((0,0),((2**pad-1)*self.gridv.shape[0],(2**pad-1)*self.gridv.shape[0]),((2**pad-1)*self.gridu.shape[1],(2**pad-1)*self.gridu.shape[1]),(0,0)), mode='constant', constant_values=0)
                            padded_wts_vuf = NP.fft.ifftshift(padded_wts_vuf, axes=(1,2))
                            wts_lmf = NP.fft.fft2(padded_wts_vuf, axes=(1,2)) / sum_wts
                            if NP.abs(wts_lmf.imag).max() > 1e-10:
                                raise ValueError('Significant imaginary component found in the synthesized beam.')
                            self.nzsp_beam_avg[p] = NP.fft.fftshift(wts_lmf.real, axes=(1,2))
                            padded_vis_vuf = NP.pad(vis_vuf, ((0,0),((2**pad-1)*self.gridv.shape[0],(2**pad-1)*self.gridv.shape[0]),((2**pad-1)*self.gridu.shape[1],(2**pad-1)*self.gridu.shape[1]),(0,0)), mode='constant', constant_values=0)
                            padded_vis_vuf = NP.fft.ifftshift(padded_vis_vuf, axes=(1,2))
                            vis_lmf = NP.fft.fft2(padded_vis_vuf, axes=(1,2)) / sum_wts
                            if NP.abs(vis_lmf.imag).max() > 1e-10:
                                raise ValueError('Significant imaginary component found in the synthesized dirty image.')

                            self.nzsp_img_avg[p] = NP.fft.fftshift(vis_lmf.real, axes=(1,2))
                            self.nzsp_grid_vis_avg[p] = vis_vuf
                            self.nzsp_grid_illumination_avg[p] = wts_vuf
                    else:
                        if self.wts_vuf[p] is not None:
                            vis_vuf = NP.copy(self.vis_vuf[p])
                            wts_vuf = NP.copy(self.wts_vuf[p])

                            vis_vuf = vis_vuf - (vis_vuf[self.gridv.shape[0],self.gridu.shape[1],:].reshape(1,1,self.f.size) / autocorr_data_vuf[p][self.gridv.shape[0],self.gridu.shape[1],:].reshape(1,1,self.f.size)) * autocorr_data_vuf[p]
                            wts_vuf = wts_vuf - (wts_vuf[self.gridv.shape[0],self.gridu.shape[1],:].reshape(1,1,self.f.size) / autocorr_wts_vuf[p][self.gridv.shape[0],self.gridu.shape[1],:].reshape(1,1,self.f.size)) * autocorr_wts_vuf[p]
                            sum_wts = NP.sum(wts_vuf, axis=(0,1), keepdims=True)
                            padded_wts_vuf = NP.pad(wts_vuf, (((2**pad-1)*self.gridv.shape[0],(2**pad-1)*self.gridv.shape[0]),((2**pad-1)*self.gridu.shape[1],(2**pad-1)*self.gridu.shape[1]),(0,0)), mode='constant', constant_values=0)
                            padded_wts_vuf = NP.fft.ifftshift(padded_wts_vuf, axes=(0,1))
                            wts_lmf = NP.fft.fft2(padded_wts_vuf, axes=(0,1)) / sum_wts
                            if NP.abs(wts_lmf.imag).max() > 1e-10:
                                raise ValueError('Significant imaginary component found in the synthesized beam.')

                            self.nzsp_beam[p] = NP.fft.fftshift(wts_lmf.real, axes=(0,1))
                            padded_vis_vuf = NP.pad(vis_vuf, (((2**pad-1)*self.gridv.shape[0],(2**pad-1)*self.gridv.shape[0]),((2**pad-1)*self.gridu.shape[1],(2**pad-1)*self.gridu.shape[1]),(0,0)), mode='constant', constant_values=0)
                            padded_vis_vuf = NP.fft.ifftshift(padded_vis_vuf, axes=(0,1))
                            vis_lmf = NP.fft.fft2(padded_vis_vuf, axes=(0,1)) / sum_wts
                            if NP.abs(vis_lmf.imag).max() > 1e-10:
                                raise ValueError('Significant imaginary component found in the synthesized dirty image.')

                            self.nzsp_img[p] = NP.fft.fftshift(vis_lmf.real, axes=(0,1))
                            self.nzsp_wts_vuf[p] = wts_vuf
                            self.nzsp_vis_vuf[p] = vis_vuf

                self.autocorr_removed = True
            else:
                print 'Antenna auto-correlations have been removed already'
            
    ############################################################################

    def evalPowerPatternSkypos(self, skypos, datapool='avg'):

        """
        ------------------------------------------------------------------------
        Evaluate power pattern at the given sky positions for the antenna from 
        its zero-centered cross-correlated footprint

        Input:

        skypos  [numpy array] Positions on sky at which power pattern is to be
                esimated. It is a 2- or 3-column numpy array in direction
                cosine coordinates. It must be of size nsrc x 2 or nsrc x 3.

        datapool
                [string] Specifies whether weights to be used in determining 
                the power pattern come from 'stack', 'current', or 'avg' 
                (default). 

        Output:
        pb_skypos is a dictionary with keys 'P1' and 'P2' for polarization. 
        Under each key is a numpy array of size nsrc x nchan with the esimated
        power pattern spectra at the specified sky locations 
        ------------------------------------------------------------------------
        """

        try:
            skypos
        except NameError:
            raise NameError('Input skypos must be specified')

        if datapool not in ['stack', 'avg', 'recent']:
            raise ValueError('Invalid value specified for input datapool')

        if not isinstance(skypos, NP.ndarray):
            raise TypeError('Input skypos must be a numpy array')

        if skypos.ndim != 2:
            raise ValueError('Input skypos must be a 2D numpy array')

        if (skypos.shape[1] < 2) or (skypos.shape[1] > 3):
            raise ValueError('Input skypos must be a 2- or 3-column array')

        skypos = skypos[:,:2]
        if NP.any(NP.sum(skypos**2, axis=1) > 1.0):
            raise ValueError('Magnitude of skypos direction cosine must not exceed unity')

        self.antenna_array.evalAllAntennaPairCorrWts()
        centered_crosscorr_wts_vuf = self.antenna_array.makeCrossCorrWtsCube()

        du = self.gridu[0,1] - self.gridu[0,0]
        dv = self.gridv[1,0] - self.gridv[0,0]
        gridu, gridv = NP.meshgrid(du*(NP.arange(2*self.gridu.shape[1])-self.gridu.shape[1]), dv*(NP.arange(2*self.gridu.shape[0])-self.gridu.shape[0]))
        griduv = NP.hstack((gridu.reshape(-1,1),gridv.reshape(-1,1)))
        pol = ['P1', 'P2']
        pb_skypos = {}
        for p in pol:
            uvind = SpM.find(centered_crosscorr_wts_vuf[p])[0]
            uniq_uvind = NP.unique(uvind)
            matFT = NP.exp(-1j*2*NP.pi*NP.dot(skypos, griduv[uniq_uvind,:].T))
            uvmeshind, srcmeshind = NP.meshgrid(uniq_uvind, NP.arange(skypos.shape[0]))
            uvmeshind = uvmeshind.ravel()
            srcmeshind = srcmeshind.ravel()
            spFTmat = SpM.csr_matrix((matFT.ravel(), (srcmeshind, uvmeshind)), shape=(skypos.shape[0],griduv.shape[0]), dtype=NP.complex64)
            sum_wts = centered_crosscorr_wts_vuf[p].sum(axis=0)
            pb_skypos[p] = spFTmat.dot(centered_crosscorr_wts_vuf[p]) / sum_wts
            # if NP.abs(pb_skypos[p].imag).max() > 1e-10:
            #     raise ValueError('Significant imaginary component found. Needs debugging')
            pb_skypos[p] = pb_skypos[p].real
        return pb_skypos
        
    ############################################################################
    
    def getStats(self, box_type='square', box_center=None, box_size=None,
                 rms_box_scale_factor=10.0, coords='physical', datapool='avg'):

        """
        ------------------------------------------------------------------------
        Get statistics from images from inside specified boxes
        NEEDS FURTHER DEVELOPMENT !!!

        Inputs:

        box_type    [string] Shape of box. Accepted values are 'square' 
                    (default) and 'circle' on the celestial plane. In 3D the
                    the box will be a cube or cylinder.

        box_center  [list] Center locations of boxes specified as a list one for
                    each box. The centers will have units as specified in input 
                    coords. Each element must be another list, tuple or numpy 
                    array of two or three elements. The first element refers to 
                    the x-coordinate of the box center, the second refers to 
                    y-coordinate of the box center. The third element (optional)
                    refers to the center of frequency around which the 3D box 
                    must be placed. If third element is not specified, it will 
                    be assumed to be center of the band. If coords is set to 
                    'physical', these three elements will have units of dircos,
                    dircos and frequency (Hz). If coords is set to 'index', 
                    these three elements must be indices of the three axes.

        box_size    [list] Sizes of boxes specified as a list one for each box.
                    Number of elements in this list will be equal to that in 
                    input box_center. They will have 'physical' (dircos, 
                    frequency in Hz) or 'index' units as specified in the input 
                    coords. Each element in the list is a one- or two-element 
                    list, tuple or numpy array. The first element is size of the 
                    box in the celestial plane (size of square if box_type is set 
                    to 'square', diameter of circle if box_type is set to 
                    'circle'). The second element (optional) is size along 
                    frequency axis. If second element is not specified, it will 
                    be assumed to be the entire band.
                    
        rms_box_scale_factor
                    [scalar] Size scale on celestial plane used to determine 
                    the box to determine the rms statistic. Must be positive. 
                    For instance, the box size used to find the rms will use a 
                    box that is rms_box_scale_factor times the box size on each
                    side used for determining the peak. Default = 10.0

        coords      [string] String specifying coordinates of box_center and
                    box_size. If set to 'physical' (default) the box_center 
                    will have units of [dircos, dircos, frequency in Hz 
                    (optional)] and box_size will have units of [dircos, 
                    frequency in Hz (optional)]. If set to 'index', box_center 
                    will have units of [index, index, index (optional)] and
                    box_size will have units of [number of pixels, number of 
                    frequency channels].

        datapool    [string] String specifying type of image on which the 
                    statistics will be estimated. Accepted values are 'avg'
                    (default), 'stack' and 'current'. These represent 
                    time-averaged, stacked and recent images respectively

        Outputs:

        outstats    [list] List of dictionaries one for each element in input
                    box_center. Each dictionary consists of the following keys
                    'P1' and 'P2'  for the two polarizations. Under each of 
                    these keys is another dictionary with the following keys and
                    values:
                    'peak-spectrum'
                            [list of numpy arrays] List of Numpy arrays
                            with peak value in each frequency channel. This array
                            is of size nchan. Length of the list is equal to the 
                            number of timestamps as determined by input datapool. 
                            If input datapool is set to 'current', the list will 
                            contain one numpy array of size nchan. If datapool is 
                            set to 'avg' or 'stack', the list will contain n_t
                            number of numpy arrays one for each processed 
                            timestamp
                    'peak-avg'
                            [list] Average of each numpy array in the list under
                            key 'peak-spectrum'. It will have n_t elements where
                            n_t is the number of timestamps as determined by 
                            input datapool
                    'mad'   [list] Median Absolute Deviation(s) in
                            the box determined by input rms_box_scale_factor. 
                            If input datapool is set to 'current', it will be a 
                            one-element list, but if set to 'avg' or 'stack', it 
                            will be a list one for each timestamp in the image
        ------------------------------------------------------------------------
        """

        if box_type not in ['square', 'circle']:
            raise ValueError('Input box_type must be specified as "square" or "circle"')
        if box_center is None:
            raise ValueError('Input box_center must be specified')
        if box_size is None:
            raise ValueError('Input box_size must be specified')

        if coords not in ['physical', 'index']:
            raise ValueError('Input coords must be specified as "physical" or "index"')
        if datapool not in ['avg', 'current', 'stack']:
            raise ValueError('Input datappol must be specified as "avg", "current" or "stack"')
        
        if not isinstance(box_center, list):
            raise TypeError('Input box_center must be a list')
        if not isinstance(box_size, list):
            raise TypeError('Input box_size must be a list')

        if len(box_center) != len(box_size):
            raise ValueError('Lengths of box_center and box_size must be equal')

        if isinstance(rms_box_scale_factor, (int,float)):
            rms_box_scale_factor = float(rms_box_scale_factor)
            if rms_box_scale_factor <= 0.0:
                raise ValueError('Input rms_box_scale_factor must be positive')
        else:
            raise TypeError('Input rms_box_scale_factor must be a scalar')

        bandwidth = (self.f[1] - self.f[0]) * self.f.size
        lfgrid = self.gridl[:,:,NP.newaxis] * NP.ones(self.f.size).reshape(1,1,-1)
        mfgrid = self.gridm[:,:,NP.newaxis] * NP.ones(self.f.size).reshape(1,1,-1)
        fgrid = NP.ones_like(self.gridl)[:,:,NP.newaxis] * self.f.reshape(1,1,-1)
        outstats = []
        for i in xrange(len(box_center)):
            stats = {}
            bc = NP.asarray(box_center[i]).reshape(-1)
            bs = NP.asarray(box_size[i]).reshape(-1)
            if (bc.size < 2) or (bc.size > 3):
                raise ValueError('Each box center must have two or three elements')
            if (bs.size < 1) or (bs.size > 2):
                raise ValueError('Each box size must have one or two elements')
            if bc.size == 2:
                if coords == 'physical':
                    bc = NP.hstack((bc, NP.mean(self.f)))
                else:
                    bc = NP.hstack((bc, self.f.size/2))
            if bs.size == 1:
                if coords == 'physical':
                    bs = NP.hstack((bs, bandwidth))
                else:
                    bs = NP.hstack((bs, self.f.size))
            if coords == 'physical':
                if NP.sum(bc[:2]**2) > 1.0:
                    raise ValueError('Invalid dirction cosines specified')
                if (bc[2] < self.f.min()) or (bc[2] > self.f.max()):
                    raise ValueError('Invalid frequency specified in input box_center')
            else:
                if (bc[0] < 0) or (bc[1] < 0) or (bc[0] > self.gridl.shape[1]) or (bc[1] > self.gridl.shape[0]):
                    raise ValueError('Invalid box center specified')
                if bc[2] > self.f.size:
                    bc[2] = self.f.size
            if coords == 'physical':
                if box_type == 'square':
                    ind3d = (NP.abs(lfgrid - bc[0]) <= 0.5*bs[0]) & (NP.abs(mfgrid - bc[1]) <= 0.5*bs[0]) & (NP.abs(fgrid - bc[2]) <= 0.5*bs[1])
                    ind3d_rmsbox = (NP.abs(lfgrid - bc[0]) <= 0.5*rms_box_scale_factor*bs[0]) & (NP.abs(mfgrid - bc[1]) <= 0.5*rms_box_scale_factor*bs[0]) & (NP.abs(fgrid - bc[2]) <= 0.5*bs[1])
                else:
                    ind3d = (NP.sqrt(NP.abs(lfgrid - bc[0])**2 + NP.abs(mfgrid - bc[0])**2) <= 0.5*bs[0]) & (NP.abs(fgrid - bc[2]) <= 0.5*bs[1])
                    ind3d_rmsbox = (NP.sqrt(NP.abs(lfgrid - bc[0])**2 + NP.abs(mfgrid - bc[0])**2) <= 0.5*rms_box_scale_factor*bs[0]) & (NP.abs(fgrid - bc[2]) <= 0.5*bs[1])
                msk = NP.logical_not(ind3d)
                msk_rms = NP.logical_not(ind3d_rmsbox)
    
                for apol in ['P1', 'P2']:
                    stats[apol] = {'peak-spectrum': [], 'peak-avg': [], 'mad': []}
                    if datapool == 'current':
                        if self.nzsp_img[apol] is not None:
                            img_masked = MA.array(self.nzsp_img[apol], mask=msk)
                            stats[apol]['peak-spectrum'] += [NP.amax(NP.abs(img_masked), axis=(0,1))]
                            stats[apol]['peak-avg'] += [NP.mean(stats[apol]['peak-spectrum'])]
                            img_masked = MA.array(self.nzsp_img[apol], mask=msk_rms)
                            mdn = NP.median(img_masked[~img_masked.mask])
                            absdev = NP.abs(img_masked - mdn)
                            stats[apol]['mad'] += [NP.median(absdev[~absdev.mask])]
                    else:
                        if datapool == 'avg':
                            if self.nzsp_img_avg[apol] is not None:
                                for ti in range(self.nzsp_img_avg[apol].shape[0]):
                                    img_masked = MA.array(self.nzsp_img_avg[apol][ti,...], mask=msk)
                                    stats[apol]['peak-spectrum'] += [NP.amax(NP.abs(img_masked), axis=(0,1))]
                                    stats[apol]['peak-avg'] += [NP.mean(stats[apol]['peak-spectrum'][ti])]
                                    img_masked = MA.array(self.nzsp_img_avg[apol][ti,...], mask=msk_rms)
                                    mdn = NP.median(img_masked[~img_masked.mask])
                                    absdev = NP.abs(img_masked - mdn)
                                    stats[apol]['mad'] += [NP.median(absdev[~absdev.mask])]
                        else:
                            if self.img_stack[apol] is not None:
                                for ti in range(self.img_stack[apol].shape[0]):
                                    img_masked = MA.array(self.img_stack[apol][ti,...], mask=msk)
                                    stats[apol]['peak-spectrum'] += [NP.amax(NP.abs(img_masked), axis=(0,1))]
                                    stats[apol]['peak-avg'] += [NP.mean(stats[apol]['peak-spectrum'][ti])]
                                    img_masked = MA.array(self.img_stack[apol][ti,...], mask=msk_rms)
                                    mdn = NP.median(img_masked[~img_masked.mask])
                                    absdev = NP.abs(img_masked - mdn)
                                    stats[apol]['mad'] += [NP.median(absdev[~absdev.mask])]
                outstats += [stats]
            else:
                pass

        return outstats

    ############################################################################

    def save(self, imgfile, pol=None, overwrite=False, verbose=True):

        """
        ------------------------------------------------------------------------
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
        ------------------------------------------------------------------------
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

################################################################################

class PolInfo:

    """
    ----------------------------------------------------------------------------
    Class to manage polarization information of an antenna. 

    Attributes:

    Et       [dictionary] holds measured complex electric field time series 
             under 2 polarizations which are stored under keys 'P1', and 'P2'

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
                   after doubling the length of the sequence with zero padding 
                   (in order to be identical to what would be obtained from a 
                   XF operation)

    update_flags() Updates the flags based on current inputs and verifies and 
                   updates flags based on current values of the electric field.

    update():      Updates the electric field time series and spectra, and 
                   flags for different polarizations
    
    delay_compensation():
                   Routine to apply delay compensation to Electric field 
                   spectra through additional phase. This assumes that the 
                   spectra have already been made

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
        return ' Instance of class "{0}" in module "{1}" \n flag (P1): {2} \n flag (P2): {3} '.format(self.__class__.__name__, self.__module__, self.flag['P1'], self.flag['P2'])

    ############################################################################ 

    def FT(self, pol=None):

        """
        ------------------------------------------------------------------------
        Perform a Fourier transform of an Electric field time series after 
        doubling the length of the sequence with zero padding (in order to be 
        identical to what would be obtained from a XF operation)

        Keyword Input(s):

        pol     [scalar or list] polarization to be Fourier transformed. Set 
                to 'P1' and/or 'P2'. If None (default) provided, time series 
                of both polarizations are Fourier transformed.
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
        ------------------------------------------------------------------------
        Routine to apply delay compensation to Electric field spectra through
        additional phase. This assumes that the spectra have already been made

        Keyword input(s):

        delaydict   [dictionary] contains one or both polarization keys, namely,
                    'P1' and 'P2'. The value under each of these keys is another 
                    dictionary with the following keys and values:
                    'frequencies': scalar, list or numpy vector specifying the 
                           frequencie(s) (in Hz) for which delays are specified. 
                           If a scalar is specified, the delays are assumed to 
                           be frequency independent and the delays are assumed 
                           to be valid for all frequencies. If a vector is 
                           specified, it must be of same size as the delays and 
                           as the number of samples in the electric field 
                           timeseries. These frequencies are assumed to match 
                           those of the electric field spectrum. No default.
                    'delays': list or numpy vector specifying the delays (in 
                           seconds) at the respective frequencies which are to 
                           be compensated through additional phase in the 
                           electric field spectrum. Must be of same size as 
                           frequencies and the size of the electric field 
                           timeseries. No default.
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

    def update_flags(self, flags=None, verify=False):

        """
        ------------------------------------------------------------------------
        Updates the flags based on current inputs and verifies and updates flags
        based on current values of the electric field.
    
        Inputs:
    
        flags    [dictionary] holds boolean flags for each of the 2 
                 polarizations which are stored under keys 'P1', and 'P2'.
                 Default=None means no new flagging to be applied. If 
                 the value under the polarization key is True, it is to be 
                 flagged and if False, it is to be unflagged.

        verify   [boolean] If True, verify and update the flags, if necessary.
                 Electric fields are checked for NaN values and if found, the
                 flag in the corresponding polarization is set to True. 
                 Default=False. 

        Flag verification and re-updating happens if flags is set to None or if
        verify is set to True.
        ------------------------------------------------------------------------
        """

        # if not isinstance(stack, bool):
        #     raise TypeError('Input keyword stack must be of boolean type')

        if not isinstance(verify, bool):
            raise TypeError('Input keyword verify must be of boolean type')

        if flags is not None:
            if not isinstance(flags, dict):
                raise TypeError('Input parameter flags must be a dictionary')
            for pol in ['P1', 'P2']:
                if pol in flags:
                    if isinstance(flags[pol], bool):
                        self.flag[pol] = flags[pol]
                    else:
                        raise TypeError('flag values must be boolean')

        # Perform flag verification and re-update current flags
        if verify or (flags is None):
            for pol in ['P1', 'P2']:
                if NP.any(NP.isnan(self.Et[pol])):
                    self.flag[pol] = True

    ############################################################################ 

    def update(self, Et=None, Ef=None, flags=None, delaydict=None,
               verify=False):
        
        """
        ------------------------------------------------------------------------
        Updates the electric field time series and spectra, and flags for 
        different polarizations

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

        verify [boolean] If True, verify and update the flags, if necessary.
               Electric fields are checked for NaN values and if found, the
               flag in the corresponding polarization is set to True. 
               Default=False. 
        ------------------------------------------------------------------------
        """

        current_flags = copy.deepcopy(self.flag)
        if flags is None:
            flags = copy.deepcopy(current_flags)
        # if flags is not None:
        #     self.update_flags(flags)
            
        if Et is not None:
            if isinstance(Et, dict):
                for pol in ['P1', 'P2']:
                    if pol in Et:
                        self.Et[pol] = Et[pol]
                        if NP.any(NP.isnan(Et[pol])):
                            # self.Et[pol] = NP.nan
                            flags[pol] = True
                            # self.flag[pol] = True
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
                            flags[pol] = True
                            # self.flag[pol] = True
            else:
                raise TypeError('Input parameter Ef must be a dictionary')

        if delaydict is not None:
            self.delay_compensation(delaydict)

        # Verify and update flags
        self.update_flags(flags=flags, verify=verify)
            
################################################################################

class Antenna:

    """
    ----------------------------------------------------------------------------
    Class to manage individual antenna information.

    Attributes:

    label:      [Scalar] A unique identifier (preferably a string) for the 
                antenna. 

    typetag     [scalar or string] Tag (integer or string) to identify antenna
                type. Will be used in determining if the antenna array is made
                of identical antennas or not

    latitude:   [Scalar] Latitude of the antenna's location.

    longitude:  [Scalar] Longitude of the antenna's location.

    location:   [Instance of GEOM.Point class] The location of the antenna in 
                local East, North, Up coordinate system.

    timestamp:  [Scalar] String or float representing the timestamp for the 
                current attributes

    timestamps  [list] list of all timestamps to be held in the stack 

    t:          [vector] The time axis for the time series of electric fields

    f:          [vector] Frequency axis obtained by a Fourier Transform of
                the electric field time series. Same length as attribute t 

    f0:         [Scalar] Center frequency in Hz.

    antpol:     [Instance of class PolInfo] polarization information for the 
                antenna. Read docstring of class PolInfo for details

    aperture    [Instance of class APR.Aperture] aperture information
                for the antenna. Read docstring of class Aperture for
                details

    Et_stack    [dictionary] holds a stack of complex electric field time series 
                measured at various time stamps under 2 polarizations which are 
                stored under keys 'P1' and 'P2'
                
    Ef_stack    [dictionary] holds a stack of complex electric field spectra 
                measured at various time stamps under 2 polarizations which are 
                stored under keys 'P1' and 'P2'

    flag_stack
                [dictionary] holds a stack of flags appropriate for different 
                time stamps as a numpy array under 2 polarizations which are 
                stored under keys 'P1' and 'P2'

    wts:        [dictionary] The gridding weights for antenna. Different 
                polarizations 'P1' and 'P2' form the keys 
                of this dictionary. These values are in general complex. Under 
                each key, the values are maintained as a list of numpy vectors, 
                where each vector corresponds to a frequency channel. See 
                wtspos_scale for more requirements.

    wtspos      [dictionary] two-dimensional locations of the gridding weights 
                in wts for each polarization under keys 'P1' and 'P2'. The 
                locations are in ENU coordinate system as a list of 2-column 
                numpy arrays. Each 2-column array in the list is the position 
                of the gridding weights for a corresponding frequency 
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

    FT()         Computes the Fourier transform of the time series of the 
                 antennas in the antenna array to compute the visibility 
                 spectra. Read docstring of member function FT() of class 
                 PolInfo

    FT_pp()     Computes the Fourier transform of the time series of the 
                 antennas in the antenna array to compute the visibility 
                 spectra. Read docstring of member function FT() of class 
                 PolInfo. Differs from FT() member function in that here 
                 an instance of class Antenna is returned and is mainly used 
                 in case of parallel processing and is not meant to be 
                 accessed directly by the user. Use FT() for all other pruposes.

    update_flags()
                 Updates flags for polarizations provided as input parameters

    update():    Updates the antenna instance with newer attribute values
                 Updates the electric field spectrum and timeseries. It also
                 applies Fourier transform if timeseries is updated

    update_pp()  Wrapper for member function update() and returns the updated 
                 instance of this class. Mostly intended to be used when 
                 parallel processing is applicable and not to be used directly.
                 Use update() instead when updates are to be applied directly.

    get_E_fields() 
                 Returns the electric fields based on selection criteria on 
                 timestamp flags, timestamps and frequency channel indices and 
                 the type of data (most recent or stacked electric fields)

    evalGridIllumination()
                 Evaluate antenna illumination function on a specified grid

    save():      Saves the antenna information to disk. Needs serious 
                 development. 

    Read the member function docstrings for details.
    ----------------------------------------------------------------------------
    """

    def __init__(self, label, typetag, latitude, longitude, location, 
                 center_freq, nsamples=1, aperture=None):

        """
        ------------------------------------------------------------------------
        Initialize the Antenna Class which manages an antenna's information 

        Class attributes initialized are:
        label, latitude, longitude, location, pol, t, timestamp, f0, f, wts, 
        wtspos, wtspos_scale, blc, trc, timestamps, antpol, Et_stack, Ef_stack, 
        flag_stack, aperture, typetag
     
        Read docstring of class Antenna for details on these attributes.
        ------------------------------------------------------------------------
        """

        try:
            label
        except NameError:
            raise NameError('Antenna label must be provided.')

        try:
            typetag
        except NameError:
            raise NameError('Antenna type tag must be provided.')

        if not isinstance(typetag, (int,str)):
            raise TypeError('Antenna type tag must be an integer or string')

        try:
            latitude
        except NameError:
            latitude = 0.0

        try:
            longitude
        except NameError:
            longitude = 0.0

        try:
            location
        except NameError:
            self.location = GEOM.Point()

        try:
            center_freq
        except NameError:
            raise NameError('Center frequency must be provided.')

        self.label = label
        self.typetag = typetag
        self.latitude = latitude
        self.longitude = longitude

        if isinstance(location, GEOM.Point):
            self.location = location
        elif isinstance(location, (list, tuple, NP.ndarray)):
            self.location = GEOM.Point(location)
        else:
            raise TypeError('Antenna position must be a 3-element tuple or an instance of GEOM.Point')

        if aperture is not None:
            if isinstance(aperture, APR.Aperture):
                if len(aperture.pol) != 2:
                    raise ValueError('Antenna aperture must contain dual polarization types')
                self.aperture = aperture
            else:
                raise TypeError('aperture must be an instance of class Aperture found in module {0}'.format(APR.__name__))
        else:
            self.aperture = APR.Aperture(pol_type='dual')

        self.antpol = PolInfo(nsamples=nsamples)
        self.t = 0.0
        self.timestamp = 0.0
        self.timestamps = []
        self.f0 = center_freq
        self.f = self.f0

        self.Et_stack = {}
        self.Ef_stack = {}
        self.flag_stack = {} 

        self.wts = {}
        self.wtspos = {}
        self.wtspos_scale = {}
        self._gridinfo = {}

        for pol in ['P1', 'P2']:
            self.Et_stack[pol] = None
            self.Ef_stack[pol] = None
            self.flag_stack[pol] = NP.asarray([])

            self.wtspos[pol] = []
            self.wts[pol] = []
            self.wtspos_scale[pol] = None
            self._gridinfo[pol] = {}
        
        self.blc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1)
        self.trc = NP.asarray([self.location.x, self.location.y]).reshape(1,-1)

    ############################################################################

    def __str__(self):
        return ' Instance of class "{0}" in module "{1}" \n label: {2} \n typetag: {3} \n location: {4}'.format(self.__class__.__name__, self.__module__, self.label, self.typetag, self.location.__str__())

    ############################################################################

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

    ############################################################################

    def FT(self, pol=None):

        """
        ------------------------------------------------------------------------
        Computes the Fourier transform of the time series of the antennas in the 
        antenna array to compute the visibility spectra. Read docstring of 
        member function FT() of class PolInfo

        Inputs:

        pol    [scalar or list] Scalar string or list of strings specifying 
               polarization. Accepted values are 'P1' and/or 'P2'. Default=None 
               means both time series of electric fields of both polarizations 
               are Fourier transformed

        # stack  [boolean] If set to True, perform Fourier transform on the 
        #        timestamp-stacked electric field time series. Default = False
        ------------------------------------------------------------------------
        """
        
        self.antpol.FT(pol=pol)
        
    ############################################################################

    def FT_pp(self, pol=None):

        """
        ------------------------------------------------------------------------
        Computes the Fourier transform of the time series of the antennas in the 
        antenna array to compute the visibility spectra. Read docstring of 
        member function FT() of class PolInfo. Differs from FT() member function 
        in that here an instance of class Antenna is returned and is mainly used 
        in case of parallel processing and is not meant to be accessed directly 
        by the user. Use FT() for all other pruposes.

        Inputs:

        pol    [scalar or list] Scalar string or list of strings specifying 
               polarization. Accepted values are 'P1' and/or 'P2'. Default=None 
               means both time series of electric fields of both polarizations 
               are Fourier transformed

        # stack  [boolean] If set to True, perform Fourier transform on the 
        #        timestamp-stacked electric field time series. Default = False

        Outputs:

        Instance of class Antenna
        ------------------------------------------------------------------------
        """
        
        self.antpol.FT(pol=pol)
        return self
        
    ############################################################################

    def update_flags(self, flags=None, stack=False, verify=True):

        """
        ------------------------------------------------------------------------
        Updates flags for antenna polarizations. Invokes member function 
        update_flags() of class PolInfo

        Inputs:

        flags  [dictionary] boolean flags for each of the 2 polarizations 
               of the antenna which are stored under keys 'P1' and 'P2',
               Default=None means no updates for flags.

        stack  [boolean] If True (default), appends the updated flag to the
               end of the stack of flags as a function of timestamp. If False,
               updates the last flag in the stack with the updated flag and 
               does not append

        verify [boolean] If True, verify and update the flags, if necessary.
               Electric fields are checked for NaN values and if found, the
               flag in the corresponding polarization is set to True. 
               Default=True 
        ------------------------------------------------------------------------
        """

        # By default carry over the flags from previous timestamp

        if flags is None:
            flags = copy.deepcopy(self.antpol.flag)

        self.antpol.update_flags(flags=flags, verify=verify)

        # Stack on to last value or update last value in stack
        for pol in ['P1', 'P2']: 
            if stack is True:
                self.flag_stack[pol] = NP.append(self.flag_stack[pol], self.antpol.flag[pol])
            else:
                if self.flag_stack[pol].size == 0:
                    self.flag_stack[pol] = NP.asarray(self.antpol.flag[pol]).reshape(-1)
                else:
                    self.flag_stack[pol][-1] = self.antpol.flag[pol]
            self.flag_stack[pol] = self.flag_stack[pol].astype(NP.bool)

    ############################################################################

    def update(self, update_dict=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Updates the antenna instance with newer attribute values. Updates 
        the electric field spectrum and timeseries. It also applies Fourier 
        transform if timeseries is updated

        Inputs:

        update_dict [dictionary] contains the following keys and values:

            label      [Scalar] A unique identifier (preferably a string) for 
                       the antenna. Default=None means no update to apply
    
            typetag    [scalar or string] Antenna type identifier (integer or
                       preferably string) which will be used in determining if
                       all antennas in the antenna array are identical

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
    
            aperture   [instance of class APR.Aperture] aperture 
                       information for the antenna. Read docstring of class 
                       Aperture for details

            wtsinfo    [dictionary] consists of weights information for each of 
                       the two polarizations under keys 'P1' and 'P2'. Each of 
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

            stack      [boolean] If True (default), appends the updated flag 
                       and data to the end of the stack as a function of 
                       timestamp. If False, updates the last flag and data in 
                       the stack and does not append

            verify     [boolean] If True, verify and update the flags, if 
                       necessary. Electric fields are checked for NaN values and 
                       if found, the flag in the corresponding polarization is 
                       set to True. Default=True 

        verbose    [boolean] If True, prints diagnostic and progress messages. 
                   If False (default), suppress printing such messages.
        ------------------------------------------------------------------------
        """

        label = None
        typetag = None
        location = None
        timestamp = None
        t = None
        flags = None
        stack = False
        verify_flags = True
        Et = None
        wtsinfo = None
        gridfunc_freq = None
        ref_freq = None
        delaydict = None
        aperture = None
            
        if update_dict is not None:
            if not isinstance(update_dict, dict):
                raise TypeError('Input parameter containing updates must be a dictionary')

            if 'label' in update_dict: label = update_dict['label']
            if 'typetag' in update_dict: typetag = update_dict['typetag']
            if 'location' in update_dict: location = update_dict['location']
            if 'timestamp' in update_dict: timestamp = update_dict['timestamp']
            if 't' in update_dict: t = update_dict['t']
            if 'Et' in update_dict: Et = update_dict['Et']
            if 'flags' in update_dict: flags = update_dict['flags']
            if 'stack' in update_dict: stack = update_dict['stack']
            if 'verify_flags' in update_dict: verify_flags = update_dict['verify_flags']            
            if 'wtsinfo' in update_dict: wtsinfo = update_dict['wtsinfo']
            if 'gridfunc_freq' in update_dict: gridfunc_freq = update_dict['gridfunc_freq']
            if 'ref_freq' in update_dict: ref_freq = update_dict['ref_freq']
            if 'delaydict' in update_dict: delaydict = update_dict['delaydict']
            if 'aperture' in update_dict: aperture = update_dict['aperture']

        if label is not None: self.label = label
        if typetag is not None: self.typetag = typetag
        if location is not None: self.location = location
        if timestamp is not None:
            self.timestamp = timestamp
            self.timestamps += [copy.deepcopy(timestamp)]

        if t is not None:
            self.t = t
            self.f = self.f0 + self.channels()     

        # Updates, Et, Ef, delays, flags and verifies flags
        if (Et is not None) or (delaydict is not None) or (flags is not None):
            self.antpol.update(Et=Et, delaydict=delaydict, flags=flags, verify=verify_flags) 

        # Stack flags and data
        self.update_flags(flags=None, stack=stack, verify=True)  
        for pol in ['P1', 'P2']:
            if self.Et_stack[pol] is None:
                self.Et_stack[pol] = copy.deepcopy(self.antpol.Et[pol].reshape(1,-1))
                self.Ef_stack[pol] = copy.deepcopy(self.antpol.Ef[pol].reshape(1,-1))
            else:
                if stack:
                    self.Et_stack[pol] = NP.vstack((self.Et_stack[pol], self.antpol.Et[pol].reshape(1,-1)))
                    self.Ef_stack[pol] = NP.vstack((self.Ef_stack[pol], self.antpol.Ef[pol].reshape(1,-1)))
                else:
                    self.Et_stack[pol][-1,:] = copy.deepcopy(self.antpol.Et[pol].reshape(1,-1))
                    self.Ef_stack[pol][-1,:] = copy.deepcopy(self.antpol.Ef[pol].reshape(1,-1))
        
        blc_orig = NP.copy(self.blc)
        trc_orig = NP.copy(self.trc)
        eps = 1e-6

        if aperture is not None:
            if isinstance(aperture, APR.Aperture):
                self.aperture = copy.deepcopy(aperture)
            else:
                raise TypeError('Update for aperture must be an instance of class Aperture.')

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

    def update_pp(self, update_dict=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Wrapper for member function update() and returns the updated instance 
        of this class. Mostly intended to be used when parallel processing is 
        applicable and not to be used directly. Use update() instead when 
        updates are to be applied directly.

        See member function update() for details on inputs.
        ------------------------------------------------------------------------
        """

        self.update(update_dict=update_dict, verbose=verbose)
        return self

    ############################################################################

    def get_E_fields(self, pol, flag=None, tselect=None, fselect=None,
                     datapool=None):

        """
        ------------------------------------------------------------------------
        Returns the electric fields based on selection criteria on timestamp 
        flags, timestamps and frequency channel indices and the type of data
        (most recent or stacked electric fields)

        Inputs:

        pol      [string] select baselines of this polarization that are either 
                 flagged or unflagged as specified by input parameter flag. 
                 Allowed values are 'P1' and 'P2'. Only one of these values 
                 must be specified.

        flag     [boolean] If False, return electric fields of unflagged 
                 timestamps, or if True return flagged ones. Default=None means 
                 all electric fields independent of flagging are returned. This 
                 flagging refers to that along the timestamp axis under each
                 polarization
 
        tselect  [scalar, list, numpy array] timestamp index for electric 
                 fields selection. For most recent electric fields, it must be 
                 set to -1. For all other selections, indices in tselect must 
                 be in the valid range of indices along time axis for stacked 
                 electric fields. Default=None means most recent data is 
                 selected. 

        fselect  [scalar, list, numpy array] frequency channel index for 
                 electric field spectrum selection. Indices must be in the 
                 valid range of indices along the frequency axis for 
                 electric fields. Default=None selects all frequency channels

        datapool [string] denotes the data pool from which electric fields are 
                 to be selected. Accepted values are 'current', 'stack', and
                 None (default, same as 'current'). If set to None or 
                 'current', the value in tselect is ignored and only 
                 electric fields of the most recent timestamp are selected. If 
                 set to None or 'current' the attribute Ef_stack is checked 
                 first and if unavailable, attribute antpol.Ef is used. For 
                 'stack', attribute Ef_stack respectively

        Output:

        outdict  [dictionary] consists of electric fields information under the 
                 following keys:

                 'label'        [string] antenna label 
                 'pol'          [string] polarization string, one of 'P1' or 
                                'P2'
                 'E-fields'     [numpy array] selected electric fields spectra
                                with dimensions n_ts x nchan which
                                are in time-frequency order. If no electric 
                                fields are found satisfying the selection 
                                criteria, the value under this key is set to 
                                None.
                 'twts'         [numpy array of boolean] weights corresponding 
                                to the time axis in the selected electric 
                                fields. A zero weight indicates unflagged 
                                electric fields were not found for that 
                                timestamp. A non-zero weight indicates how 
                                many unflagged electric fields were found for 
                                that timestamp. If no electric fields are found
                                satisfying the selection criteria, the value 
                                under this key is set to None.
        ------------------------------------------------------------------------
        """

        try: 
            pol 
        except NameError:
            raise NameError('Input parameter pol must be specified.')

        if not isinstance(pol, str):
            raise TypeError('Input parameter must be a string')
        
        if not pol in ['P1', 'P2']:
            raise ValueError('Invalid specification for input parameter pol')

        if datapool is None:
            n_timestamps = 1
            datapool = 'current'
        elif datapool == 'stack':
            n_timestamps = len(self.timestamps)
        elif datapool == 'current':
            n_timestamps = 1
        else:
            raise ValueError('Invalid datapool specified')

        if tselect is None:
            tsind = NP.asarray(-1).reshape(-1)  # Selects most recent data
        elif isinstance(tselect, (int, float, list, NP.ndarray)):
            tsind = NP.asarray(tselect).ravel()
            tsind = tsind.astype(NP.int)
            if tsind.size == 1:
                if (tsind < -1) or (tsind >= n_timestamps):
                    tsind = NP.asarray(-1).reshape(-1)
            else:
                if NP.any(tsind < 0) or NP.any(tsind >= n_timestamps):
                    raise IndexError('Timestamp indices outside available range for the specified datapool')
        else:
            raise TypeError('tselect must be None, integer, float, list or numpy array for visibilities selection')

        if fselect is None:
            chans = NP.arange(self.f.size)  # Selects all channels
        elif isinstance(fselect, (int, float, list, NP.ndarray)):
            chans = NP.asarray(fselect).ravel()
            chans = chans.astype(NP.int)
            if NP.any(chans < 0) or NP.any(chans >= self.f.size):
                raise IndexError('Channel indices outside available range')
        else:
            raise TypeError('fselect must be None, integer, float, list or numpy array for visibilities selection')

        select_ind = NP.ix_(tsind, chans)

        outdict = {}
        outdict['pol'] = pol
        outdict['twts'] = None
        outdict['label'] = self.label
        outdict['E-fields'] = None
        
        if datapool == 'current':
            if self.Ef_stack[pol] is not None:
                outdict['E-fields'] = self.Ef_stack[pol][-1,chans].reshape(1,chans.size)
                outdict['twts'] = NP.logical_not(NP.asarray(self.flag_stack[pol][-1]).astype(NP.bool).reshape(-1)).astype(NP.float)
            else:
                outdict['E-fields'] = self.antpol.Ef[pol][chans].reshape(1,chans.size)
                outdict['twts'] = NP.logical_not(NP.asarray(self.antpol.flag[pol]).astype(NP.bool).reshape(-1)).astype(NP.float)
        else:
            if self.Ef_stack[pol] is not None:
                outdict['E-fields'] = self.Ef_stack[pol][select_ind].reshape(tsind.size,chans.size)
                outdict['twts'] = NP.logical_not(NP.asarray(self.flag_stack[pol][tsind]).astype(NP.bool).reshape(-1)).astype(NP.float)
            else:
                raise ValueError('Attribute Ef_stack has not been initialized to obtain electric fields from. Consider running method stack()')

        return outdict

    ############################################################################

    def evalGridIllumination(self, uvlocs=None, xy_center=None):

        """
        ------------------------------------------------------------------------
        Evaluate antenna illumination function on a specified grid

        Inputs:

        uvlocs      [tuple] 2-element tuple where first and second elements
                    are numpy arrays that contain u- and v-locations 
                    respectively. Default=None means determine u- and v-
                    locations from attributes blc and trc

        xy_center   [tuple, list or numpy array] 2-element list, tuple or numpy
                    array denoting x- and y-locations of center of antenna.
                    Default=None means use the x- and y-locations of the 
                    antenna

        Outputs:

        antenna_grid_wts_vuf
                    [scipy sparse array] Complex antenna illumination weights 
                    placed on the specified grid. When expanded it will be of 
                    size nv x nu x nchan
        ------------------------------------------------------------------------
        """

        if xy_center is None:
            xy_center = NP.asarray([self.location.x, self.location.y])
        elif isinstance(xy_center, (list,tuple,NP.ndarray)):
            xy_center = NP.asarray(xy_center)
            if xy_center.size != 2:
                raise ValueError('Input xy_center must be a two-element numpy array')
            xy_center = xy_center.ravel()
        else:
             raise TypeError('Input xy_center must be a numpy array')

        wavelength = FCNST.c / self.f
        min_wl = NP.abs(wavelength).min()
        uvspacing = 0.5
        if uvlocs is None:
            blc = self.blc - xy_center
            trc = self.trc - xy_center
            trc = NP.amax(NP.abs(NP.vstack((blc, trc))), axis=0).ravel() / min_wl
            blc = -1 * trc
            gridu, gridv = GRD.grid_2d([(blc[0], trc[0]), (blc[1], trc[1])], pad=0.0, spacing=uvspacing, pow2=True)
            du = gridu[0,1] - gridu[0,0]
            dv = gridv[1,0] - gridv[0,0]
        elif isinstance(uvlocs, tuple):
            if len(uvlocs) != 2:
                raise ValueError('Input uvlocs must be a two-element tuple')
            ulocs, vlocs = uvlocs
            if not isinstance(ulocs, NP.ndarray):
                raise TypeError('Elements in input tuple uvlocs must be a numpy array')
            if not isinstance(vlocs, NP.ndarray):
                raise TypeError('Elements in input tuple uvlocs must be a numpy array')
            ulocs = ulocs.ravel()
            vlocs = vlocs.ravel()
            du = ulocs[1] - ulocs[0]
            dv = vlocs[1] - vlocs[0]
            gridu, gridv = NP.meshgrid(ulocs, vlocs)
        else:
            raise TypeError('Input uvlocs must be a two-element tuple')

        rmaxNN = 0.5 * NP.sqrt(du**2 + dv**2) * min_wl
        gridx = gridu[:,:,NP.newaxis] * wavelength.reshape(1,1,-1)
        gridy = gridv[:,:,NP.newaxis] * wavelength.reshape(1,1,-1)
        gridxy = NP.hstack((gridx.reshape(-1,1), gridy.reshape(-1,1)))
        wl = NP.ones(gridu.shape)[:,:,NP.newaxis] * wavelength.reshape(1,1,-1)
        max_aprtr_size = max([NP.sqrt(self.aperture.xmax['P1']**2 + NP.sqrt(self.aperture.ymax['P1']**2)), NP.sqrt(self.aperture.xmax['P2']**2 + NP.sqrt(self.aperture.ymax['P2']**2)), self.aperture.rmax['P1'], self.aperture.rmax['P2']])
        distNN = 2.0 * max_aprtr_size
        indNN_list, blind, vuf_gridind = LKP.find_NN(xy_center.reshape(1,-1), gridxy, distance_ULIM=distNN, flatten=True, parallel=False)
        dxy = gridxy[vuf_gridind,:]
        unraveled_vuf_ind = NP.unravel_index(vuf_gridind, gridu.shape+(self.f.size,))
        unraveled_vu_ind = (unraveled_vuf_ind[0], unraveled_vuf_ind[1])
        raveled_vu_ind = NP.ravel_multi_index(unraveled_vu_ind, (gridu.shape[0], gridu.shape[1]))

        antenna_grid_wts_vuf = {}
        pol = ['P1', 'P2']
        for p in pol:
            krn = self.aperture.compute(dxy, wavelength=wl.ravel()[vuf_gridind], pol=p, rmaxNN=rmaxNN, load_lookup=False)
            krn_sparse = SpM.csr_matrix((krn[p], (raveled_vu_ind,)+(unraveled_vuf_ind[2],)), shape=(gridu.size,)+(self.f.size,), dtype=NP.complex64)
            krn_sparse_sumuv = krn_sparse.sum(axis=0)
            krn_sparse_norm = krn_sparse.A / krn_sparse_sumuv.A
            sprow = raveled_vu_ind
            spcol = unraveled_vuf_ind[2]
            spval = krn_sparse_norm[(sprow,)+(spcol,)]
            antenna_grid_wts_vuf[p] = SpM.csr_matrix((spval, (sprow,)+(spcol,)), shape=(gridu.size,)+(self.f.size,), dtype=NP.complex64)
    
        return antenna_grid_wts_vuf

################################################################################

class AntennaArray:

    """
    ----------------------------------------------------------------------------
    Class to manage collective information on a group of antennas.

    Attributes:

    antennas:    [Dictionary] Dictionary consisting of keys which hold instances
                 of class Antenna. The keys themselves are identical to the
                 label attributes of the antenna instances they hold.

    latitude     [Scalar] Latitude of the antenna array location.

    longitude    [Scalar] Longitude of the antenna array location.

    blc          [2-element Numpy array] The coordinates of the bottom left 
                 corner of the array of antennas

    trc          [2-element Numpy array] The coordinates of the top right 
                 corner of the array of antennas

    grid_blc     [2-element Numpy array] The coordinates of the bottom left 
                 corner of the grid constructed for the array of antennas.
                 This may differ from blc due to any extra padding during the 
                 gridding process.

    grid_trc     [2-element Numpy array] The coordinates of the top right 
                 corner of the grid constructed for the array of antennas
                 This may differ from trc due to any extra padding during the 
                 gridding process.

    grid_ready   [boolean] True if the grid has been created, False otherwise

    gridu        [Numpy array] u-locations of the grid lattice stored as 2D 
                 array. It is the same for all frequencies and hence no third 
                 dimension for the spectral axis.

    gridv        [Numpy array] v-locations of the grid lattice stored as 2D 
                 array. It is the same for all frequencies and hence no third 
                 dimension for the spectral axis.

    antenna_autocorr_set
                 [boolean] Indicates if auto-correlation of antenna-wise weights
                 have been determined (True) or not (False).

    antenna_crosswts_set
                 [boolean] Indicates if zero-centered cross-correlation of 
                 antenna pair weights have been determined (True) or not (False)

    auto_corr_data
                 [dictionary] holds antenna auto-correlation of complex electric 
                 field spectra. It is under keys 'current', 'stack' and 'avg' 
                 for the current, stacked and time-averaged auto-correlations. 
                 Under eack of these keys is another dictionary with two keys
                 'P1' and 'P2' for the two polarizations. Under each of these
                 polarization keys is a dictionary with the following keys
                 and values:
                 'labels'   [list of strings] Contains a list of antenna 
                            labels
                 'E-fields' [numpy array] Contains time-averaged 
                            auto-correlation of antenna electric fields. It is
                            of size n_tavg x nant x nchan
                 'twts'     [numpy array] Contains number of unflagged electric
                            field spectra used in the averaging of antenna
                            auto-correlation spectra. It is of size 
                            n_tavg x nant x 1

    pairwise_typetag_crosswts_vuf
                 [dictionary] holds grid illumination wts (centered on grid 
                 origin) obtained from cross-correlation of antenna pairs that
                 belong to their respective typetags. Tuples of typetag pairs
                 form the keys. Under each key is another dictionary with two
                 keys 'P1' and 'P2' for each polarization. Under each of these
                 polarization keys is a complex numpy array of size 
                 nv x nu x nchan. It is obtained by correlating the aperture
                 illumination weights of one antenna type with the complex
                 conjugate of another.

    antennas_center
                 [Numpy array] geometrical center of the antenna array locations
                 as a 2-element array of x- and y-values of the center. This is
                 not the center of mass of the antenna locations but simply the 
                 mid-point between the extreme x- and y- coordinates of the 
                 antennas

    grid_illumination
                 [dictionary] Electric field illumination of antenna aperture
                 for each polarization held under keys 'P1' and 'P2'. Could be 
                 complex. Stored as numpy arrays in the form of cubes with 
                 same dimensions as gridu or gridv in the transverse (first two
                 dimensions) and the depth along the third dimension (spectral 
                 axis) is equal to number of frequency channels

    grid_Ef      [dictionary] Complex Electric field projected on the grid
                 for each polarization under the keys 'P1' and P2'. Stored as
                 numpy arrays in the form of cubes with same dimensions as gridu 
                 or gridv in the transverse (first two dimensions) and the depth 
                 along the third dimension (spectral axis) is equal to number of 
                 frequency channels. 

    f            [Numpy array] Frequency channels (in Hz)

    f0           [Scalar] Center frequency of the observing band (in Hz)

    typetags     [dictionary] Dictionary containing keys which are unique 
                 antenna type tags. Under each of these type tag keys is a 
                 set of antenna labels denoting antennas that are of that type

    pairwise_typetags     
                 [dictionary] Dictionary containing keys which are unique 
                 pairwise combination (tuples) of antenna type tags. Under each 
                 of these pairwise type tag keys is a dictionary with two keys 
                 'auto' and 'cross' each of which contains a set of pairwise 
                 (tuple) antenna labels denoting the antenna pairs that are of 
                 that type. Under 'auto' are tuples with same antennas while
                 under 'cross' it contains antenna pairs in which the antennas 
                 are not the same. The 'auto' key exists only when antenna
                 type tag tuple contains both antennas of same type. 

    antenna_pair_to_typetag
                 [dictionary] Dictionary containing antenna pair keys and the
                 corresponding values are typetag pairs.

    timestamp:   [Scalar] String or float representing the timestamp for the 
                 current attributes

    timestamps   [list] list of all timestamps to be held in the stack 

    tbinsize     [scalar or dictionary] Contains bin size of timestamps while
                 averaging after stacking. Default = None means all antenna 
                 E-field auto-correlation spectra over all timestamps are 
                 averaged. If scalar, the same (positive) value applies to all 
                 polarizations. If dictionary, timestamp bin size (positive) in 
                 seconds is provided under each key 'P1' and 'P2'. If any of 
                 the keys is missing the auto-correlated antenna E-field spectra 
                 for that polarization are averaged over all timestamps.

    grid_mapper [dictionary] antenna-to-grid mapping information for each of
                four polarizations under keys 'P1' and 'P2'. Under each
                polarization, it is a dictionary with values under the following 
                keys:
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
                'refwts'    [numpy array] antenna weights of size 
                            n_ant x n_wts flattened to be a vector. Indices in
                            'refind' index to this array. Currently only valid
                            when lookup weights scale with frequency.
                'labels'    [dictionary] contains mapping information from 
                            antenna (specified by key which is the 
                            antenna label). The value under each label 
                            key is another dictionary with the following keys 
                            and information:
                            'twts'         [scalar] if positive, indicates
                                           the number of timestamps that 
                                           have gone into the measurement of Ef 
                                           made by the antenna under the 
                                           specific polarization. If zero, it
                                           indicates no unflagged timestamp data
                                           was found for the antenna and will
                                           not contribute to the complex grid 
                                           illumination and electric fields
                            'gridind'      [numpy vector] one-dimensional index 
                                           into the three-dimensional grid 
                                           locations where the antenna
                                           contributes illumination and 
                                           electric fields. The one-dimensional 
                                           indices are obtained using numpy's
                                           multi_ravel_index() using the grid 
                                           shape, n_u x n_v x nchan
                            'illumination' [numpy vector] complex grid 
                                           illumination contributed by the 
                                           antenna to different grid
                                           locations in 'gridind'. It is 
                                           mapped to the grid as specified by 
                                           indices in key 'gridind'
                            'Ef'           [numpy vector] complex grid 
                                           electric fields contributed by the 
                                           antenna. It is mapped to the
                                           grid as specified by indices in 
                                           key 'gridind'
                'ant'       [dictionary] dictionary with information on 
                            contribution of all antenna lookup weights. This
                            contains another dictionary with the following 
                            keys:
                            'ind_freq'     [list] each element in the list is
                                           for a frequency channel and 
                                           consists of a numpy vector which 
                                           consists of indices of the 
                                           contributing antennas
                            'ind_all'      [numpy vector] consists of numpy 
                                           vector which consists of indices 
                                           of the contributing antennas
                                           for all frequencies appended 
                                           together. Effectively, this is just
                                           values in 'ind_freq' of all 
                                           frequencies appended together.
                            'uniq_ind_all' [numpy vector] consists of numpy
                                           vector which consists of unique 
                                           indices of contributing antennas
                                           for all frequencies.
                            'rev_ind_all'  [numpy vector] reverse indices of 
                                           'ind_all' with reference to bins of
                                           'uniq_ind_all'
                            'illumination' [numpy vector] complex grid
                                           illumination weights contributed by
                                           each antenna (including associated
                                           kernel weight locations) and has a
                                           size equal to that in 'ind_all'
                'grid'      [dictionary] contains information about populated
                            portions of the grid. It consists of values in the
                            following keys:
                            'ind_all'      [numpy vector] indices of all grid
                                           locations raveled to one dimension
                                           from three dimensions of size 
                                           n_u x n_v x nchan
                'per_ant2grid'
                            [list] each element in the list is a dictionary
                            corresponding to an antenna with information on
                            its mapping and contribution to the grid. Each 
                            dictionary has the following keys and values:
                            'label'        [string] antenna label
                            'f_gridind'    [numpy array] mapping information 
                                           with indices to the frequency axis
                                           of the grid
                            'u_gridind'    [numpy array] mapping information 
                                           with indices to the u-axis
                                           of the grid. Must be of same size 
                                           as array under 'f_gridind'
                            'v_gridind'    [numpy array] mapping information 
                                           with indices to the v-axis
                                           of the grid. Must be of same size 
                                           as array under 'f_gridind'
                            'per_ant_per_freq_norm_wts'
                                           [numpy array] mapping information 
                                           on the (complex) normalizing 
                                           multiplicative factor required to 
                                           make the sum of illumination/weights 
                                           per antenna per frequency on the 
                                           grid equal to unity. Must be of same 
                                           size as array under 'f_gridind'
                            'illumination' [numpy array] Complex aperture 
                                           illumination/weights contributed
                                           by the antenna onto the grid. The 
                                           grid pixels to which it contributes 
                                           is given by 'f_gridind', 'u_gridind',
                                           'v_gridind'. Must be of same size 
                                           as array under 'f_gridind'
                            'Ef'           [numpy array] Complex electric fields
                                           contributed by the antenna onto the 
                                           grid. The grid pixels to which it 
                                           contributes is given by 'f_gridind', 
                                           'u_gridind', 'v_gridind'. Must be of 
                                           same size as array under 'f_gridind'
                'all_ant2grid'
                            [dictionary] contains the combined information of
                            mapping of all antennas to the grid. It consists of
                            the following keys and values:
                            'antind'       [numpy array] all antenna indices (to
                                           attribute ordered labels) that map to
                                           the uvf-grid
                            'u_gridind'    [numpy array] all indices to the 
                                           u-axis of the uvf-grid mapped to by 
                                           all antennas whose indices are given
                                           in key 'antind'. Must be of same size
                                           as the array under key 'antind'
                            'v_gridind'    [numpy array] all indices to the 
                                           v-axis of the uvf-grid mapped to by 
                                           all antennas whose indices are given
                                           in key 'antind'. Must be of same size
                                           as the array under key 'antind'
                            'f_gridind'    [numpy array] all indices to the 
                                           f-axis of the uvf-grid mapped to by 
                                           all antennas whose indices are given
                                           in key 'antind'. Must be of same size
                                           as the array under key 'antind'
                            'indNN_list'   [list of lists] Each item in the top
                                           level list corresponds to an antenna
                                           in the same order as in the attribute
                                           ordered_labels. Each of these items 
                                           is another list consisting of the 
                                           unraveled grid indices it contributes 
                                           to. The unraveled indices are what 
                                           are used to obtain the u-, v- and f-
                                           indices in the grid using a 
                                           conversion assuming f is the 
                                           first axis, v is the second and u is 
                                           the third
                            'illumination' [numpy array] complex values of 
                                           aperture illumination contributed by
                                           all antennas to the grid. The antenna
                                           indices are in 'antind' and the grid 
                                           indices are in 'u_gridind', 
                                           'v_gridind' and 'f_gridind'. Must be 
                                           of same size as these indices
                            'per_ant_per_freq_norm_wts'
                                           [numpy array] mapping information 
                                           on the (complex) normalizing 
                                           multiplicative factor required to 
                                           make the sum of illumination/weights 
                                           per antenna per frequency on the 
                                           grid equal to unity. This is appended 
                                           for all antennas together. Must be of 
                                           same size as array under 
                                           'illumination'
                            'Ef'           [numpy array] Complex electric fields
                                           contributed by all antennas onto the 
                                           grid. The grid pixels to which it 
                                           contributes is given by 'f_gridind', 
                                           'u_gridind', 'v_gridind'. Must be of 
                                           same size as array under 'f_gridind'
                                           and 'illumination'
    
    ant2grid_mapper
                [sparse matrix] contains the antenna array to grid mapping 
                information in sparse matrix format. When converted to a dense
                array, it will have dimensions nrows equal to size of the 3D
                cube and ncols equal to number of electric field spectra of all
                antennas over all channels. In other words, 
                nrows = nu x nv x nchan and ncols = n_ant x nchan. Dot product
                of this matrix with flattened electric field spectra or antenna
                weights will give the 3D cubes of gridded electric fields and 
                antenna array illumination respectively

    Member Functions:

    __init__()        Initializes an instance of class AntennaArray which 
                      manages information about an array of antennas.
                      
    __str__()         Prints a summary of current attributes
                      
    __add__()         Operator overloading for adding antenna(s)
                      
    __radd__()        Operator overloading for adding antenna(s)
                      
    __sub__()         Operator overloading for removing antenna(s)
                      
    pairTypetags()    Combine antenna typetags to create pairwise typetags for 
                      antenna pairs and update attribute pairwise_typetags

    add_antennas()    Routine to add antenna(s) to the antenna array instance. 
                      A wrapper for operator overloading __add__() and 
                      __radd__()
                      
    remove_antennas() Routine to remove antenna(s) from the antenna array 
                      instance. A wrapper for operator overloading __sub__()
                      
    grid()            Routine to produce a grid based on the antenna array 

    grid_convolve()   Routine to project the electric field illumination pattern
                      and the electric fields on the grid. It can operate on the
                      entire antenna array or incrementally project the electric
                      fields and illumination patterns from specific antennas on
                      to an already existing grid.

    grid_convolve_new()   
                      Routine to project the electric field illumination pattern
                      and the electric fields on the grid. 

    genMappingMatrix() 
                      Routine to construct sparse antenna-to-grid mapping matrix 
                      that will be used in projecting illumination and electric 
                      fields from the array of antennas onto the grid. It has 
                      elements very common to grid_convolve_new()

    applyMappingMatrix()
                      Constructs the grid of complex field illumination and 
                      electric fields using the sparse antenna-to-grid mapping 
                      matrix. Intended to serve as a "matrix" alternative to 
                      make_grid_cube_new() 

    grid_unconvolve() Routine to de-project the electric field illumination 
                      pattern and the electric fields on the grid. It can 
                      operate on the entire antenna array or incrementally 
                      de-project the electric fields and illumination patterns 
                      from specific antennas from an already existing grid.

    get_E_fields()    Routine to return the antenna labels, time-based weight 
                      flags and electric fields (sorted by antenna label if 
                      specified) based on selection criteria specified by flags, 
                      timestamps, frequency channels, labels and data pool (most 
                      recent or stack)

    make_grid_cube()  Constructs the grid of complex field illumination and 
                      electric fields using the gridding information determined 
                      for every antenna. Flags are taken into account while 
                      constructing this grid.

    make_grid_cube_new()  
                      Constructs the grid of complex field illumination and 
                      electric fields using the gridding information determined 
                      for every antenna. Flags are taken into account while 
                      constructing this grid.

    evalAntennaPairCorrWts()
                      Evaluate correlation of pair of antenna illumination 
                      weights on grid. It will be computed only if it was not 
                      computed or stored in attribute 
                      pairwise_typetag_crosswts_vuf earlier

    evalAntennaPairPBeam()
                      Evaluate power pattern response on sky of an antenna pair

    avgAutoCorr()     Accumulates and averages auto-correlation of electric 
                      fields of individual antennas under each polarization

    evalAutoCorr()    Estimates antenna-wise E-field auto-correlations under 
                      both polarizations. It can be for the msot recent 
                      timestamp, stacked or averaged along timestamps.

    evalAntennaAutoCorrWts()
                      Evaluate auto-correlation of aperture illumination of 
                      each antenna on the UVF-plane

    evalAllAntennaPairCorrWts()
                      Evaluate zero-centered cross-correlation of aperture 
                      illumination of each antenna pair on the UVF-plane

    makeAutoCorrCube()
                      Constructs the grid of antenna aperture illumination 
                      auto-correlation using the gridding information 
                      determined for every antenna. Flags are taken into 
                      account while constructing this grid

    makeCrossCorrWtsCube()
                      Constructs the grid of zero-centered cross-correlation 
                      of antenna aperture pairs using the gridding information 
                      determined for every antenna. Flags are taken into account 
                      while constructing this grid

    quick_beam_synthesis()  
                      A quick generator of synthesized beam using antenna array 
                      field illumination pattern using the center frequency. Not 
                      intended to be used rigorously but rather for comparison 
                      purposes and making quick plots

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
        antennas, blc, trc, gridu, gridv, grid_ready, timestamp, 
        grid_illumination, grid_Ef, f, f0, t, ordered_labels, grid_mapper, 
        antennas_center, latitude, longitude, tbinsize, auto_corr_data, 
        antenna_autocorr_set, typetags, pairwise_typetags, antenna_crosswts_set,
        pairwise_typetag_crosswts_vuf, antenna_pair_to_typetag
     
        Read docstring of class AntennaArray for details on these attributes.

        Inputs:
    
        antenna_array 
                   [Instance of class AntennaArray, dictionary holding 
                   instance(s) instance(s) of class Antenna, list of instances 
                   of class Antenna, or a single instance of class Antenna] 
                   Read docstring of member funtion __add__() for more details 
                   on this input. If provided, this will be used to initialize 
                   the instance.
        ------------------------------------------------------------------------
        """

        self.antennas = {}
        self.blc = NP.zeros(2)
        self.trc = NP.zeros(2)
        self.grid_blc = NP.zeros(2)
        self.grid_trc = NP.zeros(2)
        self.gridu, self.gridv = None, None
        self.antennas_center = NP.zeros(2, dtype=NP.float).reshape(1,-1)
        self.grid_ready = False
        self.grid_illumination = {}
        self.grid_Ef = {}
        self.caldata = {}
        self.latitude = None
        self.longitude = None
        self.f = None
        self.f0 = None
        self.t = None
        self.timestamp = None
        self.timestamps = []
        self.typetags = {}
        self.pairwise_typetags = {}
        self.antenna_pair_to_typetag = {}

        self.auto_corr_data = {}
        self.pairwise_typetag_crosswts_vuf = {}
        self.antenna_autocorr_set = False
        self.antenna_crosswts_set = False

        self._ant_contribution = {}

        self.ordered_labels = [] # Usually output from member function baseline_vectors() or get_visibilities()
        self.grid_mapper = {}
        self.ant2grid_mapper = {}  # contains the sparse mapping matrix

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

            self.grid_mapper[pol]['per_ant2grid'] = []
            self.grid_mapper[pol]['all_ant2grid'] = {}

            self.grid_illumination[pol] = None
            self.grid_Ef[pol] = None
            self._ant_contribution[pol] = {}
            self.caldata[pol] = None

            self.ant2grid_mapper[pol] = None

        if antenna_array is not None:
            self += antenna_array
            self.f = NP.copy(self.antennas.itervalues().next().f)
            self.f0 = NP.copy(self.antennas.itervalues().next().f0)
            self.t = NP.copy(self.antennas.itervalues().next().t)
            if self.latitude is None:
                self.latitude = NP.copy(self.antennas.itervalues().next().latitude)
                self.longitude = NP.copy(self.antennas.itervalues().next().longitude)
            self.timestamp = copy.deepcopy(self.antennas.itervalues().next().timestamp)
            self.timestamps += [copy.deepcopy(self.timestamp)]
        
    ############################################################################

    def __add__(self, others):

        """
        ------------------------------------------------------------------------
        Operator overloading for adding antenna(s)
    
        Inputs:
    
        others     [Instance of class AntennaArray, dictionary holding 
                   instance(s) of class Antenna, list of instances of class 
                   Antenna, or a single instance of class Antenna] If a 
                   dictionary is provided, the keys should be the antenna 
                   labels and the values should be instances of class Antenna. 
                   If a list is provided, it should be a list of valid instances 
                   of class Antenna. These instance(s) of class Antenna will be 
                   added to the existing instance of AntennaArray class.
        ------------------------------------------------------------------------
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
                    if v.typetag not in retval.typetags:
                        retval.typetags[v.typetag] = {v.label}
                    else:
                        retval.typetags[v.typetag].add(v.label)
                    print 'Antenna "{0}" added to the list of antennas.'.format(k)
            if retval.latitude is None:
                retval.latitude = others.latitude
                retval.longitude = others.longitude
        elif isinstance(others, dict):
            # for item in others.values():
            for item in others.itervalues():
                if isinstance(item, Antenna):
                    if item.label in retval.antennas:
                        print "Antenna {0} already included in the list of antennas.".format(item.label)
                        print "For updating, use the update() method. Ignoring antenna {0}".format(item.label)
                    else:
                        retval.antennas[item.label] = item
                        if item.typetag not in retval.typetags:
                            retval.typetags[item.typetag] = {item.label}
                        else:
                            retval.typetags[item.typetag].add(item.label)
                        print 'Antenna "{0}" added to the list of antennas.'.format(item.label)
                if retval.latitude is None:
                    retval.latitude = item.latitude
                    retval.longitude = item.longitude
        elif isinstance(others, list):
            for i in range(len(others)):
                if isinstance(others[i], Antenna):
                    if others[i].label in retval.antennas:
                        print "Antenna {0} already included in the list of antennas.".format(others[i].label)
                        print "For updating, use the update() method. Ignoring antenna {0}".format(others[i].label)
                    else:
                        retval.antennas[others[i].label] = others[i]
                        if others[i].typetag not in retval.typetags:
                            retval.typetags[others[i].typetag] = {others[i].label}
                        else:
                            retval.typetags[others[i].typetag].add(others[i].label)
                        print 'Antenna "{0}" added to the list of antennas.'.format(others[i].label)
                else:
                    print 'Element \# {0} is not an instance of class Antenna.'.format(i)

                if retval.latitude is None:
                    retval.latitude = others[i].latitude
                    retval.longitude = others[i].longitude

        elif isinstance(others, Antenna):
            if others.label in retval.antennas:
                print "Antenna {0} already included in the list of antennas.".format(others.label)
                print "For updating, use the update() method. Ignoring antenna {0}".format(others.label)
            else:
                retval.antennas[others.label] = others
                if others.typetag not in retval.typetags:
                    retval.typetags[others.typetag] = {others.label}
                else:
                    retval.typetags[others.typetag].add(others.label)
                print 'Antenna "{0}" added to the list of antennas.'.format(others.label)
            if retval.latitude is None:
                retval.latitude = others.latitude
                retval.longitude = others.longitude
        else:
            print 'Input(s) is/are not instance(s) of class Antenna.'

        return retval

    ############################################################################

    def __radd__(self, others):

        """
        ------------------------------------------------------------------------
        Operator overloading for adding antenna(s)
    
        Inputs:
    
        others     [Instance of class AntennaArray, dictionary holding 
                   instance(s) of class Antenna, list of instances of class 
                   Antenna, or a single instance of class Antenna] If a 
                   dictionary is provided, the keys should be the antenna 
                   labels and the values should be instances of class Antenna. 
                   If a list is provided, it should be a list of valid 
                   instances of class Antenna. These instance(s) of class 
                   Antenna will be added to the existing instance of 
                   AntennaArray class.
        ------------------------------------------------------------------------
        """

        return self.__add__(others)

    ############################################################################

    def __sub__(self, others):
        """
        ------------------------------------------------------------------------
        Operator overloading for removing antenna(s)
    
        Inputs:
    
        others     [Instance of class AntennaArray, dictionary holding 
                   instance(s) of class Antenna, list of instances of class 
                   Antenna, list of strings containing antenna labels or a 
                   single instance of class Antenna] If a dictionary is 
                   provided, the keys should be the antenna labels and the 
                   values should be instances of class Antenna. If a list is 
                   provided, it should be a list of valid instances of class 
                   Antenna. These instance(s) of class Antenna will be removed 
                   from the existing instance of AntennaArray class.
        ------------------------------------------------------------------------
        """

        retval = self
        if isinstance(others, dict):
            for item in others.values():
                if isinstance(item, Antenna):
                    if item.label not in retval.antennas:
                        print "Antenna {0} does not exist in the list of antennas.".format(item.label)
                    else:
                        del retval.antennas[item.label]
                        retval.typetags[item.typetag].remove(item.label)
                        print 'Antenna "{0}" removed from the list of antennas.'.format(item.label)
        elif isinstance(others, list):
            for i in range(0,len(others)):
                if isinstance(others[i], str):
                    if others[i] in retval.antennas:
                        retval.typetags[retval.antennas[others[i]].typetag].remove(others[i])
                        del retval.antennas[others[i]]
                        print 'Antenna {0} removed from the list of antennas.'.format(others[i])
                elif isinstance(others[i], Antenna):
                    if others[i].label in retval.antennas:
                        retval.typetags[others[i].typetag].remove(others[i].label)
                        del retval.antennas[others[i].label]
                        print 'Antenna {0} removed from the list of antennas.'.format(others[i].label)
                    else:
                        print "Antenna {0} does not exist in the list of antennas.".format(others[i].label)
                else:
                    print 'Element \# {0} has no matches in the list of antennas.'.format(i)                        
        elif others in retval.antennas:
            retval.typetags[retval.antennas[others].typetag].remove(others)
            del retval.antennas[others]
            print 'Antenna "{0}" removed from the list of antennas.'.format(others)
        elif isinstance(others, Antenna):
            if others.label in retval.antennas:
                retval.typetags[others.typetag].remove(others.label)
                del retval.antennas[others.label]
                print 'Antenna "{0}" removed from the list of antennas.'.format(others.label)
            else:
                print "Antenna {0} does not exist in the list of antennas.".format(others.label)
        else:
            print 'No matches found in existing list of antennas.'

        return retval

    ############################################################################

    def add_antennas(self, A=None):

        """
        ------------------------------------------------------------------------
        Routine to add antenna(s) to the antenna array instance. A wrapper for
        operator overloading __add__() and __radd__()
    
        Inputs:
    
        A          [Instance of class AntennaArray, dictionary holding 
                   instance(s) of class Antenna, list of instances of class 
                   Antenna, or a single instance of class Antenna] If a 
                   dictionary is provided, the keys should be the antenna 
                   labels and the values should be instances of class Antenna. 
                   If a list is provided, it should be a list of valid 
                   instances of class Antenna. These instance(s) of class 
                   Antenna will be added to the existing instance of 
                   AntennaArray class.
        ------------------------------------------------------------------------
        """

        if A is None:
            print 'No antenna(s) supplied.'
        elif isinstance(A, (list, Antenna)):
            self = self.__add__(A)
        else:
            print 'Input(s) is/are not instance(s) of class Antenna.'

    ############################################################################

    def remove_antennas(self, A=None):

        """
        ------------------------------------------------------------------------
        Routine to remove antenna(s) from the antenna array instance. A wrapper 
        for operator overloading __sub__()
    
        Inputs:
    
        A          [Instance of class AntennaArray, dictionary holding 
                   instance(s) of class Antenna, list of instances of class 
                   Antenna, or a single instance of class Antenna] If a 
                   dictionary is provided, the keys should be the antenna 
                   labels and the values should be instances of class Antenna. 
                   If a list is provided, it should be a list of 
                   valid instances of class Antenna. These instance(s) of class
                   Antenna will be removed from the existing instance of 
                   AntennaArray class.
        ------------------------------------------------------------------------
        """

        if A is None:
            print 'No antenna specified for removal.'
        else:
            self = self.__sub__(A)

    ############################################################################

    def pairTypetags(self):

        """
        ------------------------------------------------------------------------
        Combine antenna typetags to create pairwise typetags for antenna pairs
        and update attribute pairwise_typetags
        ------------------------------------------------------------------------
        """

        typekeys = self.typetags.keys()
        pairwise_typetags = {}
        for i in range(len(typekeys)):
            labels1 = list(self.typetags[typekeys[i]])
            for j in range(i,len(typekeys)):
                labels2 = list(self.typetags[typekeys[j]])
                pairwise_typetags[(typekeys[i],typekeys[j])] = {}
                if i == j:
                    pairwise_typetags[(typekeys[i],typekeys[j])]['auto'] = set([(l1,l1) for l1 in labels1])
                    pairwise_typetags[(typekeys[i],typekeys[j])]['cross'] = set([(l1,l2) for i1,l1 in enumerate(labels1) for i2,l2 in enumerate(labels2) if i1 < i2])
                else:
                    pairwise_typetags[(typekeys[i],typekeys[j])]['cross'] = set([(l1,l2) for l1 in labels1 for l2 in labels2])
        self.pairwise_typetags = pairwise_typetags
        self.antenna_pair_to_typetag = {}
        for k,val in pairwise_typetags.iteritems():
            for subkey in val:
                for v in list(val[subkey]):
                    self.antenna_pair_to_typetag[v] = k

    ############################################################################

    def antenna_positions(self, pol=None, flag=False, sort=True,
                          centering=False):
        
        """
        ------------------------------------------------------------------------
        Routine to return the antenna label and position vectors (sorted by
        antenna label if specified)

        Keyword Inputs:

        pol      [string] select positions of this polarization that are either 
                 flagged or unflagged as specified by input parameter flag. 
                 Allowed values are 'P1' and 'P2'. Default=None. 
                 This means all positions are returned irrespective of the flags

        flag     [boolean] If False, return unflagged positions, otherwise 
                 return flagged ones. Default=None means return all positions
                 independent of flagging or polarization

        sort     [boolean] If True, returned antenna information is sorted 
                 by antenna label. Default = True.

        centering 
                 [boolean] If False (default), does not subtract the mid-point
                 between the bottom left corner and the top right corner. If
                 True, subtracts the mid-point and makes it the origin

        Output:

        outdict  [dictionary] Output consists of a dictionary with the following 
                 keys and information:
                 'labels':    list of strings of antenna labels
                 'positions': position vectors of antennas (3-column 
                              array)
        ------------------------------------------------------------------------
        """

        if not isinstance(sort, bool):
            raise TypeError('sort keyword has to be a Boolean value.')

        if flag is not None:
            if not isinstance(flag, bool):
                raise TypeError('flag keyword has to be a Boolean value.')

        if pol is None:
            if sort: # sort by antenna label
                xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in sorted(self.antennas.keys())])
                labels = sorted(self.antennas.keys())
            else:
                xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in self.antennas.keys()])
                labels = self.antennas.keys()
        else:
            if not isinstance(pol, str):
                raise TypeError('Input parameter must be a string')
            
            if pol not in ['P1', 'P2']:
                raise ValueError('Invalid specification for input parameter pol')

            if sort:                   # sort by antenna label
                if flag is None:       # get all positions
                    xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in sorted(self.antennas.keys())])
                    labels = sorted(self.antennas.keys())
                else:
                    if flag:           # get flagged positions
                        xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in sorted(self.antennas.keys()) if self.antennas[label].antpol.flag[pol]])
                        labels = [label for label in sorted(self.antennas.keys()) if self.antennas[label].antpol.flag[pol]]
                    else:              # get unflagged positions
                        xyz = NP.asarray([[self.antennas[label].location.x, self.antennas[label].location.y, self.antennas[label].location.z] for label in sorted(self.antennas.keys()) if not self.antennas[label].antpol.flag[pol]])
                        labels = [label for label in sorted(self.antennas.keys()) if not self.antennas[label].antpol.flag[pol]]

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

        if centering:
            xyzcenter = 0.5 * (NP.amin(xyz, axis=0, keepdims=True) + NP.amax(xyz, axis=0, keepdims=True))
            xyz = xyz - xyzcenter
            self.antennas_center = xyzcenter[0,:2].reshape(1,-1)

        outdict = {}
        outdict['labels'] = labels
        outdict['positions'] = xyz

        return outdict

    ############################################################################

    def get_E_fields_old(self, pol, flag=False, sort=True):

        """
        ------------------------------------------------------------------------
        Routine to return the antenna label and Electric fields (sorted by
        antenna label if specified)

        Keyword Inputs:

        pol      [string] select antenna positions of this polarization that are 
                 either flagged or unflagged as specified by input parameter 
                 flag. Allowed values are 'P1' and 'P22'. Only one of these 
                 values must be specified.

        flag     [boolean] If False, return electric fields of unflagged 
                 antennas, otherwise return flagged ones. Default=None means 
                 all electric fields independent of flagging are returned.

        sort     [boolean] If True, returned antenna information is sorted 
                 by antenna label. Default = True.

        Output:

        outdict  [dictionary] Output consists of a dictionary with the following 
                 keys and information:
                 'labels':    Contains a numpy array of strings of antenna 
                              labels
                 'E-fields':  measured electric fields (n_ant x nchan array)
        ------------------------------------------------------------------------
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

    ############################################################################

    def get_E_fields(self, pol, flag=None, tselect=None, fselect=None,
                     aselect=None, datapool=None, sort=True):

        """
        ------------------------------------------------------------------------
        Routine to return the antenna labels, time-based weight flags and 
        electric fields (sorted by antenna label if specified) based on 
        selection criteria specified by flags, timestamps, frequency channels,
        labels and data pool (most recent or stack)

        Keyword Inputs:

        pol      [string] select baselines of this polarization that are either 
                 flagged or unflagged as specified by input parameter flag. 
                 Allowed values are 'P1' and 'P2'. Only one of these values 
                 must be specified.

        flag     [boolean] If False, return electric fields of unflagged 
                 antennas, otherwise return flagged ones. Default=None means 
                 all electric fields independent of flagging are returned.

        tselect  [scalar, list, numpy array] timestamp index for electric 
                 fields selection. For most recent electric fields, it must 
                 be set to -1. For all other selections, indices in tselect 
                 must be in the valid range of indices along time axis for 
                 stacked electric fields. Default=None means most recent data 
                 is selected. 

        fselect  [scalar, list, numpy array] frequency channel index for 
                 electric fields selection. Indices must be in the valid range 
                 of indices along the frequency axis for electric fields. 
                 Default=None selects all frequency channels

        aselect  [list of strings] labels of antennas to select. If set 
                 to None (default) all antennas are selected. 

        datapool [string] denotes the data pool from which electric fields are 
                 to be selected. Accepted values are 'current', 'stack' and
                 None (default, same as 'current'). If set to None or 
                 'current', the value in tselect is ignored and only 
                 electric fields of the most recent timestamp are selected. If 
                 set to None or 'current' the attribute Ef_stack is checked 
                 first and if unavailable, attribute antpol.Ef is used. For 
                 'stack' attribute Ef_stack is used 

        sort     [boolean] If True, returned antenna information is sorted 
                 by antenna label. Default = True.

        Output:

        outdict  [dictionary] Output consists of a dictionary with the following 
                 keys and information:
                 'labels'        [list of strings] Contains a list of antenna 
                                 labels
                 'E-fields'      [list or numpy array] antenna electric fields 
                                 under the specified polarization. In general, 
                                 it is a list of numpy arrays where each 
                                 array in the list corresponds to        
                                 an individual antenna and the size of
                                 each numpy array is n_ts x nchan. If input 
                                 keyword flag is set to None, the electric 
                                 fields are rearranged into a numpy array of 
                                 size n_ts x n_ant x nchan. 
                 'twts'          [list or numpy array] weights along time axis 
                                 under the specified polarization. In general
                                 it is a list of numpy arrays where each array 
                                 in the list corresponds to an individual 
                                 antenna and the size of each array is n_ts x 1. 
                                 If input keyword flag is set to None, the 
                                 time weights are rearranged into a numpy array 
                                 of size n_ts x n_ant x 1
        ------------------------------------------------------------------------
        """

        if not isinstance(sort, bool):
            raise TypeError('sort keyword has to be a Boolean value.')

        if aselect is None:
            labels = self.antennas.keys()
        elif isinstance(aselect, list):
            labels = [label for label in aselect if label in self.antennas]
            
        if sort:
            labels = sorted(labels)

        efinfo = [self.antennas[label].get_E_fields(pol, flag=flag, tselect=tselect, fselect=fselect, datapool=datapool) for label in labels]
      
        outdict = {}
        outdict['labels'] = labels
        outdict['twts'] = [einfo['twts'] for einfo in efinfo]
        outdict['E-fields'] = [einfo['E-fields'] for einfo in efinfo]
        if flag is None:
            outdict['E-fields'] = NP.swapaxes(NP.asarray(outdict['E-fields']), 0, 1)
            outdict['twts'] = NP.swapaxes(NP.asarray(outdict['twts']), 0, 1)
            outdict['twts'] = outdict['twts'][:,:,NP.newaxis]

        return outdict

    ############################################################################

    def avgAutoCorr(self, tbinsize=None):

        """
        ------------------------------------------------------------------------
        Accumulates and averages auto-correlation of electric fields of 
        individual antennas under each polarization

        Inputs:

        tbinsize [scalar or dictionary] Contains bin size of timestamps while
                 averaging. Default = None means all antenna E-field 
                 auto-correlation spectra over all timestamps are averaged. If 
                 scalar, the same (positive) value applies to all polarizations. 
                 If dictionary, timestamp bin size (positive) in seconds is 
                 provided under each key 'P1' and 'P2'. If any of the keys is 
                 missing the auto-correlated antenna E-field spectra for that 
                 polarization are averaged over all timestamps.
        ------------------------------------------------------------------------
        """

        timestamps = NP.asarray(self.timestamps).astype(NP.float)
        twts = {}
        auto_corr_data = {}
        pol = ['P1', 'P2']
        for p in pol:
            Ef_info = self.get_E_fields(p, flag=None, tselect=NP.arange(len(self.timestamps)), fselect=None, aselect=None, datapool='stack', sort=True)
            twts[p] = []
            auto_corr_data[p] = {}
            if tbinsize is None: # Average across all timestamps
                auto_corr_data[p]['E-fields'] = NP.nansum(NP.abs(Ef_info['E-fields'])**2, axis=0, keepdims=True)
                auto_corr_data[p]['twts'] = NP.sum(Ef_info['twts'], axis=0, keepdims=True).astype(NP.float)
                auto_corr_data[p]['labels'] = Ef_info['labels']
                self.tbinsize = tbinsize
            elif isinstance(tbinsize, (int,float)): # Apply same time bin size to all polarizations
                split_ind = NP.arange(timestamps.min()+tbinsize, timstamps.max(), tbinsize)
                twts_split = NP.array_split(Ef_info['twts'], split_ind, axis=0)
                Ef_split = NP.array_split(Ef_info['E-fields'], split_ind, axis=0)
                for i in xrange(split_ind.size):
                    if 'E-fields' not in auto_corr_data[p]:
                        auto_corr_data[p]['E-fields'] = NP.nansum(NP.abs(Ef_info['E-fields'])**2, axis=0, keepdims=True)
                        auto_corr_data[p]['twts'] = NP.sum(Ef_info['twts'], axis=0, keepdims=True).astype(NP.float)
                    else:
                        auto_corr_data[p]['E-fields'] = NP.vstack((auto_corr_data[p]['E-fields'], NP.nansum(NP.abs(Ef_info['E-fields'])**2, axis=0, keepdims=True)))
                        auto_corr_data[p]['twts'] = NP.vstack((auto_corr_data[p]['twts'], NP.sum(Ef_info['twts'], axis=0, keepdims=True))).astype(NP.float)
                auto_corr_data[p]['labels'] = Ef_info['labels']
                self.tbinsize = tbinsize
            elif isinstance(tbinsize, dict):
                tbsize = {}
                if p not in tbinsize:
                    auto_corr_data[p]['E-fields'] = NP.nansum(NP.abs(Ef_info['E-fields'])**2, axis=0, keepdims=True)
                    auto_corr_data[p]['twts'] = NP.sum(Ef_info['twts'], axis=0, keepdims=True).astype(NP.float)
                    tbsize[p] = None
                elif tbinsize[p] is None:
                    auto_corr_data[p]['E-fields'] = NP.nansum(NP.abs(Ef_info['E-fields'])**2, axis=0, keepdims=True)
                    auto_corr_data[p]['twts'] = NP.sum(Ef_info['twts'], axis=0, keepdims=True).astype(NP.float)
                    tbsize[p] = None
                elif isinstance(tbinsize[p], (int,float)):
                    split_ind = NP.arange(timestamps.min()+tbinsize, timstamps.max(), tbinsize)
                    twts_split = NP.array_split(Ef_info['twts'], split_ind, axis=0)
                    Ef_split = NP.array_split(Ef_info['E-fields'], split_ind, axis=0)
                    for i in xrange(split_ind.size):
                        if 'E-fields' not in auto_corr_data[p]:
                            auto_corr_data[p]['E-fields'] = NP.nansum(NP.abs(Ef_info['E-fields'])**2, axis=0, keepdims=True)
                            auto_corr_data[p]['twts'] = NP.sum(Ef_info['twts'], axis=0, keepdims=True).astype(NP.float)
                        else:
                            auto_corr_data[p]['E-fields'] = NP.vstack((auto_corr_data[p]['E-fields'], NP.nansum(NP.abs(Ef_info['E-fields'])**2, axis=0, keepdims=True)))
                            auto_corr_data[p]['twts'] = NP.vstack((auto_corr_data[p]['twts'], NP.sum(Ef_info['twts'], axis=0, keepdims=True))).astype(NP.float)
                    tbsize[pol] = tbinsize[pol]                 
                else:
                    raise ValueError('Input tbinsize is invalid')
                auto_corr_data[p]['labels'] = Ef_info['labels']
                self.tbinsize = tbsize
            else:
                raise ValueError('Input tbinsize is invalid')

            auto_corr_data[p]['E-fields'] = auto_corr_data[p]['E-fields'] / auto_corr_data[p]['twts']
        self.auto_corr_data['avg'] = auto_corr_data

    ############################################################################

    def evalAutoCorr(self, datapool=None, tbinsize=None):

        """
        ------------------------------------------------------------------------
        Estimates antenna-wise E-field auto-correlations under both
        polarizations. It can be for the most recent timestamp, stacked or
        averaged along timestamps.

        Inputs:

        datapool [string] denotes the data pool from which electric fields are 
                 to be selected. Accepted values are 'current', 'stack', avg' or
                 None (default, same as 'current'). If set to None or 
                 'current', the value in tselect is ignored and only 
                 electric fields of the most recent timestamp are selected. If
                 set to 'avg', the auto-correlations from the stack are 
                 averaged along the timestamps using time bin size specified
                 in tbinsize

        tbinsize [scalar or dictionary] Contains bin size of timestamps while
                 averaging. Will be used only if datapool is set to 'avg'. 
                 Default = None means all antenna E-field 
                 auto-correlation spectra over all timestamps are averaged. If 
                 scalar, the same (positive) value applies to all polarizations. 
                 If dictionary, timestamp bin size (positive) in seconds is 
                 provided under each key 'P1' and 'P2'. If any of the keys is 
                 missing the auto-correlated antenna E-field spectra for that 
                 polarization are averaged over all timestamps.
        ------------------------------------------------------------------------
        """

        if datapool not in [None, 'current', 'stack', 'avg']:
            raise ValueError('Input datapool must be set to None, "current", "stack" or "avg"')

        if datapool in [None, 'current']:
            self.auto_corr_data['current'] = {}
            for p in pol:
                Ef_info = self.get_E_fields(p, flag=None, tselect=-1, fselect=None, aselect=None, datapool='', sort=True)
                self.auto_corr_data['current'][p] = Ef_info
                
        if datapool in [None, 'stack']:
            self.auto_corr_data['stack'] = {}
            for p in pol:
                Ef_info = self.get_E_fields(p, flag=None, tselect=NP.arange(len(self.timestamps)), fselect=None, aselect=None, datapool='', sort=True)
                self.auto_corr_data['current'][p] = Ef_info

        if datapool in [None, 'avg']:
            self.avgAutoCorr(tbinsize=tbinsize)

    ############################################################################

    def FT(self, pol=None, parallel=False, nproc=None):

        """
        ------------------------------------------------------------------------
        Computes the Fourier transform of the time series of the antennas in the 
        antenna array to compute the visibility spectra
        ------------------------------------------------------------------------
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
        
    ############################################################################

    def grid(self, uvspacing=0.5, xypad=None, pow2=True):
        
        """
        ------------------------------------------------------------------------
        Routine to produce a grid based on the antenna array 

        Inputs:

        uvspacing   [Scalar] Positive value indicating the maximum uv-spacing
                    desirable at the lowest wavelength (max frequency). 
                    Default = 0.5

        xypad       [List] Padding to be applied around the antenna locations 
                    before forming a grid. Units in meters. List elements should 
                    be positive. If it is a one-element list, the element is 
                    applicable to both x and y axes. If list contains three or 
                    more elements, only the first two elements are considered 
                    one for each axis. Default = None.

        pow2        [Boolean] If set to True, the grid is forced to have a size 
                    a next power of 2 relative to the actual sie required. If 
                    False, gridding is done with the appropriate size as 
                    determined by uvspacing. Default = True.
        ------------------------------------------------------------------------
        """

        if self.f is None:
            self.f = self.antennas.itervalues().next().f

        if self.f0 is None:
            self.f0 = self.antennas.itervalues().next().f0

        wavelength = FCNST.c / self.f
        min_lambda = NP.abs(wavelength).min()

        # Change itervalues() to values() when porting to Python 3.x
        # May have to change *blc and *trc with zip(*blc) and zip(*trc) when using Python 3.x

        blc = NP.asarray([[self.antennas[label].blc[0,0], self.antennas[label].blc[0,1]] for label in self.antennas]).reshape(-1,2)
        trc = NP.asarray([[self.antennas[label].trc[0,0], self.antennas[label].trc[0,1]] for label in self.antennas]).reshape(-1,2)

        xycenter = 0.5 * (NP.amin(blc, axis=0, keepdims=True) + NP.amax(trc, axis=0, keepdims=True))
        blc = blc - xycenter
        trc = trc - xycenter

        self.trc = NP.amax(NP.abs(NP.vstack((blc, trc))), axis=0).ravel() / min_lambda
        self.blc = -1 * self.trc
        self.antennas_center = xycenter

        if xypad is None:
            xypad = 0.0

        self.gridu, self.gridv = GRD.grid_2d([(self.blc[0], self.trc[0]), (self.blc[1], self.trc[1])], pad=xypad/min_lambda, spacing=uvspacing, pow2=True)

        self.grid_blc = NP.asarray([self.gridu.min(), self.gridv.min()])
        self.grid_trc = NP.asarray([self.gridu.max(), self.gridv.max()])

        self.grid_ready = True

    ############################################################################

    def grid_convolve(self, pol=None, ants=None, unconvolve_existing=False,
                      normalize=False, method='NN', distNN=NP.inf, tol=None,
                      maxmatch=None, identical_antennas=True, cal_loop=False,
                      gridfunc_freq=None, mapping='weighted', wts_change=False,
                      parallel=False, nproc=None, pp_method='pool',
                      verbose=True): 

        """
        ------------------------------------------------------------------------
        Routine to project the complex illumination field pattern and the 
        electric fields on the grid. It can operate on the entire antenna array 
        or incrementally project the electric fields and complex illumination 
        field patterns from specific antennas on to an already existing grid. 
        (The latter is not implemented yet)

        Inputs:

        pol        [String] The polarization to be gridded. Can be set to 'P1' 
                   or 'P2'. If set to None, gridding for all the polarizations 
                   is performed. Default = None

        ants       [instance of class AntennaArray, single instance or list 
                   of instances of class Antenna, or a dictionary holding 
                   instances of class Antenna] If a dictionary is provided, 
                   the keys should be the antenna labels and the values 
                   should be instances of class Antenna. If a list is 
                   provided, it should be a list of valid instances of class 
                   Antenna. These instance(s) of class Antenna will 
                   be merged to the existing grid contained in the instance of 
                   AntennaArray class. If ants is not provided (set to 
                   None), the gridding operations will be performed on the 
                   set of antennas contained in the instance of class 
                   entire AntennaArray. Default = None.

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

        normalize  [Boolean] Default = False. If set to True, the gridded 
                   weights are divided by the sum of weights so that the gridded 
                   weights add up to unity. (Need to work on normaliation)

        method     [string] The gridding method to be used in applying the 
                   antenna weights on to the antenna array grid. 
                   Accepted values are 'NN' (nearest neighbour - default), 'CS' 
                   (cubic spline), or 'BL' (Bi-linear). In case of applying grid 
                   weights by 'NN' method, an optional distance upper bound for 
                   the nearest neighbour can be provided in the parameter distNN 
                   to prune the search and make it efficient. Currently, only 
                   the nearest neighbour method is operational.

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
                   considered for nearest neighbour lookup. Default=None implies 
                   all lookup values will be considered for nearest neighbour 
                   determination. tol is to be interpreted as a minimum value 
                   considered as significant in the lookup table. 

        identical_antennas
                   [boolean] indicates if all antenna elements are to be
                   treated as identical. If True (default), they are identical
                   and their gridding kernels are identical. If False, they are
                   not identical and each one has its own gridding kernel.

        cal_loop   [boolean] If True, the calibration loop is assumed to be ON 
                   and hence the calibrated electric fields are set in the 
                   calibration loop. If False (default), the calibration loop is
                   assumed to be OFF and the current electric fields are assumed 
                   to be the calibrated data to be mapped to the grid 
                   via gridding convolution.

        gridfunc_freq
                   [String scalar] If set to None (not provided) or to 'scale'
                   assumes that attribute wtspos is given for a
                   reference frequency which need to be scaled for the frequency
                   channels. Will be ignored if the number of elements of list 
                   in this attribute under the specific polarization are the 
                   same as the number of frequency channels.

        mapping    [string] indicates the type of mapping between antenna 
                   locations and the grid locations. Allowed values are 
                   'sampled' and 'weighted' (default). 'sampled' means only the 
                   antenna measurement closest ot a grid location contributes to 
                   that grid location, whereas, 'weighted' means that all the 
                   antennas contribute in a weighted fashion to their nearest 
                   grid location. The former is faster but possibly discards 
                   antenna data whereas the latter is slower but includes all 
                   data along with their weights.

        wts_change [boolean] indicates if weights and/or their lcoations have 
                   changed from the previous intergration or snapshot. 
                   Default=False means they have not changed. In such a case the 
                   antenna-to-grid mapping and grid illumination pattern do not 
                   have to be determined, and mapping and values from the 
                   previous snapshot can be used. If True, a new mapping has to 
                   be determined.

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
        ------------------------------------------------------------------------
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
                        ant_dict = self.antenna_positions(pol=apol, flag=False, sort=True, centering=True)
                        ant_xy = ant_dict['positions'][:,:2]
                        self.ordered_labels = ant_dict['labels']
                        n_ant = ant_xy.shape[0]

                        Ef_dict = self.get_E_fields_old(apol, flag=False, sort=True)
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
                    ant_dict = self.antenna_positions(pol=apol, flag=None, sort=True, centering=True)
                    self.ordered_labels = ant_dict['labels']
                    ant_xy = ant_dict['positions'][:,:2] # n_ant x 2
                    n_ant = ant_xy.shape[0]

                    # Ef_dict = self.get_E_fields(apol, flag=None, sort=True)
                    # Ef = Ef_dict['E-fields'].astype(NP.complex64)  # n_ant x nchan

                    if not cal_loop:
                        self.caldata[apol] = self.get_E_fields(apol, flag=None, tselect=-1, fselect=None, aselect=None, datapool='current', sort=True)
                    else:
                        if self.caldata[apol] is None:
                            self.caldata[apol] = self.get_E_fields(apol, flag=None, tselect=-1, fselect=None, aselect=None, datapool='current', sort=True)

                    Ef = self.caldata[apol]['E-fields'].astype(NP.complex64)  #  (n_ts=1) x n_ant x nchan
                    Ef = NP.squeeze(Ef, axis=0)  # n_ant x nchan
                    if Ef.shape[0] != n_ant:
                        raise ValueError('Encountered unexpected behavior. Need to debug.')
                    ant_labels = self.caldata[apol]['labels']
                    twts = self.caldata[apol]['twts']  # (n_ts=1) x n_ant x (nchan=1)
                    twts = NP.squeeze(twts, axis=(0,2)) # n_ant

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
                                            self.grid_mapper[apol]['labels'][label]['twts'] = twts[ant_labels.index(label)]                                        
                                            # self.grid_mapper[apol]['labels'][label]['flag'] = self.antennas[label].antpol.flag[apol]
        
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
                                        self.grid_mapper[apol]['labels'][label]['twts'] = twts[ant_labels.index(label)] 
                                        # self.grid_mapper[apol]['labels'][label]['flag'] = self.antennas[label].antpol.flag[apol]
    
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
    
                                if nproc is not None:
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
                                    self.grid_mapper[apol]['labels'][label]['twts'] = twts[ant_labels.index(label)] 
                                    # self.grid_mapper[apol]['labels'][label]['flag'] = self.antennas[label].antpol.flag[apol]
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
                                        self.grid_mapper[apol]['labels'][label]['twts'] = twts[ant_labels.index(label)]    
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
                                        self.grid_mapper[apol]['labels'][label]['twts'] = twts[ant_labels.index(label)]    
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
                                    self.grid_mapper[apol]['labels'][label]['twts'] = twts[ant_labels.index(label)]                                        
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

    ############################################################################
    
    def grid_convolve_new(self, pol=None, normalize=False, method='NN',
                          distNN=NP.inf, identical_antennas=True,
                          cal_loop=False, gridfunc_freq=None, wts_change=False,
                          parallel=False, nproc=None, pp_method='pool',
                          verbose=True): 

        """
        ------------------------------------------------------------------------
        Routine to project the complex illumination field pattern and the 
        electric fields on the grid from the antenna array

        Inputs:

        pol        [String] The polarization to be gridded. Can be set to 'P1' 
                   or 'P2'. If set to None, gridding for all the polarizations 
                   is performed. Default = None

        normalize  [Boolean] Default = False. If set to True, the gridded 
                   weights are divided by the sum of weights so that the gridded 
                   weights add up to unity. (Need to work on normaliation)

        method     [string] The gridding method to be used in applying the 
                   antenna weights on to the antenna array grid. 
                   Accepted values are 'NN' (nearest neighbour - default), 'CS' 
                   (cubic spline), or 'BL' (Bi-linear). In case of applying grid 
                   weights by 'NN' method, an optional distance upper bound for 
                   the nearest neighbour can be provided in the parameter distNN 
                   to prune the search and make it efficient. Currently, only 
                   the nearest neighbour method is operational.

        distNN     [scalar] A positive value indicating the upper bound on 
                   distance to the nearest neighbour in the gridding process. It 
                   has units of distance, the same units as the antenna 
                   attribute location and antenna array attribute gridx 
                   and gridy. Default is NP.inf (infinite distance). It will be 
                   internally converted to have same units as antenna 
                   attributes wtspos (units in number of wavelengths). To ensure
                   all relevant pixels in the grid, the search distance used 
                   internally will be a fraction more than distNN

        identical_antennas
                   [boolean] indicates if all antenna elements are to be
                   treated as identical. If True (default), they are identical
                   and their gridding kernels are identical. If False, they are
                   not identical and each one has its own gridding kernel.

        cal_loop   [boolean] If True, the calibration loop is assumed to be ON 
                   and hence the calibrated electric fields are set in the 
                   calibration loop. If False (default), the calibration loop is
                   assumed to be OFF and the current electric fields are assumed 
                   to be the calibrated data to be mapped to the grid 
                   via gridding convolution.

        gridfunc_freq
                   [String scalar] If set to None (not provided) or to 'scale'
                   assumes that attribute wtspos is given for a
                   reference frequency which need to be scaled for the frequency
                   channels. Will be ignored if the number of elements of list 
                   in this attribute under the specific polarization are the 
                   same as the number of frequency channels.

        wts_change [boolean] indicates if weights and/or their lcoations have 
                   changed from the previous intergration or snapshot. 
                   Default=False means they have not changed. In such a case the 
                   antenna-to-grid mapping and grid illumination pattern do not 
                   have to be determined, and mapping and values from the 
                   previous snapshot can be used. If True, a new mapping has to 
                   be determined.

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
        ------------------------------------------------------------------------
        """

        if pol is None:
            pol = ['P1', 'P2']
        elif not isinstance(pol, list):
            pol = [pol]

        if not self.grid_ready:
            self.grid()

        du = self.gridu[0,1] - self.gridu[0,0]
        dv = self.gridv[1,0] - self.gridv[0,0]
        wavelength = FCNST.c / self.f
        min_lambda = NP.abs(wavelength).min()
        rmaxNN = 0.5 * NP.sqrt(du**2 + dv**2) * min_lambda
 
        krn = {}
        antpol = ['P1', 'P2']
        for apol in antpol:
            krn[apol] = None
            if apol in pol:
                ant_dict = self.antenna_positions(pol=apol, flag=None, sort=True, centering=True)
                self.ordered_labels = ant_dict['labels']
                ant_xy = ant_dict['positions'][:,:2] # n_ant x 2
                n_ant = ant_xy.shape[0]

                if not cal_loop:
                    self.caldata[apol] = self.get_E_fields(apol, flag=None, tselect=-1, fselect=None, aselect=None, datapool='current', sort=True)
                else:
                    if self.caldata[apol] is None:
                        self.caldata[apol] = self.get_E_fields(apol, flag=None, tselect=-1, fselect=None, aselect=None, datapool='current', sort=True)

                Ef = self.caldata[apol]['E-fields'].astype(NP.complex64)  #  (n_ts=1) x n_ant x nchan
                Ef = NP.squeeze(Ef, axis=0)  # n_ant x nchan
                if Ef.shape[0] != n_ant:
                    raise ValueError('Encountered unexpected behavior. Need to debug.')
                ant_labels = self.caldata[apol]['labels']
                twts = self.caldata[apol]['twts']  # (n_ts=1) x n_ant x (nchan=1)
                twts = NP.squeeze(twts, axis=(0,2)) # n_ant

                if verbose:
                    print 'Gathered antenna data for gridding convolution for timestamp {0}'.format(self.timestamp)

                if wts_change or (not self.grid_mapper[apol]['all_ant2grid']):
                    self.grid_mapper[apol]['per_ant2grid'] = []
                    self.grid_mapper[apol]['all_ant2grid'] = {}
                    gridlocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                    if gridfunc_freq == 'scale':
                        grid_xy = gridlocs[NP.newaxis,:,:] * wavelength.reshape(-1,1,1)   # nchan x nv x nu
                        wl = NP.ones(gridlocs.shape[0])[NP.newaxis,:] * wavelength.reshape(-1,1)
                        grid_xy = grid_xy.reshape(-1,2)
                        wl = wl.reshape(-1)
                        indNN_list, antind, fvu_gridind = LKP.find_NN(ant_xy, grid_xy, distance_ULIM=2.0*distNN, flatten=True, parallel=False)
                        dxy = grid_xy[fvu_gridind,:] - ant_xy[antind,:]
                        fvu_gridind_unraveled = NP.unravel_index(fvu_gridind, (self.f.size,)+self.gridu.shape)   # f-v-u order since temporary grid was created as nchan x nv x nu
                        self.grid_mapper[apol]['all_ant2grid']['antind'] = NP.copy(antind)
                        self.grid_mapper[apol]['all_ant2grid']['u_gridind'] = NP.copy(fvu_gridind_unraveled[2])
                        self.grid_mapper[apol]['all_ant2grid']['v_gridind'] = NP.copy(fvu_gridind_unraveled[1])                            
                        self.grid_mapper[apol]['all_ant2grid']['f_gridind'] = NP.copy(fvu_gridind_unraveled[0])
                        self.grid_mapper[apol]['all_ant2grid']['indNN_list'] = copy.deepcopy(indNN_list)

                        if identical_antennas:
                            arbitrary_antenna_aperture = self.antennas.itervalues().next().aperture
                            krn = arbitrary_antenna_aperture.compute(dxy, wavelength=wl[fvu_gridind], pol=apol, rmaxNN=rmaxNN, load_lookup=False)
                        else:
                            # This block #1 is one way to go about per antenna
                            for ai,gi in enumerate(indNN_list):
                                if len(gi) > 0:
                                    label = self.ordered_labels[ai]
                                    ind = NP.asarray(gi)
                                    diffxy = grid_xy[ind,:].reshape(-1,2) - ant_xy[ai,:].reshape(-1,2)
                                    krndict = self.antennas[label].aperture.compute(diffxy, wavelength=wl[ind], pol=apol, rmaxNN=rmaxNN, load_lookup=False)
                                    if krn[apol] is None:
                                        krn[apol] = NP.copy(krndict[apol])
                                    else:
                                        krn[apol] = NP.append(krn[apol], krndict[apol])
                                    
                            # # This block #2 is another way equivalent to above block #1
                            # uniq_antind = NP.unique(antind)
                            # anthist, antbe, antbn, antri = OPS.binned_statistic(antind, statistic='count', bins=NP.append(uniq_antind, uniq_antind.max()+1))
                            # for i,uantind in enumerate(uniq_antind):
                            #     label = self.ordered_labels[uantind]
                            #     ind = antri[antri[i]:antri[i+1]]
                            #     krndict = self.antennas[label].aperture.compute(dxy[ind,:], wavelength=wl[ind], pol=apol, rmaxNN=rmaxNN, load_lookup=False)
                            #     if krn[apol] is None:
                            #         krn[apol] = NP.copy(krndict[apol])
                            #     else:
                            #         krn[apol] = NP.append(krn[apol], krndict[apol])

                        self.grid_mapper[apol]['all_ant2grid']['illumination'] = NP.copy(krn[apol])
                    else: # Weights do not scale with frequency (needs serious development)
                        pass
                        
                    # Determine weights that can normalize sum of kernel per antenna per frequency to unity
                    per_ant_per_freq_norm_wts = NP.zeros(antind.size, dtype=NP.complex64)
                    # per_ant_per_freq_norm_wts = NP.ones(antind.size, dtype=NP.complex64)                    
                    
                    runsum = 0
                    for ai,gi in enumerate(indNN_list):
                        if len(gi) > 0:
                            fvu_ind = NP.asarray(gi)
                            unraveled_fvu_ind = NP.unravel_index(fvu_ind, (self.f.size,)+self.gridu.shape)
                            f_ind = unraveled_fvu_ind[0]
                            v_ind = unraveled_fvu_ind[1]
                            u_ind = unraveled_fvu_ind[2]
                            chanhist, chanbe, chanbn, chanri = OPS.binned_statistic(f_ind, statistic='count', bins=NP.arange(self.f.size+1))
                            for ci in xrange(self.f.size):
                                if chanhist[ci] > 0.0:
                                    select_chan_ind = chanri[chanri[ci]:chanri[ci+1]]
                                    per_ant_per_freq_kernel_sum = NP.sum(krn[apol][runsum:runsum+len(gi)][select_chan_ind])
                                    per_ant_per_freq_norm_wts[runsum:runsum+len(gi)][select_chan_ind] = 1.0 / per_ant_per_freq_kernel_sum

                        per_ant2grid_info = {}
                        per_ant2grid_info['label'] = self.ordered_labels[ai]
                        per_ant2grid_info['f_gridind'] = NP.copy(f_ind)
                        per_ant2grid_info['u_gridind'] = NP.copy(u_ind)
                        per_ant2grid_info['v_gridind'] = NP.copy(v_ind)
                        # per_ant2grid_info['fvu_gridind'] = NP.copy(gi)
                        per_ant2grid_info['per_ant_per_freq_norm_wts'] = per_ant_per_freq_norm_wts[runsum:runsum+len(gi)]
                        per_ant2grid_info['illumination'] = krn[apol][runsum:runsum+len(gi)]
                        self.grid_mapper[apol]['per_ant2grid'] += [copy.deepcopy(per_ant2grid_info)]
                        runsum += len(gi)

                    self.grid_mapper[apol]['all_ant2grid']['per_ant_per_freq_norm_wts'] = NP.copy(per_ant_per_freq_norm_wts)

                # Determine the gridded electric fields
                Ef_on_grid = Ef[(self.grid_mapper[apol]['all_ant2grid']['antind'], self.grid_mapper[apol]['all_ant2grid']['f_gridind'])]
                self.grid_mapper[apol]['all_ant2grid']['Ef'] = copy.deepcopy(Ef_on_grid)
                runsum = 0
                for ai,gi in enumerate(self.grid_mapper[apol]['all_ant2grid']['indNN_list']):
                    if len(gi) > 0:
                        self.grid_mapper[apol]['per_ant2grid'][ai]['Ef'] = Ef_on_grid[runsum:runsum+len(gi)]
                        runsum += len(gi)

    ############################################################################

    def genMappingMatrix(self, pol=None, normalize=True, method='NN',
                         distNN=NP.inf, identical_antennas=True,
                         gridfunc_freq=None, wts_change=False, parallel=False,
                         nproc=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Routine to construct sparse antenna-to-grid mapping matrix that will be
        used in projecting illumination and electric fields from the array of 
        antennas onto the grid. It has elements very common to 
        grid_convolve_new()

        Inputs:

        pol        [String] The polarization to be gridded. Can be set to 'P1' 
                   or 'P2'. If set to None, gridding for all the polarizations 
                   is performed. Default = None

        normalize  [Boolean] Default = False. If set to True, the gridded 
                   weights are divided by the sum of weights so that the gridded 
                   weights add up to unity. (Need to work on normalization)

        method     [string] The gridding method to be used in applying the 
                   antenna weights on to the antenna array grid. 
                   Accepted values are 'NN' (nearest neighbour - default), 'CS' 
                   (cubic spline), or 'BL' (Bi-linear). In case of applying grid 
                   weights by 'NN' method, an optional distance upper bound for 
                   the nearest neighbour can be provided in the parameter distNN 
                   to prune the search and make it efficient. Currently, only 
                   the nearest neighbour method is operational.

        distNN     [scalar] A positive value indicating the upper bound on 
                   distance to the nearest neighbour in the gridding process. It 
                   has units of distance, the same units as the antenna 
                   attribute location and antenna array attribute gridx 
                   and gridy. Default is NP.inf (infinite distance). It will be 
                   internally converted to have same units as antenna 
                   attributes wtspos (units in number of wavelengths). To ensure
                   all relevant pixels in the grid, the search distance used 
                   internally will be a fraction more than distNN

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
                   in this attribute under the specific polarization are the 
                   same as the number of frequency channels.

        wts_change [boolean] indicates if weights and/or their lcoations have 
                   changed from the previous intergration or snapshot. 
                   Default=False means they have not changed. In such a case the 
                   antenna-to-grid mapping and grid illumination pattern do not 
                   have to be determined, and mapping and values from the 
                   previous snapshot can be used. If True, a new mapping has to 
                   be determined.

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

        verbose    [boolean] If True, prints diagnostic and progress messages. 
                   If False (default), suppress printing such messages.

        NOTE: Although certain portions are parallelizable, the overheads in 
        these processes seem to make it worse than serial processing. It is 
        advisable to stick to serialized version unless testing with larger
        data sets clearly indicates otherwise.
        ------------------------------------------------------------------------
        """

        if pol is None:
            pol = ['P1', 'P2']
        elif not isinstance(pol, list):
            pol = [pol]

        if not self.grid_ready:
            self.grid()

        du = self.gridu[0,1] - self.gridu[0,0]
        dv = self.gridv[1,0] - self.gridv[0,0]
        wavelength = FCNST.c / self.f
        min_lambda = NP.abs(wavelength).min()
        rmaxNN = 0.5 * NP.sqrt(du**2 + dv**2) * min_lambda
 
        krn = {}
        self.ant2grid_mapper = {}
        antpol = ['P1', 'P2']
        for apol in antpol:
            krn[apol] = None
            self.ant2grid_mapper[apol] = None
            if apol in pol:
                ant_dict = self.antenna_positions(pol=apol, flag=None, sort=True, centering=True)
                self.ordered_labels = ant_dict['labels']
                ant_xy = ant_dict['positions'][:,:2] # n_ant x 2
                n_ant = ant_xy.shape[0]

                if verbose:
                    print 'Gathered antenna data for gridding convolution for timestamp {0}'.format(self.timestamp)

                if wts_change or (not self.grid_mapper[apol]['all_ant2grid']):
                    self.grid_mapper[apol]['per_ant2grid'] = []
                    self.grid_mapper[apol]['all_ant2grid'] = {}
                    gridlocs = NP.hstack((self.gridu.reshape(-1,1), self.gridv.reshape(-1,1)))
                    if gridfunc_freq == 'scale':
                        grid_xy = gridlocs[NP.newaxis,:,:] * wavelength.reshape(-1,1,1)   # nchan x nv x nu
                        wl = NP.ones(gridlocs.shape[0])[NP.newaxis,:] * wavelength.reshape(-1,1)
                        grid_xy = grid_xy.reshape(-1,2)
                        wl = wl.reshape(-1)
                        indNN_list, antind, fvu_gridind = LKP.find_NN(ant_xy, grid_xy, distance_ULIM=2.0*distNN, flatten=True, parallel=False)
                        dxy = grid_xy[fvu_gridind,:] - ant_xy[antind,:]
                        fvu_gridind_unraveled = NP.unravel_index(fvu_gridind, (self.f.size,)+self.gridu.shape)   # f-v-u order since temporary grid was created as nchan x nv x nu
                        self.grid_mapper[apol]['all_ant2grid']['antind'] = NP.copy(antind)
                        self.grid_mapper[apol]['all_ant2grid']['u_gridind'] = NP.copy(fvu_gridind_unraveled[2])
                        self.grid_mapper[apol]['all_ant2grid']['v_gridind'] = NP.copy(fvu_gridind_unraveled[1])                            
                        self.grid_mapper[apol]['all_ant2grid']['f_gridind'] = NP.copy(fvu_gridind_unraveled[0])
                        # self.grid_mapper[apol]['all_ant2grid']['indNN_list'] = copy.deepcopy(indNN_list)

                        if identical_antennas:
                            arbitrary_antenna_aperture = self.antennas.itervalues().next().aperture
                            krn = arbitrary_antenna_aperture.compute(dxy, wavelength=wl[fvu_gridind], pol=apol, rmaxNN=rmaxNN, load_lookup=False)
                        else:
                            # This block #1 is one way to go about per antenna
                            for ai,gi in enumerate(indNN_list):
                                if len(gi) > 0:
                                    label = self.ordered_labels[ai]
                                    ind = NP.asarray(gi)
                                    diffxy = grid_xy[ind,:].reshape(-1,2) - ant_xy[ai,:].reshape(-1,2)
                                    krndict = self.antennas[label].aperture.compute(diffxy, wavelength=wl[ind], pol=apol, rmaxNN=rmaxNN, load_lookup=False)
                                    if krn[apol] is None:
                                        krn[apol] = NP.copy(krndict[apol])
                                    else:
                                        krn[apol] = NP.append(krn[apol], krndict[apol])
                                    
                            # # This block #2 is another way equivalent to above block #1
                            # uniq_antind = NP.unique(antind)
                            # anthist, antbe, antbn, antri = OPS.binned_statistic(antind, statistic='count', bins=NP.append(uniq_antind, uniq_antind.max()+1))
                            # for i,uantind in enumerate(uniq_antind):
                            #     label = self.ordered_labels[uantind]
                            #     ind = antri[antri[i]:antri[i+1]]
                            #     krndict = self.antennas[label].aperture.compute(dxy[ind,:], wavelength=wl[ind], pol=apol, rmaxNN=rmaxNN, load_lookup=False)
                            #     if krn[apol] is None:
                            #         krn[apol] = NP.copy(krndict[apol])
                            #     else:
                            #         krn[apol] = NP.append(krn[apol], krndict[apol])

                        self.grid_mapper[apol]['all_ant2grid']['illumination'] = NP.copy(krn[apol])
                    else: # Weights do not scale with frequency (needs serious development)
                        pass
                        
                    # Determine weights that can normalize sum of kernel per antenna per frequency to unity
                    per_ant_per_freq_norm_wts = NP.zeros(antind.size, dtype=NP.complex64)
                    # per_ant_per_freq_norm_wts = NP.ones(antind.size, dtype=NP.complex64)                    
                    
                    if parallel or (nproc is not None):
                        list_of_val = []
                        list_of_rowcol_tuple = []
                    else:
                        spval = []
                        sprow = []
                        spcol = []
                        
                    runsum = 0
                    if verbose:
                        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(n_ant), PGB.ETA()], maxval=n_ant).start()
                    for ai,gi in enumerate(indNN_list):
                        if len(gi) > 0:
                            fvu_ind = NP.asarray(gi)
                            unraveled_fvu_ind = NP.unravel_index(fvu_ind, (self.f.size,)+self.gridu.shape)
                            f_ind = unraveled_fvu_ind[0]
                            v_ind = unraveled_fvu_ind[1]
                            u_ind = unraveled_fvu_ind[2]
                            chanhist, chanbe, chanbn, chanri = OPS.binned_statistic(f_ind, statistic='count', bins=NP.arange(self.f.size+1))
                            for ci in xrange(self.f.size):
                                if chanhist[ci] > 0.0:
                                    select_chan_ind = chanri[chanri[ci]:chanri[ci+1]]
                                    per_ant_per_freq_kernel_sum = NP.sum(krn[apol][runsum:runsum+len(gi)][select_chan_ind])
                                    per_ant_per_freq_norm_wts[runsum:runsum+len(gi)][select_chan_ind] = 1.0 / per_ant_per_freq_kernel_sum

                        per_ant2grid_info = {}
                        per_ant2grid_info['label'] = self.ordered_labels[ai]
                        per_ant2grid_info['f_gridind'] = NP.copy(f_ind)
                        per_ant2grid_info['u_gridind'] = NP.copy(u_ind)
                        per_ant2grid_info['v_gridind'] = NP.copy(v_ind)
                        # per_ant2grid_info['fvu_gridind'] = NP.copy(gi)
                        per_ant2grid_info['per_ant_per_freq_norm_wts'] = per_ant_per_freq_norm_wts[runsum:runsum+len(gi)]
                        per_ant2grid_info['illumination'] = krn[apol][runsum:runsum+len(gi)]
                        self.grid_mapper[apol]['per_ant2grid'] += [copy.deepcopy(per_ant2grid_info)]
                        runsum += len(gi)

                        # determine the sparse interferometer-to-grid mapping matrix pre-requisites
                        val = per_ant2grid_info['per_ant_per_freq_norm_wts']*per_ant2grid_info['illumination']
                        vuf_gridind_unraveled = (per_ant2grid_info['v_gridind'],per_ant2grid_info['u_gridind'],per_ant2grid_info['f_gridind'])
                        vuf_gridind_raveled = NP.ravel_multi_index(vuf_gridind_unraveled, (self.gridu.shape+(self.f.size,)))
                        
                        if (not parallel) and (nproc is None):
                            spval += val.tolist()
                            sprow += vuf_gridind_raveled.tolist()
                            spcol += (per_ant2grid_info['f_gridind'] + ai*self.f.size).tolist()
                        else:
                            list_of_val += [per_ant2grid_info['per_ant_per_freq_norm_wts']*per_ant2grid_info['illumination']]
                            list_of_rowcol_tuple += [(vuf_gridind_raveled, per_ant2grid_info['f_gridind'])]
                        if verbose:
                            progress.update(ai+1)

                    if verbose:
                        progress.finish()

                    # determine the sparse interferometer-to-grid mapping matrix
                    if parallel or (nproc is not None):
                        list_of_shapes = [(self.gridu.size*self.f.size, self.f.size)] * n_ant
                        if nproc is None:
                            nproc = max(MP.cpu_count()-1, 1) 
                        else:
                            nproc = min(nproc, max(MP.cpu_count()-1, 1))
                        pool = MP.Pool(processes=nproc)
                        list_of_spmat = pool.map(genMatrixMapper_arg_splitter, IT.izip(list_of_val, list_of_rowcol_tuple, list_of_shapes))
                        self.ant2grid_mapper[apol] = SpM.hstack(list_of_spmat, format='csr')
                    else:
                        spval = NP.asarray(spval)
                        sprowcol = (NP.asarray(sprow), NP.asarray(spcol))
                        self.ant2grid_mapper[apol] = SpM.csr_matrix((spval, sprowcol), shape=(self.gridu.size*self.f.size, n_ant*self.f.size))

                    self.grid_mapper[apol]['all_ant2grid']['per_ant_per_freq_norm_wts'] = NP.copy(per_ant_per_freq_norm_wts)

    ############################################################################

    def applyMappingMatrix(self, pol=None, cal_loop=False, verbose=True):

        """
        ------------------------------------------------------------------------
        Constructs the grid of complex field illumination and electric fields 
        using the sparse antenna-to-grid mapping matrix. Intended to serve as a 
        "matrix" alternative to make_grid_cube_new() 

        Inputs:

        pol     [String] The polarization to be gridded. Can be set to 'P1' or 
                'P2'. If set to None, gridding for all the polarizations is 
                performed. Default=None
        
        cal_loop
                [boolean] If True, the calibration loop is assumed to be ON 
                and hence the calibrated electric fields are set in the 
                calibration loop. If False (default), the calibration loop is
                assumed to be OFF and the current electric fields are assumed 
                to be the calibrated data to be mapped to the grid 
                via gridding convolution.

        verbose [boolean] If True, prints diagnostic and progress messages. 
                If False (default), suppress printing such messages.
        ------------------------------------------------------------------------
        """
        
        if pol is None:
            pol = ['P1', 'P2']

        pol = NP.unique(NP.asarray(pol))
        
        for apol in pol:

            if verbose:
                print 'Gridding aperture illumination and electric fields for polarization {0} ...'.format(apol)

            if apol not in ['P1', 'P2']:
                raise ValueError('Invalid specification for input parameter pol')

            if not cal_loop:
                self.caldata[apol] = self.get_E_fields(apol, flag=None, tselect=-1, fselect=None, aselect=None, datapool='current', sort=True)
            else:
                if self.caldata[apol] is None:
                    self.caldata[apol] = self.get_E_fields(apol, flag=None, tselect=-1, fselect=None, aselect=None, datapool='current', sort=True)

            Ef = self.caldata[apol]['E-fields'].astype(NP.complex64)  #  (n_ts=1) x n_ant x nchan
            Ef = NP.squeeze(Ef, axis=0)  # n_ant x nchan

            twts = self.caldata[apol]['twts']  # (n_ts=1) x n_ant x 1
            twts = NP.squeeze(twts, axis=0)  # n_ant x 1

            Ef = Ef * twts    # applies antenna flagging, n_ant x nchan
            wts = twts * NP.ones(self.f.size).reshape(1,-1)  # n_ant x nchan

            wts[NP.isnan(Ef)] = 0.0
            Ef[NP.isnan(Ef)] = 0.0

            Ef = Ef.ravel()
            wts = wts.ravel()

            sparse_Ef = SpM.csr_matrix(Ef)
            sparse_wts = SpM.csr_matrix(wts)

            # Store as sparse matrices
            self.grid_illumination[apol] = self.ant2grid_mapper[apol].dot(sparse_wts.T)
            self.grid_Ef[apol] = self.ant2grid_mapper[apol].dot(sparse_Ef.T)

            # # Store as dense matrices
            # self.grid_illumination[apol] = self.ant2grid_mapper[apol].dot(wts).reshape(self.gridu.shape+(self.f.size,))
            # self.grid_Ef[apol] = self.ant2grid_mapper[apol].dot(Ef).reshape(self.gridu.shape+(self.f.size,))   
            
            if verbose:
                print 'Gridded aperture illumination and electric fields for polarization {0} from {1:0d} unflagged contributing antennas'.format(apol, NP.sum(twts).astype(int))

    ############################################################################

    def make_grid_cube(self, pol=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Constructs the grid of complex field illumination and electric fields 
        using the gridding information determined for every antenna. Flags are 
        taken into account while constructing this grid.

        Inputs:

        pol     [String] The polarization to be gridded. Can be set to 'P1' or 
                'P2'. If set to None, gridding for all the polarizations is 
                performed. Default=None
        
        verbose [boolean] If True, prints diagnostic and progress messages. 
                If False (default), suppress printing such messages.
        ------------------------------------------------------------------------
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
            for antlabel, antinfo in self.grid_mapper[apol]['labels'].iteritems():
                if not self.antennas[antlabel].antpol.flag[apol]:
                    num_unflagged += 1
                    gridind_unraveled = NP.unravel_index(antinfo['gridind'], self.gridu.shape+(self.f.size,))
                    self.grid_illumination[apol][gridind_unraveled] += antinfo['illumination']
                    self.grid_Ef[apol][gridind_unraveled] += antinfo['Ef']

                if verbose:
                    progress.update(loopcount+1)
                    loopcount += 1
            if verbose:
                progress.finish()
                
            if verbose:
                print 'Gridded aperture illumination and electric fields for polarization {0} from {1:0d} unflagged contributing antennas'.format(apol, num_unflagged)

    ############################################################################ 

    def make_grid_cube_new(self, pol=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Constructs the grid of complex field illumination and electric fields 
        using the gridding information determined for every antenna. Flags are 
        taken into account while constructing this grid.

        Inputs:

        pol     [String] The polarization to be gridded. Can be set to 'P1' or 
                'P2'. If set to None, gridding for all the polarizations is 
                performed. Default=None
        
        verbose [boolean] If True, prints diagnostic and progress messages. 
                If False (default), suppress printing such messages.
        ------------------------------------------------------------------------
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
    
            nlabels = len(self.grid_mapper[apol]['per_ant2grid'])
            loopcount = 0
            num_unflagged = 0
            if verbose:
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(nlabels), PGB.ETA()], maxval=nlabels).start()
            for ai,per_ant2grid_info in enumerate(self.grid_mapper[apol]['per_ant2grid']):
                antlabel = per_ant2grid_info['label']
                if not self.antennas[antlabel].antpol.flag[apol]:
                    num_unflagged += 1
                    vuf_gridind_unraveled = (per_ant2grid_info['v_gridind'],per_ant2grid_info['u_gridind'],per_ant2grid_info['f_gridind'])
                    self.grid_illumination[apol][vuf_gridind_unraveled] += per_ant2grid_info['per_ant_per_freq_norm_wts'] * per_ant2grid_info['illumination']
                    self.grid_Ef[apol][vuf_gridind_unraveled] += per_ant2grid_info['per_ant_per_freq_norm_wts'] * per_ant2grid_info['Ef'] * per_ant2grid_info['illumination']

                if verbose:
                    progress.update(loopcount+1)
                    loopcount += 1
            if verbose:
                progress.finish()
                
            if verbose:
                print 'Gridded aperture illumination and electric fields for polarization {0} from {1:0d} unflagged contributing antennas'.format(apol, num_unflagged)

    ############################################################################ 

    def evalAntennaPairCorrWts(self, label1, label2=None, forceeval=False):

        """
        ------------------------------------------------------------------------
        Evaluate correlation of pair of antenna illumination weights on grid. 
        It will be computed only if it was not computed or stored in attribute 
        pairwise_typetag_crosswts_vuf earlier

        Inputs:

        label1  [string] Label of first antenna. Must be specified (no default)

        label2  [string] Label of second antenna. If specified as None 
                (default), it will be set equal to label1 in which case the
                auto-correlation of antenna weights is evaluated

        forceeval 
                [boolean] When set to False (default) the correlation in
                the UV plane is not evaluated if it was already evaluated 
                earlier. If set to True, it will be forcibly evaluated 
                independent of whether they were already evaluated or not
        ------------------------------------------------------------------------
        """

        try:
            label1
        except NameError:
            raise NameError('Input label1 must be specified')

        if label1 not in self.antennas:
            raise KeyError('Input label1 not found in current instance of class AntennaArray')

        if label2 is None:
            label2 = label1

        if label2 not in self.antennas:
            raise KeyError('Input label2 not found in current instance of class AntennaArray')

        if (label1, label2) in self.antenna_pair_to_typetag:
            typetag_pair = self.antenna_pair_to_typetag[(label1,label2)]
        elif (label2, label1) in self.antenna_pair_to_typetag:
            typetag_pair = self.antenna_pair_to_typetag[(label2,label1)]
        else:
            raise KeyError('Antenna pair not found in attribute antenna_pair_to_type. Needs debugging')

        typetag1, typetag2 = typetag_pair
        if forceeval or (typetag_pair not in self.pairwise_typetag_crosswts_vuf):
            pol = ['P1', 'P2']
            self.pairwise_typetag_crosswts_vuf[typetag_pair] = {}
            du = self.gridu[0,1] - self.gridu[0,0]
            dv = self.gridv[1,0] - self.gridv[0,0]
            if (typetag1 == typetag2) and (self.antennas[label1].aperture.kernel_type['P1'] == 'func') and (self.antennas[label1].aperture.kernel_type['P2'] == 'func'):
                gridu, gridv = NP.meshgrid(du*(NP.arange(2*self.gridu.shape[1])-self.gridu.shape[1]), dv*(NP.arange(2*self.gridu.shape[0])-self.gridu.shape[0]))
                wavelength = FCNST.c / self.f
                min_lambda = NP.abs(wavelength).min()
                rmaxNN = 0.5 * NP.sqrt(du**2 + dv**2) * min_lambda 
                gridx = gridu[:,:,NP.newaxis] * wavelength.reshape(1,1,-1)
                gridy = gridv[:,:,NP.newaxis] * wavelength.reshape(1,1,-1)
                gridxy = NP.hstack((gridx.reshape(-1,1), gridy.reshape(-1,1)))
                wl = NP.ones(gridu.shape)[:,:,NP.newaxis] * wavelength.reshape(1,1,-1)
                ant_aprtr = copy.deepcopy(self.antennas[label1].aperture)
                pol_type = 'dual'
                kerntype = ant_aprtr.kernel_type
                shape = ant_aprtr.shape
                kernshapeparms = {p: {'xmax': ant_aprtr.xmax[p], 'ymax': ant_aprtr.ymax[p], 'rmax': ant_aprtr.rmax[p], 'rmin': ant_aprtr.rmin[p], 'rotangle': ant_aprtr.rotangle[p]} for p in pol}
                for p in pol:
                    if shape[p] == 'rect':
                        shape[p] = 'auto_convolved_rect'
                    elif shape[p] == 'square':
                        shape[p] = 'auto_convolved_square'
                    elif shape[p] == 'circular':
                        shape[p] = 'auto_convolved_circular'
                    else:
                        raise ValueError('Aperture kernel footprint shape - {0} - currently unsupported'.format(shape[p]))
                        
                aprtr = APR.Aperture(pol_type=pol_type, kernel_type=kerntype,
                                     shape=shape, parms=kernshapeparms,
                                     lkpinfo=None, load_lookup=True)
                
                max_aprtr_size = max([NP.sqrt(aprtr.xmax['P1']**2 + NP.sqrt(aprtr.ymax['P1']**2)), NP.sqrt(aprtr.xmax['P2']**2 + NP.sqrt(aprtr.ymax['P2']**2)), aprtr.rmax['P1'], aprtr.rmax['P2']])
                distNN = 2.0 * max_aprtr_size
                indNN_list, blind, vuf_gridind = LKP.find_NN(NP.zeros(2).reshape(1,-1), gridxy, distance_ULIM=distNN, flatten=True, parallel=False)
                dxy = gridxy[vuf_gridind,:]
                unraveled_vuf_ind = NP.unravel_index(vuf_gridind, gridu.shape+(self.f.size,))
                unraveled_vu_ind = (unraveled_vuf_ind[0], unraveled_vuf_ind[1])
                raveled_vu_ind = NP.ravel_multi_index(unraveled_vu_ind, (gridu.shape[0], gridu.shape[1]))
                for p in pol:
                    krn = aprtr.compute(dxy, wavelength=wl.ravel()[vuf_gridind], pol=p, rmaxNN=rmaxNN, load_lookup=False)
                    krn_sparse = SpM.csr_matrix((krn[p], (raveled_vu_ind,)+(unraveled_vuf_ind[2],)), shape=(gridu.size,)+(self.f.size,), dtype=NP.complex64)
                    krn_sparse_sumuv = krn_sparse.sum(axis=0)
                    krn_sparse_norm = krn_sparse.A / krn_sparse_sumuv.A
                    sprow = raveled_vu_ind
                    spcol = unraveled_vuf_ind[2]
                    spval = krn_sparse_norm[(sprow,)+(spcol,)]
                    self.pairwise_typetag_crosswts_vuf[typetag_pair][p] = SpM.csr_matrix((spval, (sprow,)+(spcol,)), shape=(gridu.size,)+(self.f.size,), dtype=NP.complex64)
            else:
                ulocs = du*(NP.arange(2*self.gridu.shape[1])-self.gridu.shape[1])
                vlocs = dv*(NP.arange(2*self.gridu.shape[0])-self.gridu.shape[0])
                antenna_grid_wts_vuf_1 = self.antennas[label1].evalGridIllumination(uvlocs=(ulocs, vlocs), xy_center=NP.zeros(2))
                shape_tuple = (vlocs.size, ulocs.size) + (self.f.size,)
                eps = 1e-10
                if label1 == label2:
                    for p in pol:
                        sum_wts1 = antenna_grid_wts_vuf_1[p].sum(axis=0).A
                        sum_wts = NP.abs(sum_wts1)**2
                        antpair_beam = NP.abs(NP.fft.fft2(antenna_grid_wts_vuf_1[p].toarray().reshape(shape_tuple), axes=(0,1)))**2
                        antpair_grid_wts_vuf = NP.fft.ifft2(antpair_beam/sum_wts[NP.newaxis,:,:], axes=(0,1)) # Inverse FFT
                        antpair_grid_wts_vuf = NP.fft.ifftshift(antpair_grid_wts_vuf, axes=(0,1))
                        antpair_grid_wts_vuf[NP.abs(antpair_grid_wts_vuf) < eps] = 0.0
                        self.pairwise_typetag_crosswts_vuf[typetag_pair][p] = SpM.csr_matrix(antpair_grid_wts_vuf.reshape(-1,self.f.size))
                else:
                    antenna_grid_wts_vuf_2 = self.antennas[label2].evalGridIllumination(uvlocs=(ulocs, vlocs), xy_center=NP.zeros(2))
                    for p in pol:
                        sum_wts1 = antenna_grid_wts_vuf_1[p].sum(axis=0).A
                        sum_wts2 = antenna_grid_wts_vuf_2[p].sum(axis=0).A
                        sum_wts = sum_wts1 * sum_wts2.conj()
                        antpair_beam = NP.fft.fft2(antenna_grid_wts_vuf_1[p].toarray().reshape(shape_tuple), axes=(0,1)) * NP.fft.fft2(antenna_grid_wts_vuf_1[p].toarray().reshape(shape_tuple).conj(), axes=(0,1))
                        antpair_grid_wts_vuf = NP.fft.ifft2(antpair_beam/sum_wts[NP.newaxis,:,:], axes=(0,1)) # Inverse FFT
                        antpair_grid_wts_vuf = NP.fft.ifftshift(antpair_grid_wts_vuf, axes=(0,1))
                        antpair_grid_wts_vuf[NP.abs(antpair_grid_wts_vuf) < eps] = 0.0
                        self.pairwise_typetag_crosswts_vuf[typetag_pair][p] = SpM.csr_matrix(antpair_grid_wts_vuf.reshape(-1,self.f.size))
        else:
            print 'Specified antenna pair correlation weights have already been evaluated'

    ############################################################################ 

    def evalAntennaAutoCorrWts(self, forceeval=False):

        """
        ------------------------------------------------------------------------
        Evaluate auto-correlation of aperture illumination of each antenna on
        the UVF-plane

        Inputs:

        forceeval [boolean] When set to False (default) the auto-correlation in
                  the UV plane is not evaluated if it was already evaluated 
                  earlier. If set to True, it will be forcibly evaluated 
                  independent of whether they were already evaluated or not
        ------------------------------------------------------------------------
        """

        if forceeval or (not self.antenna_autocorr_set):
            self.antenna_autocorr_set = False
            for antkey in self.antennas:
                self.evalAntennaPairCorrWts(antkey, label2=None, forceeval=forceeval)
            self.antenna_autocorr_set = True
            
    ############################################################################ 

    def evalAllAntennaPairCorrWts(self, forceeval=False):

        """
        ------------------------------------------------------------------------
        Evaluate zero-centered cross-correlation of aperture illumination of 
        each antenna pair on the UVF-plane

        Inputs:

        forceeval [boolean] When set to False (default) the zero-centered 
                  cross-correlation of antenna illumination weights on
                  the UV plane is not evaluated if it was already evaluated 
                  earlier. If set to True, it will be forcibly evaluated 
                  independent of whether they were already evaluated or not
        ------------------------------------------------------------------------
        """

        if forceeval or (not self.antenna_crosswts_set):
            for label_pair in self.antenna_pair_to_typetag:
                label1, label2 = label_pair
                self.evalAntennaPairCorrWts(label1, label2=label2, forceeval=forceeval)
            self.antenna_crosswts_set = True
            
    ############################################################################ 

    def makeAutoCorrCube(self, pol=None, data=None, datapool='stack',
                         tbinsize=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Constructs the grid of antenna aperture illumination auto-correlation 
        using the gridding information determined for every antenna. Flags are 
        taken into account while constructing this grid

        Inputs:

        pol     [String] The polarization to be gridded. Can be set to 'P1' or 
                'P2'. If set to None, gridding for all the polarizations is 
                performed. Default=None
        
        data    [dictionary] dictionary containing data that will be used to
                determine the auto-correlations of antennas. This will be used 
                only if input datapool is set to 'custom'. It consists of the 
                following keys and information:
                'labels'    Contains a numpy array of strings of antenna 
                            labels
                'data'      auto-correlated electric fields 
                            (n_ant x nchan array)

        datapool 
                [string] Specifies whether data to be used in determining the
                auto-correlation the E-fields to be used come from
                'stack' (default), 'current', 'avg' or 'custom'. If set to
                'custom', the data provided in input data will be used. 
                Otherwise squared electric fields will be used if set to 
                'current' or 'stack', and averaged squared electric fields if
                set to 'avg'

        tbinsize 
                [scalar or dictionary] Contains bin size of timestamps while
                averaging. Only used when datapool is set to 'avg' and if the 
                attribute auto_corr_data does not contain the key 'avg'. In 
                that case, default = None means all antenna E-field 
                auto-correlation spectra over all timestamps are averaged. If 
                scalar, the same (positive) value applies to all polarizations. 
                If dictionary, timestamp bin size (positive) in seconds is 
                provided under each key 'P1' and 'P2'. If any of the keys is 
                missing the auto-correlated antenna E-field spectra for that 
                polarization are averaged over all timestamps.

        verbose [boolean] If True, prints diagnostic and progress messages. 
                If False (default), suppress printing such messages.

        Outputs:

        Tuple (autocorr_wts_cube, autocorr_data_cube). autocorr_wts_cube is a
        dictionary with polarization keys 'P1' and 'P2. Under each key is a 
        matrix of size nt x nv x nu x nchan. autocorr_data_cube is also a 
        dictionary with polarization keys 'P1' and 'P2. Under each key is a 
        matrix of size nt x nv x nu x nchan where nt=1, nt=n_timestamps,
        or nt=n_tavg if datapool is set to 'current', 'stack' or 'avg'
        respectively
        ------------------------------------------------------------------------
        """
        
        if pol is None:
            pol = ['P1', 'P2']

        pol = NP.unique(NP.asarray(pol))
        
        if datapool not in ['stack', 'current', 'avg', 'custom']:
            raise ValueError('Input datapool must be set to "stack" or "current"')

        if not self.antenna_autocorr_set:
            self.evalAntennaAutoCorrWts()

        data_info = {}
        if datapool in ['current', 'stack', 'avg']:
            if datapool not in self.auto_corr_data:
                self.evalAutoCorr(datapool=datapool, tbinsize=tbinsize)
            for apol in pol:
                data_info[apol] = {'labels': self.auto_corr_data[datapool][apol]['labels'], 'twts': self.auto_corr_data[datapool][apol]['twts'], 'data': self.auto_corr_data[datapool][apol]['E-fields']}
        else:
            if not isinstance(data, dict):
                raise TypeError('Input data must be a dictionary')
            for apol in pol:
                if apol not in data:
                    raise KeyError('Key {)} not found in input data'.format(apol))
                if not isinstance(data[apol], dict):
                    raise TypeError('Value under polarization key "{0}" under input data must be a dictionary'.format(apol))
                if ('labels' not in data[apol]) or ('data' not in data[apol]):
                    raise KeyError('Keys "labels" and "data" not found under input data[{0}]'.format(apol))

        autocorr_wts_cube = {p: None for p in ['P1', 'P2']}
        autocorr_data_cube = {p: None for p in ['P1', 'P2']}
        for apol in pol:
            if verbose:
                print 'Gridding auto-correlation of aperture illumination and electric fields for polarization {0} ...'.format(apol)

            if apol not in ['P1', 'P2']:
                raise ValueError('Invalid specification for input parameter pol')

            for antind, antkey in enumerate(data_info[apol]['labels']):
                typetag_pair = self.antenna_pair_to_typetag[(antkey,antkey)]
                shape_tuple = tuple(2*NP.asarray(self.gridu.shape))+(self.f.size,)
                if autocorr_wts_cube[apol] is None:
                    autocorr_wts_cube[apol] = self.pairwise_typetag_crosswts_vuf[typetag_pair][apol].toarray().reshape(shape_tuple)[NP.newaxis,:,:,:] * data_info[apol]['twts'][:,antind,:][:,NP.newaxis,NP.newaxis,:] # nt x nv x nu x nchan
                    autocorr_data_cube[apol] = self.pairwise_typetag_crosswts_vuf[typetag_pair][apol].toarray().reshape(shape_tuple)[NP.newaxis,:,:,:] * data_info[apol]['twts'][:,antind,:][:,NP.newaxis,NP.newaxis,:] * data_info[apol]['data'][:,antind,:][:,NP.newaxis,NP.newaxis,:] # nt x nv x nu x nchan
                else:
                    autocorr_wts_cube[apol] += self.pairwise_typetag_crosswts_vuf[typetag_pair][apol].toarray().reshape(shape_tuple)[NP.newaxis,:,:,:] * data_info[apol]['twts'][:,antind,:][:,NP.newaxis,NP.newaxis,:] # nt x nv x nu x nchan
                    autocorr_data_cube[apol] += self.pairwise_typetag_crosswts_vuf[typetag_pair][apol].toarray().reshape(shape_tuple)[NP.newaxis,:,:,:] * data_info[apol]['twts'][:,antind,:][:,NP.newaxis,NP.newaxis,:] * data_info[apol]['data'][:,antind,:][:,NP.newaxis,NP.newaxis,:] # nt x nv x nu x nchan
            sum_wts = NP.sum(data_info[apol]['twts'], axis=1) # nt x 1
            autocorr_wts_cube[apol] = autocorr_wts_cube[apol] / sum_wts[:,NP.newaxis,NP.newaxis,:] # nt x nv x nu x nchan
            autocorr_data_cube[apol] = autocorr_data_cube[apol] / sum_wts[:,NP.newaxis,NP.newaxis,:] # nt x nv x nu x nchan

        return (autocorr_wts_cube, autocorr_data_cube)
                    
    ############################################################################

    def makeCrossCorrWtsCube(self, pol=None, data=None, datapool='stack',
                             verbose=True):

        """
        ------------------------------------------------------------------------
        Constructs the grid of zero-centered cross-correlation of antenna 
        aperture pairs using the gridding information determined for every 
        antenna. Flags are taken into account while constructing this grid

        Inputs:

        pol     [String] The polarization to be gridded. Can be set to 'P1' or 
                'P2'. If set to None, gridding for all the polarizations is 
                performed. Default=None
        
        datapool 
                [string] Specifies whether flags that come from data to be 
                used in determining the zero-centered cross-correlation come 
                from 'stack' (default), 'current', or 'avg'. 

        verbose [boolean] If True, prints diagnostic and progress messages. 
                If False (default), suppress printing such messages.

        Outputs:

        centered_crosscorr_wts_vuf is a dictionary with polarization keys 
        'P1' and 'P2. Under each key is a sparse matrix of size 
        (nv x nu) x nchan. 
        ------------------------------------------------------------------------
        """
        
        if pol is None:
            pol = ['P1', 'P2']

        pol = NP.unique(NP.asarray(pol))
        
        if datapool not in ['stack', 'current', 'avg', 'custom']:
            raise ValueError('Input datapool must be set to "stack" or "current"')

        if not self.antenna_crosswts_set:
            self.evalAllAntennaPairCorrWts()

        centered_crosscorr_wts_cube = {p: None for p in ['P1', 'P2']}
        for apol in pol:
            if verbose:
                print 'Gridding centered cross-correlation of aperture illumination for polarization {0} ...'.format(apol)

            if apol not in ['P1', 'P2']:
                raise ValueError('Invalid specification for input parameter pol')

            for typetag_pair in self.pairwise_typetags:
                if 'cross' in self.pairwise_typetags[typetag_pair]:
                    n_bl = len(self.pairwise_typetags[typetag_pair]['cross'])
                    if centered_crosscorr_wts_cube[apol] is None:
                        centered_crosscorr_wts_cube[apol] = n_bl * self.pairwise_typetag_crosswts_vuf[typetag_pair][apol]
                    else:
                        centered_crosscorr_wts_cube[apol] += n_bl * self.pairwise_typetag_crosswts_vuf[typetag_pair][apol]

        return centered_crosscorr_wts_cube
                    
    ############################################################################
    
    def evalAntennaPairPBeam(self, typetag_pair=None, label_pair=None,
                             pad=0, skypos=None):

        """
        ------------------------------------------------------------------------
        Evaluate power pattern response on sky of an antenna pair

        Inputs:

        typetag_pair    
                    [dictionary] dictionary with two keys '1' and '2' denoting
                    the antenna typetag. At least one of them must be specified.
                    If one of them is not specified, it is assumed to be the
                    same as the other. Only one of the inputs typetag_pair or 
                    label_pair must be set

        label_pair  [dictionary] dictionary with two keys '1' and '2' denoting 
                    the antenna label. At least one of them must be specified.
                    If one of them is not specified, it is assumed to be the
                    same as the other. Only one of the inputs typetag_pair or 
                    label_pair must be set

        pad         [integer] indicates the amount of padding before estimating
                    power pattern. Applicable only when skypos is set to None. 
                    The output power pattern will be of size 2**pad-1 times the 
                    size of the UV-grid along l- and m-axes. Value must 
                    not be negative. Default=0 (implies no padding). pad=1 
                    implies padding by factor 2 along u- and v-axes

        skypos      [numpy array] Positions on sky at which power pattern is 
                    to be esimated. It is a 2- or 3-column numpy array in 
                    direction cosine coordinates. It must be of size nsrc x 2 
                    or nsrc x 3. If set to None (default), the power pattern is 
                    estimated over a grid on the sky. If a numpy array is
                    specified, then power pattern at the given locations is 
                    estimated.

        Outputs:

        pbinfo is a dictionary with the following keys and values:
        'pb'    [dictionary] Dictionary with keys 'P1' and 'P2' for 
                polarization. Under each key is a numpy array of estimated 
                power patterns. If skypos was set to None, the numpy array is 
                3D masked array of size nm x nl x nchan. The mask is based on 
                which parts of the grid are valid direction cosine coordinates 
                on the sky. If skypos was a numpy array denoting specific sky 
                locations, the value in this key is a 2D numpy array of size 
                nsrc x nchan
        'llocs' [None or numpy array] If the power pattern estimated is a grid
                (if input skypos was set to None), it contains the l-locations
                of the grid on the sky. If input skypos was not set to None, 
                the value under this key is set to None
        'mlocs' [None or numpy array] If the power pattern estimated is a grid
                (if input skypos was set to None), it contains the m-locations
                of the grid on the sky. If input skypos was not set to None, 
                the value under this key is set to None
        ------------------------------------------------------------------------
        """

        if (typetag_pair is None) and (label_pair is None):
            raise ValueError('One of the inputs typetag_pair or label_pair must be specified')
        elif (typetag_pair is not None) and (label_pair is not None):
            raise ValueError('Only one of the inputs typetag_pair or label_pair must be specified')

        if typetag_pair is not None:
            if ('1' not in typetag_pair) and ('2' not in typetag_pair):
                raise KeyError('Required keys not found in input typetag_pair')
            elif ('1' not in typetag_pair) and ('2' in typetag_pair):
                typetag_pair['1'] = typetag_pair['2']
            elif ('1' in typetag_pair) and ('2' not in typetag_pair):
                typetag_pair['2'] = typetag_pair['1']
            typetag_tuple = (typetag_pair['1'], typetag_pair['2'])
            if typetag_tuple not in self.pairwise_typetags:
                if typetag_tuple[::-1] not in self.pairwise_typetags:
                    raise KeyError('typetag pair not found in antenna cross weights')
                else:
                    typetag_tuple = typetag_tuple[::-1]
            if 'auto' in self.pairwise_typetags[typetag_tuple]:
                label1, label2 = list(self.pairwise_typetags[typetag_tuple]['auto'])[0]
            else:
                label1, label2 = list(self.pairwise_typetags[typetag_tuple]['cross'])[0]
        else:
            if ('1' not in label_pair) and ('2' not in label_pair):
                raise KeyError('Required keys not found in input label_pair')
            elif ('1' not in label_pair) and ('2' in label_pair):
                label_pair['1'] = label_pair['2']
            elif ('1' in label_pair) and ('2' not in label_pair):
                label_pair['2'] = label_pair['1']
            label1 = label_pair['1']
            label2 = label_pair['2']
            label_tuple = (label1, label2)
            if label_tuple not in self.antenna_pair_to_typetag:
                if label_tuple[::-1] not in self.antenna_pair_to_typetag:
                    raise KeyError('label pair not found in antenna pairs')
                else:
                    label_tuple = label_tuple[::-1]
            label1, label2 = label_tuple
            typetag_tuple = self.antenna_pair_to_typetag[label_tuple]

        if typetag_tuple not in self.pairwise_typetag_crosswts_vuf:
            self.evalAntennaPairCorrWts(label1, label2=label2)
        centered_crosscorr_wts_vuf = self.pairwise_typetag_crosswts_vuf[typetag_tuple]

        du = self.gridu[0,1] - self.gridu[0,0]
        dv = self.gridv[1,0] - self.gridv[0,0]
        ulocs = du*(NP.arange(2*self.gridu.shape[1])-self.gridu.shape[1])
        vlocs = dv*(NP.arange(2*self.gridv.shape[0])-self.gridv.shape[0])        
        
        pol = ['P1', 'P2']
        pbinfo = {'pb': {}}
        for p in pol:
            pb = evalCorrWts2PB(centered_crosscorr_wts_vuf[p], ulocs, vlocs, pad=pad, skypos=skypos)
            pbinfo['pb'][p] = pb['pb']
            pbinfo['llocs'] = pb['llocs']
            pbinfo['mlocs'] = pb['mlocs']

        return pbinfo

    ############################################################################ 

    def quick_beam_synthesis(self, pol=None, keep_zero_spacing=True):
        
        """
        ------------------------------------------------------------------------
        A quick generator of synthesized beam using antenna array field 
        illumination pattern using the center frequency. Not intended to be used
        rigorously but rather for comparison purposes and making quick plots

        Inputs:

        pol     [String] The polarization of the synthesized beam. Can be set 
                to 'P1' or 'P2'. If set to None, synthesized beam for all the 
                polarizations are generated. Default=None

        keep_zero_spacing
                [boolean] If set to True (default), keep the zero spacing in
                uv-plane grid illumination and as a result the average value
                of the synthesized beam could be non-zero. If False, the zero
                spacing is forced to zero by removing the average value fo the
                synthesized beam

        Outputs:

        Dictionary with the following keys and information:

        'syn_beam'  [numpy array] synthesized beam of size twice as that of the 
                    antenna array grid. It is FFT-shifted to place the 
                    origin at the center of the array. The peak value of the 
                    synthesized beam is fixed at unity

        'grid_power_illumination'
                    [numpy array] complex grid illumination obtained from 
                    inverse fourier transform of the synthesized beam in 
                    'syn_beam' and has size twice as that of the antenna 
                    array grid. It is FFT-shifted to have the origin at the 
                    center. The sum of this array is set to unity to match the 
                    peak of the synthesized beam

        'l'         [numpy vector] x-values of the direction cosine grid 
                    corresponding to x-axis (axis=1) of the synthesized beam

        'm'         [numpy vector] y-values of the direction cosine grid 
                    corresponding to y-axis (axis=0) of the synthesized beam
        ------------------------------------------------------------------------
        """

        if not self.grid_ready:
            raise ValueError('Need to perform gridding of the antenna array before an equivalent UV grid can be simulated')

        if pol is None:
            pol = ['P1', 'P2']
        elif isinstance(pol, str):
            if pol in ['P1', 'P2']:
                pol = [pol]
            else:
                raise ValueError('Invalid polarization specified')
        elif isinstance(pol, list):
            p = [apol for apol in pol if apol in ['P1', 'P2']]
            if len(p) == 0:
                raise ValueError('Invalid polarization specified')
            pol = p
        else:
            raise TypeError('Input keyword pol must be string, list or set to None')

        pol = sorted(pol)

        for apol in pol:
            if self.grid_illumination[apol] is None:
                raise ValueError('Grid illumination for the specified polarization is not determined yet. Must use make_grid_cube()')

        chan = NP.argmin(NP.abs(self.f - self.f0))
        grid_field_illumination = NP.empty(self.gridu.shape+(len(pol),), dtype=NP.complex)
        for pind, apol in enumerate(pol):
            grid_field_illumination[:,:,pind] = self.grid_illumination[apol][:,:,chan]

        syn_beam = NP.fft.fft2(grid_field_illumination, s=[4*self.gridu.shape[0], 4*self.gridv.shape[1]], axes=(0,1))
        syn_beam = NP.abs(syn_beam)**2

        if not keep_zero_spacing:
            dclevel = NP.sum(syn_beam, axis=(0,1), keepdims=True) / (1.0*syn_beam.size/len(pol))
            syn_beam = syn_beam - dclevel

        syn_beam /= syn_beam.max()  # Normalize to get unit peak for PSF
        syn_beam_in_uv = NP.fft.ifft2(syn_beam, axes=(0,1)) # Inverse FT
        du = self.gridu[0,1] - self.gridu[0,0]
        dv = self.gridv[1,0] - self.gridv[0,0]
        # if not keep_zero_spacing:  # Filter out the interferometer aperture kernel footprint centered on zero
        #     l4 = DSP.spectax(4*self.gridu.shape[1], resolution=du, shift=False)
        #     m4 = DSP.spectax(4*self.gridv.shape[0], resolution=dv, shift=False)
        #     u4 = DSP.spectax(l4.size, resolution=l4[1]-l4[0], shift=False)
        #     v4 = DSP.spectax(m4.size, resolution=m4[1]-m4[0], shift=False)
        #     gridu4, gridv4 = NP.meshgrid(u4,v4)
        #     gridxy4 = NP.hstack((gridu4.reshape(-1,1), gridv4.reshape(-1,1))) * FCNST.c/self.f[chan]
    
        #     # assume identical antennas
        #     aperture = self.antennas.itervalues().next().aperture
        #     zero_vind = []
        #     zero_uind = []
        #     zero_pind = []
        #     for pi,apol in enumerate(pol):
        #         if aperture.kernel_type[apol] == 'func':
        #             if aperture.shape[apol] == 'circular':
        #                 z_ind = NP.where(NP.sqrt(NP.sum(gridxy4**2, axis=1)) <= 2*aperture.rmax[apol])[0]
        #             else:
        #                 rotang = aperture.rotangle[apol]
        #                 rotmat = NP.asarray([[NP.cos(-rotang), -NP.sin(-rotang)],
        #                                      [NP.sin(-rotang),  NP.cos(-rotang)]])
        #                 gridxy4 = NP.dot(gridxy4, rotmat.T)
        #                 if aperture.shape[apol] == 'square':
        #                     z_ind = NP.where(NP.logical_and(NP.abs(gridxy4[:,0]) <= 2*aperture.xmax[apol], NP.abs(gridxy4[:,1]) <= 2*aperture.xmax[apol]))[0]
        #                 else:
        #                     z_ind = NP.where(NP.logical_and(NP.abs(gridxy4[:,0]) <= 2*aperture.xmax[apol], NP.abs(gridxy4[:,1]) <= 2*aperture.ymax[apol]))[0]
                
        #         z_vind, z_uind = NP.unravel_index(z_ind, gridu4.shape)
        #         zero_vind += z_vind.tolist()
        #         zero_uind += z_uind.tolist()
        #         zero_pind += [pi]*z_vind.size
        #     zero_vind = NP.asarray(zero_vind).ravel()
        #     zero_uind = NP.asarray(zero_uind).ravel()
        #     zero_pind = NP.asarray(zero_pind).ravel()            
        #     syn_beam_in_uv[(zero_vind, zero_uind, zero_pind)] = 0.0
        #     syn_beam = NP.fft.fft2(syn_beam_in_uv, axes=(0,1))  # FT
        #     if NP.abs(syn_beam.imag).max() > 1e-10:
        #         raise ValueError('Synthesized beam after zero spacing aperture removal has significant imaginary component')
        #     else:
        #         syn_beam = syn_beam.real
        #         norm_factor = 1.0 / syn_beam.max()
        #         syn_beam *= norm_factor  # Normalize to get unit peak for PSF
        #         syn_beam_in_uv *= norm_factor  # Normalize to get unit peak for PSF
        
        # shift the array to be centered
        syn_beam_in_uv = NP.fft.ifftshift(syn_beam_in_uv, axes=(0,1)) # Shift array to be centered

        # Discard pads at either end and select only the central values of twice the original size
        syn_beam_in_uv = syn_beam_in_uv[grid_field_illumination.shape[0]:3*grid_field_illumination.shape[0],grid_field_illumination.shape[1]:3*grid_field_illumination.shape[1],:]
        syn_beam = NP.fft.fftshift(syn_beam[::2,::2,:], axes=(0,1))  # Downsample by factor 2 to get native resolution and shift to be centered
        
        l = DSP.spectax(2*self.gridu.shape[1], resolution=du, shift=True)
        m = DSP.spectax(2*self.gridv.shape[0], resolution=dv, shift=True)

        return {'syn_beam': syn_beam, 'grid_power_illumination': syn_beam_in_uv, 'l': l, 'm': m}

    ############################################################################ 

    def quick_beam_synthesis_new(self, pol=None, keep_zero_spacing=True):
        
        """
        ------------------------------------------------------------------------
        A quick generator of synthesized beam using antenna array field 
        illumination pattern using the center frequency. Not intended to be used
        rigorously but rather for comparison purposes and making quick plots

        Inputs:

        pol     [String] The polarization of the synthesized beam. Can be set 
                to 'P1' or 'P2'. If set to None, synthesized beam for all the 
                polarizations are generated. Default=None

        keep_zero_spacing
                [boolean] If set to True (default), keep the zero spacing in
                uv-plane grid illumination and as a result the average value
                of the synthesized beam could be non-zero. If False, the zero
                spacing is forced to zero by removing the average value fo the
                synthesized beam

        Outputs:

        Dictionary with the following keys and information:

        'syn_beam'  [numpy array] synthesized beam of size twice as that of the 
                    antenna array grid. It is FFT-shifted to place the 
                    origin at the center of the array. The peak value of the 
                    synthesized beam is fixed at unity

        'grid_power_illumination'
                    [numpy array] complex grid illumination obtained from 
                    inverse fourier transform of the synthesized beam in 
                    'syn_beam' and has size twice as that of the antenna 
                    array grid. It is FFT-shifted to have the origin at the 
                    center. The sum of this array is set to unity to match the 
                    peak of the synthesized beam

        'l'         [numpy vector] x-values of the direction cosine grid 
                    corresponding to x-axis (axis=1) of the synthesized beam

        'm'         [numpy vector] y-values of the direction cosine grid 
                    corresponding to y-axis (axis=0) of the synthesized beam
        ------------------------------------------------------------------------
        """

        if not self.grid_ready:
            raise ValueError('Need to perform gridding of the antenna array before an equivalent UV grid can be simulated')

        if pol is None:
            pol = ['P1', 'P2']
        elif isinstance(pol, str):
            if pol in ['P1', 'P2']:
                pol = [pol]
            else:
                raise ValueError('Invalid polarization specified')
        elif isinstance(pol, list):
            p = [apol for apol in pol if apol in ['P1', 'P2']]
            if len(p) == 0:
                raise ValueError('Invalid polarization specified')
            pol = p
        else:
            raise TypeError('Input keyword pol must be string, list or set to None')

        pol = sorted(pol)

        for apol in pol:
            if self.grid_illumination[apol] is None:
                raise ValueError('Grid illumination for the specified polarization is not determined yet. Must use make_grid_cube()')

        chan = NP.argmin(NP.abs(self.f - self.f0))
        grid_field_illumination = NP.empty(self.gridu.shape+(len(pol),), dtype=NP.complex)
        for pind, apol in enumerate(pol):
            grid_field_illumination[:,:,pind] = self.grid_illumination[apol][:,:,chan]

        syn_beam = NP.fft.fft2(grid_field_illumination, s=[4*self.gridu.shape[0], 4*self.gridv.shape[1]], axes=(0,1))
        syn_beam = NP.abs(syn_beam)**2

        # if not keep_zero_spacing:
        #     dclevel = NP.sum(syn_beam, axis=(0,1), keepdims=True) / (1.0*syn_beam.size/len(pol))
        #     syn_beam = syn_beam - dclevel

        syn_beam /= syn_beam.max()  # Normalize to get unit peak for PSF
        syn_beam_in_uv = NP.fft.ifft2(syn_beam, axes=(0,1)) # Inverse FT
        norm_factor = 1.0

        du = self.gridu[0,1] - self.gridu[0,0]
        dv = self.gridv[1,0] - self.gridv[0,0]
        if not keep_zero_spacing:  # Filter out the interferometer aperture kernel footprint centered on zero
            l4 = DSP.spectax(4*self.gridu.shape[1], resolution=du, shift=False)
            m4 = DSP.spectax(4*self.gridv.shape[0], resolution=dv, shift=False)
            u4 = DSP.spectax(l4.size, resolution=l4[1]-l4[0], shift=False)
            v4 = DSP.spectax(m4.size, resolution=m4[1]-m4[0], shift=False)
            gridu4, gridv4 = NP.meshgrid(u4,v4)
            gridxy4 = NP.hstack((gridu4.reshape(-1,1), gridv4.reshape(-1,1))) * FCNST.c/self.f[chan]
    
            # assume identical antennas
            aperture = self.antennas.itervalues().next().aperture
            zero_vind = []
            zero_uind = []
            zero_pind = []
            for pi,apol in enumerate(pol):
                if aperture.kernel_type[apol] == 'func':
                    if aperture.shape[apol] == 'circular':
                        z_ind = NP.where(NP.sqrt(NP.sum(gridxy4**2, axis=1)) <= 2*aperture.rmax[apol])[0]
                    else:
                        rotang = aperture.rotangle[apol]
                        rotmat = NP.asarray([[NP.cos(-rotang), -NP.sin(-rotang)],
                                             [NP.sin(-rotang),  NP.cos(-rotang)]])
                        gridxy4 = NP.dot(gridxy4, rotmat.T)
                        if aperture.shape[apol] == 'square':
                            z_ind = NP.where(NP.logical_and(NP.abs(gridxy4[:,0]) <= 2*aperture.xmax[apol], NP.abs(gridxy4[:,1]) <= 2*aperture.xmax[apol]))[0]
                        else:
                            z_ind = NP.where(NP.logical_and(NP.abs(gridxy4[:,0]) <= 2*aperture.xmax[apol], NP.abs(gridxy4[:,1]) <= 2*aperture.ymax[apol]))[0]
                
                z_vind, z_uind = NP.unravel_index(z_ind, gridu4.shape)
                zero_vind += z_vind.tolist()
                zero_uind += z_uind.tolist()
                zero_pind += [pi]*z_vind.size
            zero_vind = NP.asarray(zero_vind).ravel()
            zero_uind = NP.asarray(zero_uind).ravel()
            zero_pind = NP.asarray(zero_pind).ravel()            
            syn_beam_in_uv[(zero_vind, zero_uind, zero_pind)] = 0.0
            syn_beam = NP.fft.fft2(syn_beam_in_uv, axes=(0,1))  # FT
            if NP.abs(syn_beam.imag).max() > 1e-10:
                raise ValueError('Synthesized beam after zero spacing aperture removal has significant imaginary component')
            else:
                syn_beam = syn_beam.real
                norm_factor = 1.0 / syn_beam.max()
                syn_beam *= norm_factor  # Normalize to get unit peak for PSF
                syn_beam_in_uv *= norm_factor  # Normalize to get unit peak for PSF
        
        # shift the array to be centered
        syn_beam_in_uv = NP.fft.ifftshift(syn_beam_in_uv, axes=(0,1)) # Shift array to be centered

        # Discard pads at either end and select only the central values of twice the original size
        syn_beam_in_uv = syn_beam_in_uv[grid_field_illumination.shape[0]:3*grid_field_illumination.shape[0],grid_field_illumination.shape[1]:3*grid_field_illumination.shape[1],:]
        syn_beam = NP.fft.fftshift(syn_beam[::2,::2,:], axes=(0,1))  # Downsample by factor 2 to get native resolution and shift to be centered
        
        l = DSP.spectax(2*self.gridu.shape[1], resolution=du, shift=True)
        m = DSP.spectax(2*self.gridv.shape[0], resolution=dv, shift=True)

        return {'syn_beam': syn_beam, 'grid_power_illumination': syn_beam_in_uv, 'l': l, 'm': m}

    ############################################################################ 

    def update_flags(self, dictflags=None, stack=True, verify=False):

        """
        ------------------------------------------------------------------------
        Updates all flags in the antenna array followed by any flags that
        need overriding through inputs of specific flag information

        Inputs:

        dictflags  [dictionary] contains flag information overriding after 
                   default flag updates are determined. Antenna based flags are 
                   given as further dictionaries with each under under a key 
                   which is the same as the antenna label. Flags for each 
                   antenna are specified as a dictionary holding boolean flags 
                   for each of the two polarizations which are stored under 
                   keys 'P1' and 'P2'. An absent key just means it is not a 
                   part of the update. Flag information under each antenna must 
                   be of same type as input parameter flags in member function 
                   update_flags() of class PolInfo

        stack      [boolean] If True (default), appends the updated flag to the
                   end of the stack of flags as a function of timestamp. If 
                   False, updates the last flag in the stack with the updated 
                   flag and does not append

        verify     [boolean] If True, verify and update the flags, if necessary.
                   Electric fields are checked for NaN values and if found, the
                   flag in the corresponding polarization is set to True. 
                   Default=False. 
        ------------------------------------------------------------------------
        """

        for label in self.antennas:
            self.antennas[label].update_flags(stack=stack, verify=verify)

        if dictflags is not None:  # Performs flag overriding. Use stack=False
            if not isinstance(dictflags, dict):
                raise TypeError('Input parameter dictflags must be a dictionary')
            
            for label in dictflags:
                if label in self.antennas:
                    self.antennas[label].update_flags(flags=dictflags[label], stack=False, verify=True)

    ############################################################################

    def update(self, updates=None, parallel=False, nproc=None, verbose=False):

        """
        ------------------------------------------------------------------------
        Updates the antenna array instance with newer attribute values. Can also 
        be used to add and/or remove antennas with/without affecting the 
        existing grid.

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
                                'do_grid'     [boolean] If set to True, create 
                                              or recreate a grid. To be 
                                              specified when the antenna 
                                              locations are updated.
                    'antennas': Holds a list of dictionaries consisting of 
                                updates for individual antennas. Each element 
                                in the list contains update for one antenna. 
                                For each of these dictionaries, one of the keys 
                                is 'label' which indicates an antenna label. If 
                                absent, the code execution stops by throwing an 
                                exception. The other optional keys and the 
                                information they hold are listed below:
                                'action'      [String scalar] Indicates the type 
                                              of update operation. 'add' adds 
                                              the Antenna instance to the 
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
                                              False, gridding effects will 
                                              remain unchanged. Default=None
                                              (=False).
                                'antenna'     [instance of class Antenna] 
                                              Updated Antenna class instance. 
                                              Can work for action key 'remove' 
                                              even if not set (=None) or set to 
                                              an empty string '' as long as 
                                              'label' key is specified. 
                                'gridpol'     [Optional. String scalar] 
                                              Initiates the specified action on 
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
                                'stack'       [boolean] If True (default), 
                                              appends the updated flag and data 
                                              to the end of the stack as a 
                                              function of timestamp. If False, 
                                              updates the last flag and data in 
                                              the stack and does not append
                                't'           [Optional. Numpy array] Time axis 
                                              of the time series. Is used only 
                                              if set and if 'action' key value 
                                              is set to 'modify'. Default=None.
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
                                'aperture'    [instance of class 
                                              APR.Aperture] aperture 
                                              information for the antenna. Read 
                                              docstring of class 
                                              Aperture for details
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
                                              on delay compensation to be 
                                              applied to the fourier transformed 
                                              electric fields under each 
                                              polarization which are stored 
                                              under keys 'P1' and 'P2'. 
                                              Default=None (no delay 
                                              compensation to be applied). Refer 
                                              to the docstring of member 
                                              function delay_compensation() of 
                                              class PolInfo for more details.
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
                                              If set to True, the gridded 
                                              weights are divided by the sum of 
                                              weights so that the gridded 
                                              weights add up to unity. This is 
                                              used only when grid_action keyword 
                                              is set when action keyword is set 
                                              to 'add' or 'modify'
                                'gridmethod'  [Optional. String] Indicates 
                                              gridding method. It accepts the 
                                              following values 'NN' (nearest 
                                              neighbour), 'BL' (Bi-linear
                                              interpolation), and'CS' (Cubic
                                              Spline interpolation). 
                                              Default='NN'
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

        ------------------------------------------------------------------------
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
                            if 'stack' not in dictitem: dictitem['stack']=True
                            if 'gridfunc_freq' not in dictitem: dictitem['gridfunc_freq']=None
                            if 'ref_freq' not in dictitem: dictitem['ref_freq']=None
                            if 'pol_type' not in dictitem: dictitem['pol_type']=None
                            if 'norm_wts' not in dictitem: dictitem['norm_wts']=False
                            if 'gridmethod' not in dictitem: dictitem['gridmethod']='NN'
                            if 'distNN' not in dictitem: dictitem['distNN']=NP.inf
                            if 'maxmatch' not in dictitem: dictitem['maxmatch']=None
                            if 'tol' not in dictitem: dictitem['tol']=None
                            if 'delaydict' not in dictitem: dictitem['delaydict']=None
                            if 'aperture' not in dictitem: dictitem['aperture']=None
                            
                            if not parallel:
                                self.antennas[dictitem['label']].update(dictitem, verbose)
                            else:
                                list_of_antennas += [self.antennas[dictitem['label']]]
                                list_of_antenna_updates += [dictitem]

                            if 'grid_action' in dictitem:
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
                    self.timestamps += [copy.deepcopy(self.timestamp)] # Stacks new timestamp

                if 'do_grid' in updates['antenna_array']:
                    if isinstance(updates['antenna_array']['do_grid'], boolean):
                        self.grid()
                    else:
                        raise TypeError('Value in key "do_grid" inside key "antenna_array" of input dictionary updates must be boolean.')

        self.t = self.antennas.itervalues().next().t # Update time axis
        self.f = self.antennas.itervalues().next().f # Update frequency axis
        self.update_flags(stack=False, verify=True)  # Refreshes current flags, no stacking

################################################################################
