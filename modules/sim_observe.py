import copy
import numpy as NP
import scipy.sparse as SpM
import scipy.constants as FCNST
import ephem as EP
import multiprocessing as MP
import itertools as IT
from astropy.io import fits, ascii
import h5py
import progressbar as PGB
from astroutils import writer_module as WM
from astroutils import DSP_modules as DSP
from astroutils import mathops as OPS
from astroutils import geometry as GEOM
from astroutils import constants as CNST
from astroutils import catalog as SM
from astroutils import gridding_modules as GRD
from astroutils import lookup_operations as LKP
import antenna_array as AA
import antenna_beams as AB

sday = CNST.sday
sday_correction = 1 / sday

################### Routines essential for parallel processing ################

def interp_beam_arg_splitter(args, **kwargs):
    return interp_beam(*args, **kwargs)

def stochastic_E_timeseries_arg_splitter(args, **kwargs):
    return stochastic_E_timeseries(*args, **kwargs)

def generate_E_spectrum_arg_splitter(args, **kwargs):
    return generate_E_spectrum(*args, **kwargs)

###############################################################################

def interp_beam(beamfile, theta_phi, freqs):

    """
    ---------------------------------------------------------------------------
    Read and interpolate antenna pattern to the specified frequencies and 
    angular locations.

    Inputs:

    beamfile    [string] Full path to file containing antenna pattern. Must be
                specified, no default.

    theta_phi   [numpy array] Zenith angle and Azimuth as a nsrc x 2 numpy 
                array. It must be specified in radians. If not specified, no 
                interpolation is performed spatially.

    freqs       [numpy array] Frequencies (in Hz) at which the antenna pattern 
                is to be interpolated. If not specified, no spectral 
                interpolation is performed

    Outputs:

    Antenna pattern interpolated at locations and frequencies specified. It will
    be a numpy array of size nsrc x nchan
    ----------------------------------------------------------------------------
    """

    try:
        beamfile
    except NameError:
        raise NameError('Input beamfile must be specified')

    try:
        theta_phi
    except NameError:
        theta_phi = None

    if theta_phi is not None:
        if not isinstance(theta_phi, NP.ndarray):
            raise TypeError('Input theta_phi must be a numpy array')

        if theta_phi.ndim != 2:
            raise ValueError('Input theta_phi must be a nsrc x 2 numpy array')

    try:
        freqs
    except NameError:
        freqs = None

    try:
        hdulist = fits.open(beamfile)
    except IOError:
        raise IOError('Error opening file containing antenna voltage pattern')

    extnames = [hdu.header['EXTNAME'] for hdu in hdulist]
    if 'BEAM' not in extnames:
        raise KeyError('Key "BEAM" not found in file containing antenna voltage pattern')

    if 'FREQS' not in extanmes:
        if freqs is not None:
            vbfreqs = freqs
        else:
            raise ValueError('Frequencies not specified in file containing antenna voltage pattern')
    else:
        vbfreqs = hdulist['FREQS']
        if not isinstance(vbfreqs, NP.ndarray):
            raise TypeError('Frequencies in antenna voltage pattern must be a numpy array')

    vbeam = hdulist['BEAM']
    if not isinstance(vbeam, NP.ndarray):
        raise TypeError('Reference antenna voltage pattern must be a numpy array')

    if vbeam.ndim == 1:
        vbeam = vbeam[:,NP.newaxis]
    elif vbeam.ndim == 2:
        if vbeam.shape[1] != 1:
            if vbeam.shape[1] != vbfreqs.size:
                raise ValueError('Shape of antenna voltage pattern not compatible with number of frequency channels')
    else:
        raise ValueError('Antenna voltage pattern must be of size nsrc x nchan')

    if vbeam.shape[1] == 1:
        vbeam = vbeam + NP.zeros(vbfreqs.size).reshape(1,-1)

    return OPS.healpix_interp_along_axis(vbeam, theta_phi=theta_phi, inloc_axis=vbfreqs, outloc_axis=freqs, axis=1, kind='cubic', assume_sorted=True)

###############################################################################

def generate_E_spectrum(freqs, skypos=[0.0,0.0,1.0], flux_ref=1.0,
                        freq_ref=None, spectral_index=0.0, spectrum=None,
                        antpos=[0.0,0.0,0.0], voltage_pattern=None,
                        ref_point=None, randomseed=None, randvals=None,
                        verbose=True):

    """
    ----------------------------------------------------------------------------
    Compute a stochastic electric field spectrum obtained from sources with 
    given spectral information of sources, their positions at specified 
    antenna locations with respective voltage patterns

    Inputs:

    freqs            [numpy array] Frequencies (in Hz) of the frequency channels

    Keyword Inputs:

    skypos           [list, tuple, list of lists, list of tuples, numpy array]
                     Sky positions of sources provided in direction cosine
                     coordinates aligned with local ENU axes. It should be a
                     3-element list, a 3-element tuple, a list of 3-element
                     lists, list of 3-element tuples, or a 3-column numpy array.
                     Each 3-element entity corresponds to a source position. 
                     Number of 3-element entities should equal the number of 
                     sources as specified by the size of flux_ref. Rules of 
                     direction cosine quantities should be followed. If only 
                     one  source is specified by flux_ref and skypos is not
                     specified, skypos defaults to the zenith (0.0, 0.0, 1.0)

    flux_ref         [list or numpy array of float] Flux densities of sources
                     at the respective reference frequencies. Units are 
                     arbitrary. Values have to be positive. Default = 1.0. 

    freq_ref         [list or numpy array of float] Reference frequency (Hz). 
                     If not provided, default is set to center frequency given
                     in freq_center for each of the sources. If a single value 
                     is provided, it will be applicable to all the sources. If a 
                     list or numpy array is provided, it should be of size equal
                     to that of flux_ref. 

    spectral_index   [list or numpy array of float] Spectral Index 
                     (flux ~ freq ** alpha). If not provided, default is set to 
                     zero, a flat spectrum, for each of the sources. If a single 
                     value is provided, it will be applicable to all the sources. 
                     If a list or numpy array is provided, it should be of size 
                     equal to that of flux_ref. 

    spectrum         [numpy array] Spectrum of catalog objects whose locations 
                     are specified in skypos and frequencies in freqs. It is of
                     size nsrc x nchan. Default=None means determine spectral 
                     information from the spectral index. If not set to None, 
                     spectral information from this input will be used and info
                     in spectral index will be ignored. 

    ref_point        [3-element list, tuple, or numpy vector] Point on sky used
                     as a phase reference. Same units as skypos (which is
                     direction cosines and must satisfy rules of direction
                     cosines). If None provided, it defaults to zenith
                     (0.0, 0.0, 1.0)

    antpos           [list, tuple, list of lists, list of tuples, numpy array]
                     Antenna positions provided along local ENU axes. 
                     It should be a 3-element list, a 3-element tuple, a list of 
                     3-element lists, list of 3-element tuples, or a 3-column 
                     numpy array. Each 3-element entity corresponds to an
                     antenna position. If not specified, antpos by default is 
                     assigned the origin (0.0, 0.0, 0.0).

    voltage_pattern  [numpy array] Voltage pattern for each frequency channel
                     at each source location for each antenna. It must be of
                     shape nsrc x nchan x nant. If any of these dimensions are
                     1, it is assumed to be identical along that direction. 
                     If specified as None (default), it is assumed to be unity
                     and identical across antennas, sky locations and frequency
                     channels. 

    randomseed       [integer] Seed to initialize the randon generator. If set
                     to None (default), the random sequences generated are not
                     reproducible. Set to an integer to generate reproducible
                     random sequences. Will be used only if the other input 
                     randvals is set to None

    randvals         [numpy array] Externally generated complex random numbers.
                     Both real and imaginary parts must be drawn from a normal
                     distribution (mean=0, var=1). Always must have size equal 
                     to nsrc x nchan. If specified as a vector, it must be of 
                     size nsrc x nchan. If specified as a 2D or higher 
                     dimensional array its first two dimensions must be of 
                     shape nsrc x nchan and total size equal to nsrc x nchan. 
                     If randvals is specified, no fresh random numbers will be
                     generated and the input randomseed will be ignored.

    verbose:         [boolean] If set to True, prints progress and diagnostic
                     messages. Default = True.
    
    Output:

    dictout          [dictionary] Consists of the following tags and info:
                     'f'        [numpy array] frequencies of the channels in the 
                                spectrum of size nchan
                     'Ef'       [complex numpy array] nchan x nant numpy array 
                                consisting of complex stochastic electric field
                                spectra. nchan is the number of channels in the 
                                spectrum and nant is the number of antennas.

    ----------------------------------------------------------------------------
    """

    try:
        freqs
    except NameError:
        raise NameError('Input freqs must be provided')

    if isinstance(freqs, (int, float)):
        freqs = NP.asarray(freqs).reshape(-1)
    elif isinstance(freqs, list):
        freqs = NP.asarray(freqs)
    elif not isinstance(freqs, NP.ndarray):
        raise TypeError('Input freqs must be a scalar, list or numpy array')

    freqs = freqs.ravel()
    if NP.any(freqs <= 0.0):
        raise ValueError('Frequencies must be positive')

    if isinstance(antpos, (list, tuple)):
        antpos = NP.asarray(antpos)
        if antpos.ndim == 1:
            if antpos.size != 3:
                raise IndexError('Antenna position must be a three-element vector and aligned with the local ENU coordinate system.')
            else:
                antpos = antpos.reshape(1,-1)
        elif antpos.shape[1] != 3:
            raise IndexError('Antenna position must be a three-element vector aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    elif isinstance(antpos, NP.ndarray):
        if antpos.ndim == 1:
            if antpos.size != 3:
                raise IndexError('Antenna position must be a three-element vector aligned with the local ENU coordinate system.')
            else:
                antpos = antpos.reshape(1,-1)
        elif antpos.shape[1] != 3:
            raise IndexError('Antenna position must be a three-element vector aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    else:
        raise TypeError('Antenna position (antpos) must be a three-element list or tuple, list of lists or list of tuples with each of the inner lists or tuples holding three elements, or a three-column numpy array.')
     
    if skypos is None:
        if nsrc > 1:
            raise ValueError('Sky positions (skypos) must be specified for each of the multiple flux densities.')
        skypos = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    elif isinstance(skypos, (list, tuple)):
        skypos = NP.asarray(skypos)
        if len(skypos.shape) == 1:
            if skypos.size != 3:
                raise IndexError('Sky position must be a three-element vector of direction cosines for each source, and aligned with the local ENU coordinate system.')
            else:
                skypos = skypos.reshape(1,-1)
        elif skypos.shape[1] != 3:
            raise IndexError('Sky position must be a three-element vector for each source given as direction cosines aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    elif isinstance(skypos, NP.ndarray):
        if len(skypos.shape) == 1:
            if skypos.size != 3:
                raise IndexError('Sky position must be a three-element vector for each source given as direction cosines aligned with the local ENU coordinate system.')
            else:
                skypos = skypos.reshape(1,-1)
        elif skypos.shape[1] != 3:
            raise IndexError('Sky position must be a three-element vector for each source given as direction cosines aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    else:
        raise TypeError('Sky position (skypos) must be a three-element list or tuple, list of lists or list of tuples with each of the inner lists or tuples holding three elements, or a three-column numpy array.')
            
    eps = 1e-10
    if NP.any(NP.abs(skypos) >= 1.0+eps):
        raise ValueError('Components of direction cosines must not exceed unity')
    if NP.any(NP.abs(NP.sum(skypos**2,axis=1)-1.0) >= eps):
        raise ValueError('Magnitudes of direction cosines must not exceed unity')
    
    if ref_point is None:
        ref_point = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    elif isinstance(ref_point, (list, tuple, NP.ndarray)):
        ref_point = NP.asarray(ref_point).reshape(1,-1)
    else:
        raise TypeError('Reference position must be a list, tuple or numpy array.')

    if ref_point.size != 3:
        raise ValueError('Reference position must be a 3-element list, tuple or numpy array of direction cosines.')

    eps = 1.0e-10
    if NP.any(NP.abs(skypos) > 1.0):
        raise ValueError('Some direction cosine values have absolute values greater than unity.')
    elif NP.any(NP.abs(1.0-NP.sqrt(NP.sum(skypos**2,axis=1))) > eps):
        raise ValueError('Some sky positions specified in direction cosines do not have unit magnitude by at least {0:.1e}.'.format(eps))

    if NP.any(NP.abs(ref_point) > 1.0):
        raise ValueError('Direction cosines in reference position cannot exceed unit magnitude.')
    elif NP.abs(1.0-NP.sqrt(NP.sum(ref_point**2))) > eps:
        raise ValueError('Unit vector denoting reference position in direction cosine units must have unit magnitude.')

    freqs = freqs.reshape(1,-1,1) # 1 x nchan x 1
    nchan = freqs.size
    nant = antpos.shape[0]
    nsrc = skypos.shape[0]

    if spectrum is None:
        if freq_ref is None:
            if verbose:
                print '\tNo reference frequency (freq_ref) provided. Setting it equal to center \n\t\tfrequency.'
            freq_ref = NP.mean(freqs).reshape(-1)
    
        if isinstance(freq_ref, (int,float)):
            freq_ref = NP.asarray(freq_ref).reshape(-1)
        elif isinstance(freq_ref, (list, tuple)):
            freq_ref = NP.asarray(freq_ref)
        elif isinstance(freq_ref, NP.ndarray):
            freq_ref = freq_ref.ravel()
        else:
            raise TypeError('Reference frequency (freq_ref) must be a scalar, list, tuple or numpy array. Aborting stochastic_E_spectrum().')
    
        if NP.any(freq_ref <= 0.0):
            raise ValueError('freq_ref must be a positive value. Aborting stochastic_E_spectrum().')

        if freq_ref.size > 1:
            if freq_ref.size != nsrc:
                raise ValueError('Size of freq_ref does not match number of sky positions')

        if isinstance(flux_ref, (int,float)):
            flux_ref = NP.asarray(flux_ref).reshape(-1)
        elif isinstance(flux_ref, (list, tuple)):
            flux_ref = NP.asarray(flux_ref)
        elif isinstance(flux_ref, NP.ndarray):
            flux_ref = flux_ref.ravel()
        else:
            raise TypeError('Flux density at reference frequency (flux_ref) must be a scalar, list, tuple or numpy array. Aborting stochastic_E_spectrum().')
    
        if NP.any(flux_ref <= 0.0):
            raise ValueError('flux_ref must be a positive value. Aborting stochastic_E_spectrum().')
    
        if flux_ref.size > 1:
            if flux_ref.size != nsrc:
                raise ValueError('Size of flux_ref does not match number of sky positions')

        if isinstance(spectral_index, (int,float)):
            spectral_index = NP.asarray(spectral_index).reshape(-1)
        elif isinstance(spectral_index, (list, tuple)):
            spectral_index = NP.asarray(spectral_index)
        elif isinstance(spectral_index, NP.ndarray):
            spectral_index = spectral_index.ravel()
        else:
            raise TypeError('Spectral index (spectral_index) must be a scalar, list, tuple or numpy array. Aborting stochastic_E_spectrum().')

        if spectral_index.size > 1:
            if spectral_index.size != nsrc:
                raise ValueError('Size of spectral_index does not match number of sky positions')

        nsi = spectral_index.size 

        alpha = spectral_index.reshape(-1,1,1) # nsrc x 1 x 1
        freq_ratio = freqs / freq_ref.reshape(-1,1,1) # nsrc x nchan x 1
        spectrum = flux_ref.reshape(-1,1,1) * (freq_ratio ** alpha) # nsrc x nchan x 1
    else:
        if not isinstance(spectrum, NP.ndarray):
            raise TypeError('Input spectrum must be a numpy array')
        if spectrum.ndim == 1:
            spectrum = spectrum.reshape(-1,1) # nsrc x 1
        elif spectrum.ndim != 2:
            raise ValueError('Input spectrum has too many dimensions')

        if spectrum.shape[1] > 1:
            if spectrum.shape[1] != nchan:
                raise ValueError('Number of frequency channels in spectrum does not match number of frequency channels specified')
        else:
            spectrum = spectrum + NP.zeros(nchan).reshape(1,-1) # nsrc x nchan or 1 x nchan

        if spectrum.shape[0] > 1:
            if spectrum.shape[0] != nsrc:
                raise ValueError('Number of locations in spectrum does not match number of sources')
        else:
            spectrum = spectrum + NP.zeros(nsrc).reshape(-1,1) # nsrc x nchan

        spectrum = spectrum[:,:,NP.newaxis] # nsrc x nchan x 1

    if voltage_pattern is None:
        voltage_pattern = NP.ones(1).reshape(1,1,1)
    elif not isinstance(voltage_pattern, NP.ndarray):
        raise TypeError('Input antenna voltage pattern must be an array')

    if voltage_pattern.ndim == 2:
        voltage_pattern = voltage_pattern[:,:,NP.newaxis] # nsrc x nchan x 1
    elif voltage_pattern.ndim != 3:
        raise ValueError('Dimensions of voltage pattern incompatible')

    vb_shape = voltage_pattern.shape
    if (vb_shape[2] != 1) and (vb_shape[2] != nant):
        raise ValueError('Input voltage pattern must be specified for each antenna or assumed to be identical to all antennas')
    if (vb_shape[0] != 1) and (vb_shape[0] != nsrc):
        raise ValueError('Input voltage pattern must be specified at each sky location or assumed to be identical at all locations')
    if (vb_shape[1] != 1) and (vb_shape[1] != nchan):
        raise ValueError('Input voltage pattern must be specified at each frequency channel or assumed to be identical for all')

    if randvals is not None:
        if not isinstance(randvals, NP.ndarray):
            raise TypeError('Input randvals must be a numpy array')
        if randvals.size != nsrc * nchan:
            raise ValueError('Input randvals found to be of invalid size')
        if randvals.ndim >= 2:
            if (randvals.shape[0] != nsrc) or (randvals.shape[1] != nchan):
                raise ValueError('Input randvals found to be invalid dimensions')
        randvals = randvals.reshape(nsrc,nchan,1)
    else:
        randstate = NP.random.RandomState(randomseed)
        randvals = randstate.normal(loc=0.0, scale=1.0, size=(nsrc,nchan,1)) + 1j * randstate.normal(loc=0.0, scale=1.0, size=(nsrc,nchan,1)) # nsrc x nchan x 1

    if verbose:
        print '\tArguments verified for compatibility.'
        print '\tSetting up the recipe for producing stochastic Electric field spectra...'

    sigmas = NP.sqrt(spectrum) # nsrc x nchan x 1
    Ef_amp = sigmas/NP.sqrt(2) * randvals
    # Ef_amp = sigmas/NP.sqrt(2) * (NP.random.normal(loc=0.0, scale=1.0, size=(nsrc,nchan,1)) + 1j * NP.random.normal(loc=0.0, scale=1.0, size=(nsrc,nchan,1))) # nsrc x nchan x 1
    Ef_phase = 1.0
    Ef = Ef_amp * Ef_phase # nsrc x nchan x 1
    skypos_dot_antpos = NP.dot(skypos-ref_point, antpos.T) # nsrc x nant
    k_dot_r_phase = 2.0 * NP.pi * freqs / FCNST.c * skypos_dot_antpos[:,NP.newaxis,:] # nsrc x nchan x nant
    Ef = Ef * voltage_pattern * NP.exp(1j * k_dot_r_phase) # nsrc x nchan x nant
    Ef = NP.sum(Ef, axis=0) # nchan x nant
    if verbose:
        print '\tPerformed linear superposition of electric fields from source(s).'
    
    dictout = {}
    dictout['f'] = freqs.ravel()
    dictout['Ef'] = Ef
    if verbose:
        print 'stochastic_E_spectrum() executed successfully.\n'

    return dictout

###############################################################################

def stochastic_E_spectrum(freq_center, nchan, channel_width, flux_ref=1.0,
                          freq_ref=None, spectral_index=0.0, skypos=None, 
                          ref_point=None, antpos=[0.0,0.0,0.0],
                          voltage_pattern=None, verbose=True):

    """
    ----------------------------------------------------------------------------
    Compute a stochastic electric field spectrum obtained from sources with 
    given flux densities and spectral indices at given positions at specified 
    antenna locations. 

    Inputs:

    freq_center      [float] Center frequency in Hz. Center frequency must be
                     greater than half the bandwidth.

    nchan            [integer] Number of frequency channels in spectrum

    channel_width    [float] Channel width in Hz

    Keyword Inputs:

    flux_ref         [list or numpy array of float] Flux densities of sources
                     at the respective reference frequencies. Units are 
                     arbitrary. Values have to be positive. Default = 1.0. 

    freq_ref         [list or numpy array of float] Reference frequency (Hz). 
                     If not provided, default is set to center frequency given
                     in freq_center for each of the sources. If a single value 
                     is provided, it will be applicable to all the sources. If a 
                     list or numpy array is provided, it should be of size equal
                     to that of flux_ref. 

    spectral_index   [list or numpy array of float] Spectral Index 
                     (flux ~ freq ** alpha). If not provided, default is set to 
                     zero, a flat spectrum, for each of the sources. If a single 
                     value is provided, it will be applicable to all the sources. 
                     If a list or numpy array is provided, it should be of size 
                     equal to that of flux_ref. 

    skypos           [list, tuple, list of lists, list of tuples, numpy array]
                     Sky positions of sources provided in direction cosine
                     coordinates aligned with local ENU axes. It should be a
                     3-element list, a 3-element tuple, a list of 3-element
                     lists, list of 3-element tuples, or a 3-column numpy array.
                     Each 3-element entity corresponds to a source position. 
                     Number of 3-element entities should equal the number of 
                     sources as specified by the size of flux_ref. Rules of 
                     direction cosine quantities should be followed. If only 
                     one  source is specified by flux_ref and skypos is not
                     specified, skypos defaults to the zenith (0.0, 0.0, 1.0)

    ref_point        [3-element list, tuple, or numpy vector] Point on sky used
                     as a phase reference. Same units as skypos (which is
                     direction cosines and must satisfy rules of direction
                     cosines). If None provided, it defaults to zenith
                     (0.0, 0.0, 1.0)

    antpos           [list, tuple, list of lists, list of tuples, numpy array]
                     Antenna positions provided along local ENU axes. 
                     It should be a 3-element list, a 3-element tuple, a list of 
                     3-element lists, list of 3-element tuples, or a 3-column 
                     numpy array. Each 3-element entity corresponds to an
                     antenna position. If not specified, antpos by default is 
                     assigned the origin (0.0, 0.0, 0.0).

    voltage_pattern  [numpy array] Voltage pattern for each frequency channel
                     at each source location for each antenna. It must be of
                     shape nsrc x nchan x nant. If any of these dimensions are
                     1, it is assumed to be identical along that direction. 
                     If specified as None (default), it is assumed to be unity
                     and identical across antennas, sky locations and frequency
                     channels. 

    verbose:         [boolean] If set to True, prints progress and diagnostic
                     messages. Default = True.
    
    Output:

    dictout          [dictionary] Consists of the following tags and info:
                     'f'        [numpy array] frequencies of the channels in the 
                                spectrum of size nchan
                     'Ef'       [complex numpy array] nchan x nant numpy array 
                                consisting of complex stochastic electric field
                                spectra. nchan is the number of channels in the 
                                spectrum and nant is the number of antennas.
                     'antpos'   [numpy array] 3-column array of antenna
                                positions (same as the input argument antpos)

    ----------------------------------------------------------------------------
    """

    if verbose:
        print '\nExecuting stochastic_E_spectrum()...'
        print '\tChecking data compatibility...'

    try:
        freq_center, nchan, channel_width
    except NameError:
        raise NameError('    Center frequency (freq_center), number of channels (nchan) and frequency resolution (channel_width) must be provided. Aborting stochastic_E_spectrum().')

    if not isinstance(nchan, (int,float)):
        raise TypeError('    nchan must be a scalar value. Aborting stochastic_E_spectrum().')
    nchan = int(nchan)
    if nchan <= 1:
        raise ValueError('    nchan must be an integer greater than unity. Aborting stochastic_E_spectrum().')
        
    if not isinstance(channel_width, (int,float)):
        raise TypeError('    channel_width must be a scalar value. Aborting stochastic_E_spectrum().')
    if channel_width <= 0.0:
        raise ValueError('    channel_width must be a positive value. Aborting stochastic_E_spectrum().')

    if not isinstance(freq_center, (int,float)):
        raise TypeError('    freq_center must be a scalar value. Aborting stochastic_E_spectrum().')

    freq_center = float(freq_center)
    if freq_center <= 0.0:
        raise ValueError('    freq_center must be a positive value. Aborting stochastic_E_spectrum().')

    if (freq_center - 0.5*nchan*channel_width) <= 0.0:
        raise ValueError('    Center frequency must be greater than half the bandwidth. Aborting stochastic_E_spectrum().')

    if freq_ref is None:
        if verbose:
            print '\tNo reference frequency (freq_ref) provided. Setting it equal to center \n\t\tfrequency.'
        freq_ref = freq_center * NP.ones(1)

    if isinstance(freq_ref, (int,float)):
        freq_ref = NP.asarray(freq_ref).reshape(-1)
    elif isinstance(freq_ref, (list, tuple)):
        freq_ref = NP.asarray(freq_ref)
    elif isinstance(freq_ref, NP.ndarray):
        freq_ref = freq_ref.ravel()
    else:
        raise TypeError('Reference frequency (freq_ref) must be a scalar, list, tuple or numpy array. Aborting stochastic_E_spectrum().')

    if NP.any(freq_ref <= 0.0):
        raise ValueError('freq_ref must be a positive value. Aborting stochastic_E_spectrum().')

    if isinstance(flux_ref, (int,float)):
        flux_ref = NP.asarray(flux_ref).reshape(-1)
    elif isinstance(flux_ref, (list, tuple)):
        flux_ref = NP.asarray(flux_ref)
    elif isinstance(flux_ref, NP.ndarray):
        flux_ref = flux_ref.ravel()
    else:
        raise TypeError('Flux density at reference frequency (flux_ref) must be a scalar, list, tuple or numpy array. Aborting stochastic_E_spectrum().')

    if NP.any(flux_ref <= 0.0):
        raise ValueError('flux_ref must be a positive value. Aborting stochastic_E_spectrum().')

    if isinstance(spectral_index, (int,float)):
        spectral_index = NP.asarray(spectral_index).reshape(-1)
    elif isinstance(spectral_index, (list, tuple)):
        spectral_index = NP.asarray(spectral_index)
    elif isinstance(spectral_index, NP.ndarray):
        spectral_index = spectral_index.ravel()
    else:
        raise TypeError('Spectral index (spectral_index) must be a scalar, list, tuple or numpy array. Aborting stochastic_E_spectrum().')

    nsrc = flux_ref.size
    nref = freq_ref.size
    nsi = spectral_index.size 

    if skypos is None:
        if nsrc > 1:
            raise ValueError('Sky positions (skypos) must be specified for each of the multiple flux densities.')
        skypos = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    elif isinstance(skypos, (list, tuple)):
        skypos = NP.asarray(skypos)
        if len(skypos.shape) == 1:
            if skypos.size != 3:
                raise IndexError('Sky position must be a three-element vector of direction cosines for each source, and aligned with the local ENU coordinate system.')
            else:
                skypos = skypos.reshape(1,-1)
        elif skypos.shape[1] != 3:
            raise IndexError('Sky position must be a three-element vector for each source given as direction cosines aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    elif isinstance(skypos, NP.ndarray):
        if len(skypos.shape) == 1:
            if skypos.size != 3:
                raise IndexError('Sky position must be a three-element vector for each source given as direction cosines aligned with the local ENU coordinate system.')
            else:
                skypos = skypos.reshape(1,-1)
        elif skypos.shape[1] != 3:
            raise IndexError('Sky position must be a three-element vector for each source given as direction cosines aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    else:
        raise TypeError('Sky position (skypos) must be a three-element list or tuple, list of lists or list of tuples with each of the inner lists or tuples holding three elements, or a three-column numpy array.')
            
    if nsrc > skypos.shape[0]:
        raise ValueError('Sky positions must be provided for each of source flux densities.')
    elif nsrc < skypos.shape[0]:
        skypos = skypos[:nsrc,:]

    if ref_point is None:
        ref_point = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    elif isinstance(ref_point, (list, tuple, NP.ndarray)):
        ref_point = NP.asarray(ref_point).reshape(1,-1)
    else:
        raise TypeError('Reference position must be a list, tuple or numpy array.')

    if ref_point.size != 3:
        raise ValueError('Reference position must be a 3-element list, tuple or numpy array of direction cosines.')

    eps = 1.0e-10
    if NP.any(NP.abs(skypos) > 1.0):
        raise ValueError('Some direction cosine values have absolute values greater than unity.')
    elif NP.any(NP.abs(1.0-NP.sqrt(NP.sum(skypos**2,axis=1))) > eps):
        raise ValueError('Some sky positions specified in direction cosines do not have unit magnitude by at least {0:.1e}.'.format(eps))

    if NP.any(NP.abs(ref_point) > 1.0):
        raise ValueError('Direction cosines in reference position cannot exceed unit magnitude.')
    elif NP.abs(1.0-NP.sqrt(NP.sum(ref_point**2))) > eps:
        raise ValueError('Unit vector denoting reference position in direction cosine units must have unit magnitude.')

    if nsrc == 1:
        freq_ref = NP.asarray(freq_ref[0]).reshape(-1)
        spectral_index = NP.asarray(spectral_index[0]).reshape(-1)
        nref = 1
        nsi = 1
    else: 
        if nref == 1:
            freq_ref = NP.repeat(freq_ref, nsrc)
            nref = nsrc
        elif nref != nsrc:
            raise ValueError('Number of reference frequencies should be either 1 or match the number of flux densities of sources.')

        if nsi == 1:
            spectral_index = NP.repeat(spectral_index, nsrc)
            nsi = nsrc
        elif nsi != nsrc:
            raise ValueError('Number of spectral indices should be either 1 or match the number of flux densities of sources.')

    if antpos is None:
        antpos = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    elif isinstance(antpos, (list, tuple)):
        antpos = NP.asarray(antpos)
        if len(antpos.shape) == 1:
            if antpos.size != 3:
                raise IndexError('Antenna position must be a three-element vector and aligned with the local ENU coordinate system.')
            else:
                antpos = antpos.reshape(1,-1)
        elif antpos.shape[1] != 3:
            raise IndexError('Antenna position must be a three-element vector aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    elif isinstance(antpos, NP.ndarray):
        if len(antpos.shape) == 1:
            if antpos.size != 3:
                raise IndexError('Antenna position must be a three-element vector aligned with the local ENU coordinate system.')
            else:
                antpos = antpos.reshape(1,-1)
        elif antpos.shape[1] != 3:
            raise IndexError('Antenna position must be a three-element vector aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    else:
        raise TypeError('Antenna position (antpos) must be a three-element list or tuple, list of lists or list of tuples with each of the inner lists or tuples holding three elements, or a three-column numpy array.')
     
    nant = antpos.shape[0]

    if voltage_pattern is None:
        voltage_pattern = NP.ones(1).reshape(1,1,1)
    elif not isinstance(voltage_pattern, NP.ndarray):
        raise TypeError('Input antenna voltage pattern must be an array')

    if voltage_pattern.ndim == 2:
        voltage_pattern = voltage_pattern[:,:,NP.newaxis]
    elif voltage_pattern.ndim != 3:
        raise ValueError('Dimensions of voltage pattern incompatible')

    vb_shape = voltage_pattern.shape
    if (vb_shape[2] != 1) and (vb_shape[2] != nant):
        raise ValueError('Input voltage pattern must be specified for each antenna or assumed to be identical to all antennas')
    if (vb_shape[0] != 1) and (vb_shape[0] != nsrc):
        raise ValueError('Input voltage pattern must be specified at each sky location or assumed to be identical at all locations')
    if (vb_shape[1] != 1) and (vb_shape[1] != nchan):
        raise ValueError('Input voltage pattern must be specified at each frequency channel or assumed to be identical for all')

    if verbose:
        print '\tArguments verified for compatibility.'
        print '\tSetting up the recipe for producing stochastic Electric field spectra...'

    center_channel = int(NP.floor(0.5*nchan))
    freqs = freq_center + channel_width * (NP.arange(nchan) - center_channel)
    alpha = spectral_index.reshape(-1,1,1)
    freqs = freqs.reshape(1,-1,1)
    freq_ratio = freqs / freq_ref.reshape(-1,1,1)
    fluxes = flux_ref.reshape(-1,1,1) * (freq_ratio ** alpha)
    sigmas = NP.sqrt(fluxes)
    Ef_amp = sigmas/NP.sqrt(2) * (NP.random.normal(loc=0.0, scale=1.0, size=(nsrc,nchan,1)) + 1j * NP.random.normal(loc=0.0, scale=1.0, size=(nsrc,nchan,1)))
    # Ef_amp = sigmas * NP.random.normal(loc=0.0, scale=1.0, size=(nsrc,nchan))
    # print Ef_amp
    # Ef_amp = sigmas * NP.ones((nsrc,nchan))
    # Ef_phase = NP.exp(1j*NP.random.uniform(low=0.0, high=2*NP.pi, size=(nsrc,nchan)))
    # Ef_phase = NP.exp(1j*NP.random.uniform(low=0.0, high=2*NP.pi, size=1))
    Ef_phase = 1.0
    Ef = Ef_amp * Ef_phase

    # Ef = Ef[:,:,NP.newaxis]
    skypos_dot_antpos = NP.dot(skypos-ref_point, antpos.T)
    k_dot_r_phase = 2.0 * NP.pi * freqs / FCNST.c * skypos_dot_antpos[:,NP.newaxis,:]
    Ef = voltage_pattern * Ef * NP.exp(1j * k_dot_r_phase)
    Ef = NP.sum(Ef, axis=0)
    if verbose:
        print '\tPerformed linear superposition of electric fields from source(s).'
    dictout = {}
    dictout['f'] = freqs.ravel()
    dictout['Ef'] = Ef
    dictout['antpos'] = antpos

    if verbose:
        print 'stochastic_E_spectrum() executed successfully.\n'

    return dictout

#################################################################################

def stochastic_E_timeseries(freq_center, nchan, channel_width, flux_ref=1.0,
                            freq_ref=None, spectral_index=0.0, skypos=None, 
                            ref_point=None, antpos=[0.0,0.0,0.0], spectrum=True,
                            tshift=True, voltage_pattern=None, verbose=True):

    """
    -----------------------------------------------------------------------------
    Compute a stochastic electric field timeseries obtained from sources with 
    given flux densities and spectral indices at given positions at specified 
    antenna locations. It is computed from FFT of stochastic electric field
    spectra by calling stochastic_E_spectrum().

    Inputs:

    freq_center      [float] Center frequency in Hz. Center frequency must be
                     greater than half the bandwidth.

    nchan            [integer] Number of frequency channels in spectrum

    channel_width    [float] Channel width in Hz

    Keyword Inputs:

    flux_ref         [list or numpy array of float] Flux densities of sources
                     at the respective reference frequencies. Units are 
                     arbitrary. Values have to be positive. Default = 1.0. 

    freq_ref         [list or numpy array of float] Reference frequency (Hz). 
                     If not provided, default is set to center frequency given
                     in freq_center for each of the sources. If a single value 
                     is provided, it will be applicable to all the sources. If a 
                     list or numpy array is provided, it should be of size equal
                     to that of flux_ref. 

    spectral_index   [list or numpy array of float] Spectral Index 
                     (flux ~ freq ** alpha). If not provided, default is set to 
                     zero, a flat spectrum, for each of the sources. If a single 
                     value is provided, it will be applicable to all the sources. 
                     If a list or numpy array is provided, it should be of size 
                     equal to that of flux_ref. 

    skypos           [list, tuple, list of lists, list of tuples, numpy array]
                     Sky positions of sources provided in direction cosine
                     coordinates aligned with local ENU axes. It should be a
                     3-element list, a 3-element tuple, a list of 3-element
                     lists, list of 3-element tuples, or a 3-column numpy array.
                     Each 3-element entity corresponds to a source position. 
                     Number of 3-element entities should equal the number of 
                     sources as specified by the size of flux_ref. Rules of 
                     direction cosine quantities should be followed. If only 
                     one  source is specified by flux_ref and skypos is not
                     specified, skypos defaults to the zenith (0.0, 0.0, 1.0)

    ref_point        [3-element list, tuple, or numpy vector] Point on sky used
                     as a phase reference. Same units as skypos (which is
                     direction cosines and must satisfy rules of direction
                     cosines). If None provided, it defaults to zenith
                     (0.0, 0.0, 1.0)

    antpos           [list, tuple, list of lists, list of tuples, numpy array]
                     Antenna positions provided along local ENU axes. 
                     It should be a 3-element list, a 3-element tuple, a list of 
                     3-element lists, list of 3-element tuples, or a 3-column 
                     numpy array. Each 3-element entity corresponds to an
                     antenna position. If not specified, antpos by default is 
                     assigned the origin (0.0, 0.0, 0.0).

    spectrum         [boolean] If set to True, returns the spectrum from which
                     the tiemseries was created. The spectral information 
                     (frequency and electric field spectrum) is returned with 
                     keys 'f' and 'Ef' in the returned dictionary dictout.

    voltage_pattern  [numpy array] Voltage pattern for each frequency channel
                     at each source location for each antenna. It must be of
                     shape nsrc x nchan x nant. If any of these dimensions are
                     1, it is assumed to be identical along that direction. 
                     If specified as None (default), it is assumed to be unity
                     and identical across antennas, sky locations and frequency
                     channels. 

    verbose          [boolean] If set to True, prints progress and diagnostic
                     messages. Default = True.
    
    Output:

    dictout          [dictionary] Consists of the following tags and info:
                     't'          [numpy array] time stamps in the timeseries of
                                  size nchan
                     'Et'         [complex numpy array] nchan x nant numpy array 
                                  consisting of complex stochastic electric field
                                  timeseries. nchan is the number of time steps 
                                  in the timeseries and nant is the number of
                                  antennas
                     'antpos'     [numpy array] 3-column array of antenna
                                  positions (same as the input argument antpos)
                     'f'          [numpy array] frequencies in the electric field
                                  spectrum. Same size as the timeseries. Set only
                                  if keyword input spectrum is set to True
                     'Ef'         [complex numpy array] nchan x nant numpy array 
                                  consisting of complex stochastic electric field
                                  spectrum. nchan is the number of frequency  
                                  channels in the spectrum and nant is the number 
                                  of antennas. Set only if keyword input spectrum
                                  is set to True
                     'tres'       [numpy vector] Residual delays after removal of 
                                  delays that are integral multiples of delay in 
                                  a bin of the timeseries, in the process of 
                                  phasing of antennas. It is computed only if the 
                                  input parameter 'tshift' is set to True. Length
                                  of the vector is equal to the number of
                                  antennas. If 'tshift' is set to False, the key
                                  'tres' is set to None
                     'tshift'     [numpy vector] if input parameter 'tshift' is
                                  set to True, this key 'tshift' in the output
                                  dictionary holds number of bins by which the
                                  timeseries of antennas have been shifted 
                                  (positive values indicate delay of tiemseries
                                  and negative values indicate advacing of
                                  timeseries). The size of this vector equals the
                                  number of antennas. If input parameter 'tshift'
                                  is set to False, the value in this key 'tshift' 
                                  is set to None.

    -----------------------------------------------------------------------------
    """

    if verbose:
        print '\nExecuting stochastic_E_timeseries()...'

    if ref_point is None:
        ref_point = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    elif isinstance(ref_point, (list, tuple, NP.ndarray)):
        ref_point = NP.asarray(ref_point).reshape(1,-1)
    else:
        raise TypeError('Reference position must be a list, tuple or numpy array.')

    if ref_point.size != 3:
        raise ValueError('Reference position must be a 3-element list, tuple or numpy array of direction cosines.')

    eps = 1.0e-10
    if NP.any(NP.abs(ref_point) > 1.0):
        raise ValueError('Direction cosines in reference position cannot exceed unit magnitude.')
    elif NP.abs(1.0-NP.sqrt(NP.sum(ref_point**2))) > eps:
        raise ValueError('Unit vector denoting reference position in direction cosine units must have unit magnitude.')

    if verbose:
        print '\tCalling stochastic_E_spectrum() to compute stochastic electric \n\t\tfield spectra...'

    if tshift:
        spectrum_info = stochastic_E_spectrum(freq_center, nchan, channel_width,
                                              flux_ref, freq_ref, spectral_index,
                                              skypos=skypos, antpos=antpos,
                                              voltage_pattern=voltage_pattern,
                                              verbose=verbose)
    else:
        spectrum_info = stochastic_E_spectrum(freq_center, nchan, channel_width,
                                              flux_ref, freq_ref, spectral_index,
                                              skypos=skypos, antpos=antpos,
                                              voltage_pattern=voltage_pattern,
                                              ref_point=ref_point, verbose=verbose)
        
    if verbose:
        print '\tContinuing to execute stochastic_E_timeseries()...'
        print '\tComputing timeseries from spectrum using inverse FFT.'

    Ef_shifted = NP.fft.ifftshift(spectrum_info['Ef'], axes=0)
    Et = NP.fft.ifft(Ef_shifted, axis=0)
    f = spectrum_info['f']
    t = NP.fft.fftshift(NP.fft.fftfreq(nchan, f[1]-f[0]))
    t = t - NP.amin(t)

    dictout = {}
    dictout['t'] = t
    dictout['antpos'] = spectrum_info['antpos']
    if spectrum:
        dictout['f'] = spectrum_info['f']
        dictout['Ef'] = spectrum_info['Ef']
    if tshift:
        td = NP.dot(ref_point, antpos.T)/FCNST.c
        tdbins_shift = NP.round(td/(t[1]-t[0]))
        td_residual = td - tdbins_shift * (t[1] - t[0])
        dictout['tres'] = td_residual
        dictout['tshift'] = tdbins_shift
        for i in xrange(Et.shape[1]):
            Et[:,i] = NP.roll(Et[:,i], tdbins_shift)
    dictout['Et'] = Et
   
    if verbose:
        print 'stochastic_E_timeseries() executed successfully.\n'

    return dictout

#################################################################################

def monochromatic_E_spectrum(freq, flux_ref=1.0, freq_ref=None, 
                             spectral_index=0.0, skypos=None, ref_point=None,
                             antpos=[0.0,0.0,0.0], voltage_pattern=None,
                             verbose=True):

    """
    -----------------------------------------------------------------------------
    Compute a monochromatic electric field spectrum obtained from sources with 
    given flux densities at respective reference frequencies at given positions at 
    specified antenna locations. The monochromatic spectrum corresponds to the 
    first frequency channel and the rest of the channels are just used for zero
    padding

    Inputs:

    freq             [float] Center frequency in Hz. Center frequency must be
                     greater than half the bandwidth.

    Keyword Inputs:

    flux_ref         [list or numpy array of float] Flux densities of sources
                     at the specified frequency. Units are 
                     arbitrary. Values have to be positive. Default = 1.0. 

    freq_ref         [list or numpy array of float] Reference frequency (Hz). 
                     If not provided, default is set to center frequency given
                     in freq for each of the sources. If a single value 
                     is provided, it will be applicable to all the sources. If a 
                     list or numpy array is provided, it should be of size equal
                     to that of flux_ref. 

    spectral_index   [list or numpy array of float] Spectral Index 
                     (flux ~ freq ** alpha). If not provided, default is set to 
                     zero, a flat spectrum, for each of the sources. If a single 
                     value is provided, it will be applicable to all the sources. 
                     If a list or numpy array is provided, it should be of size 
                     equal to that of flux_ref. 

    skypos           [list, tuple, list of lists, list of tuples, numpy array]
                     Sky positions of sources provided in direction cosine
                     coordinates aligned with local ENU axes. It should be a
                     3-element list, a 3-element tuple, a list of 3-element
                     lists, list of 3-element tuples, or a 3-column numpy array.
                     Each 3-element entity corresponds to a source position. 
                     Number of 3-element entities should equal the number of 
                     sources as specified by the size of flux_ref. Rules of 
                     direction cosine quantities should be followed. If only 
                     one  source is specified by flux_ref and skypos is not
                     specified, skypos defaults to the zenith (0.0, 0.0, 1.0)

    ref_point        [3-element list, tuple, or numpy vector] Point on sky used
                     as a phase reference. Same units as skypos (which is
                     direction cosines and must satisfy rules of direction
                     cosines). If None provided, it defaults to zenith
                     (0.0, 0.0, 1.0)

    antpos           [list, tuple, list of lists, list of tuples, numpy array]
                     Antenna positions provided along local ENU axes. 
                     It should be a 3-element list, a 3-element tuple, a list of 
                     3-element lists, list of 3-element tuples, or a 3-column 
                     numpy array. Each 3-element entity corresponds to an
                     antenna position. If not specified, antpos by default is 
                     assigned the origin (0.0, 0.0, 0.0).

    voltage_pattern  [numpy array] Voltage pattern for given frequency channel
                     at each source location for each antenna. It must be of
                     shape nsrc x nant. If any of these dimensions 
                     are 1, it is assumed to be identical along that direction. 
                     If specified as None (default), it is assumed to be unity
                     and identical across antennas, and sky locations

    verbose:         [boolean] If set to True, prints progress and diagnostic
                     messages. Default = True.
    
    Output:

    dictout          [dictionary] Consists of the following tags and info:
                     'f'        [numpy array] frequencies of the channels in the 
                                spectrum of size nchan
                     'Ef'       [complex numpy array] 1 x nant numpy array 
                                consisting of complex monochromatic electric 
                                field spectra. nant is the number of antennas.
                     'antpos'   [numpy array] 3-column array of antenna
                                positions (same as the input argument antpos)

    ----------------------------------------------------------------------------
    """

    if verbose:
        print '\nExecuting monochromatic_E_spectrum()...'
        print '\tChecking data compatibility...'

    try:
        freq
    except NameError:
        raise NameError('Center frequency (freq) must be provided. Aborting monochromatic_E_spectrum().')

    if not isinstance(freq, (int,float)):
        raise TypeError('    freq must be a scalar value. Aborting monochromatic_E_spectrum().')

    freq = float(freq)
    if freq <= 0.0:
        raise ValueError('    freq must be a positive value. Aborting monochromatic_E_spectrum().')

    if freq_ref is None:
        if verbose:
            print '\tNo reference frequency (freq_ref) provided. Setting it equal to center \n\t\tfrequency.'
        freq_ref = freq * NP.ones(1)

    if isinstance(freq_ref, (int,float)):
        freq_ref = NP.asarray(freq_ref).reshape(-1)
    elif isinstance(freq_ref, (list, tuple)):
        freq_ref = NP.asarray(freq_ref)
    elif isinstance(freq_ref, NP.ndarray):
        freq_ref = freq_ref.ravel()
    else:
        raise TypeError('Reference frequency (freq_ref) must be a scalar, list, tuple or numpy array. Aborting monochromatic_E_spectrum().')

    if NP.any(freq_ref <= 0.0):
        raise ValueError('freq_ref must be a positive value. Aborting monochromatic_E_spectrum().')

    if isinstance(flux_ref, (int,float)):
        flux_ref = NP.asarray(flux_ref).reshape(-1)
    elif isinstance(flux_ref, (list, tuple)):
        flux_ref = NP.asarray(flux_ref)
    elif isinstance(flux_ref, NP.ndarray):
        flux_ref = flux_ref.ravel()
    else:
        raise TypeError('Flux density at reference frequency (flux_ref) must be a scalar, list, tuple or numpy array. Aborting monochromatic_E_spectrum().')

    if NP.any(flux_ref <= 0.0):
        raise ValueError('flux_ref must be a positive value. Aborting monochromatic_E_spectrum().')

    if isinstance(spectral_index, (int,float)):
        spectral_index = NP.asarray(spectral_index).reshape(-1)
    elif isinstance(spectral_index, (list, tuple)):
        spectral_index = NP.asarray(spectral_index)
    elif isinstance(spectral_index, NP.ndarray):
        spectral_index = spectral_index.ravel()
    else:
        raise TypeError('Spectral index (spectral_index) must be a scalar, list, tuple or numpy array. Aborting monochromatic_E_spectrum().')

    nsrc = flux_ref.size
    nref = freq_ref.size
    nsi = spectral_index.size 

    if skypos is None:
        if nsrc > 1:
            raise ValueError('Sky positions (skypos) must be specified for each of the multiple flux densities.')
        skypos = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    elif isinstance(skypos, (list, tuple)):
        skypos = NP.asarray(skypos)
        if len(skypos.shape) == 1:
            if skypos.size != 3:
                raise IndexError('Sky position must be a three-element vector of direction cosines for each source, and aligned with the local ENU coordinate system.')
            else:
                skypos = skypos.reshape(1,-1)
        elif skypos.shape[1] != 3:
            raise IndexError('Sky position must be a three-element vector for each source given as direction cosines aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    elif isinstance(skypos, NP.ndarray):
        if len(skypos.shape) == 1:
            if skypos.size != 3:
                raise IndexError('Sky position must be a three-element vector for each source given as direction cosines aligned with the local ENU coordinate system.')
            else:
                skypos = skypos.reshape(1,-1)
        elif skypos.shape[1] != 3:
            raise IndexError('Sky position must be a three-element vector for each source given as direction cosines aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    else:
        raise TypeError('Sky position (skypos) must be a three-element list or tuple, list of lists or list of tuples with each of the inner lists or tuples holding three elements, or a three-column numpy array.')
            
    if ref_point is None:
        ref_point = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    elif isinstance(ref_point, (list, tuple, NP.ndarray)):
        ref_point = NP.asarray(ref_point).reshape(1,-1)
    else:
        raise TypeError('Reference position must be a list, tuple or numpy array.')

    if ref_point.size != 3:
        raise ValueError('Reference position must be a 3-element list, tuple or numpy array of direction cosines.')

    if nsrc > skypos.shape[0]:
        raise ValueError('Sky positions must be provided for each of source flux densities.')
    elif nsrc < skypos.shape[0]:
        skypos = skypos[:nsrc,:]

    eps = 1.0e-10
    if NP.any(NP.abs(skypos) > 1.0):
        raise ValueError('Some direction cosine values have absolute values greater than unity.')
    elif NP.any(NP.abs(1.0-NP.sqrt(NP.sum(skypos**2,axis=1))) > eps):
        raise ValueError('Some sky positions specified in direction cosines do not have unit magnitude by at least {0:.1e}.'.format(eps))

    if NP.any(NP.abs(ref_point) > 1.0):
        raise ValueError('Direction cosines in reference position cannot exceed unit magnitude.')
    elif NP.abs(1.0-NP.sqrt(NP.sum(ref_point**2))) > eps:
        raise ValueError('Unit vector denoting reference position in direction cosine units must have unit magnitude.')

    if nsrc == 1:
        freq_ref = NP.asarray(freq_ref[0]).reshape(-1)
        spectral_index = NP.asarray(spectral_index[0]).reshape(-1)
        nref = 1
        nsi = 1
    else: 
        if nref == 1:
            freq_ref = NP.repeat(freq_ref, nsrc)
            nref = nsrc
        elif nref != nsrc:
            raise ValueError('Number of reference frequencies should be either 1 or match the number of flux densities of sources.')

        if nsi == 1:
            spectral_index = NP.repeat(spectral_index, nsrc)
            nsi = nsrc
        elif nsi != nsrc:
            raise ValueError('Number of spectral indices should be either 1 or match the number of flux densities of sources.')

    if antpos is None:
        antpos = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    elif isinstance(antpos, (list, tuple)):
        antpos = NP.asarray(antpos)
        if len(antpos.shape) == 1:
            if antpos.size != 3:
                raise IndexError('Antenna position must be a three-element vector and aligned with the local ENU coordinate system.')
            else:
                antpos = antpos.reshape(1,-1)
        elif antpos.shape[1] != 3:
            raise IndexError('Antenna position must be a three-element vector aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    elif isinstance(antpos, NP.ndarray):
        if len(antpos.shape) == 1:
            if antpos.size != 3:
                raise IndexError('Antenna position must be a three-element vector aligned with the local ENU coordinate system.')
            else:
                antpos = antpos.reshape(1,-1)
        elif antpos.shape[1] != 3:
            raise IndexError('Antenna position must be a three-element vector aligned with the local ENU coordinate system in the form of a three-column numpy array.')
    else:
        raise TypeError('Antenna position (antpos) must be a three-element list or tuple, list of lists or list of tuples with each of the inner lists or tuples holding three elements, or a three-column numpy array.')
     
    nant = antpos.shape[0]
    nchan = 1

    if voltage_pattern is None:
        voltage_pattern = NP.ones(1).reshape(1,1)
    elif not isinstance(voltage_pattern, NP.ndarray):
        raise TypeError('Input antenna voltage pattern must be an array')

    if voltage_pattern.ndim == 1:
        voltage_pattern = voltage_pattern[:,NP.newaxis]
    elif voltage_pattern.ndim != 2:
        raise ValueError('Dimensions of voltage pattern incompatible')

    vb_shape = voltage_pattern.shape
    if (vb_shape[1] != 1) and (vb_shape[1] != nant):
        raise ValueError('Input voltage pattern must be specified for each antenna or assumed to be identical to all antennas')
    if (vb_shape[0] != 1) and (vb_shape[0] != nsrc):
        raise ValueError('Input voltage pattern must be specified at each sky location or assumed to be identical at all locations')

    if verbose:
        print '\tArguments verified for compatibility.'
        print '\tSetting up the recipe for producing monochromatic Electric field...'

    alpha = spectral_index.reshape(-1) # size nsrc
    freq_ratio = freq / freq_ref # size nchan=1
    flux = flux_ref * (freq_ratio ** alpha) # size nsrc
    sigma = NP.sqrt(flux).reshape(-1,1) # size nsrc x (nchan=1)
    Ef_amp = sigma # size nsrc x (nchan=1)
    Ef_phase = NP.random.uniform(low=0.0, high=2*NP.pi, size=(nsrc,nchan)) # size nsrc x (nchan=1)
    Ef_sky =  Ef_amp * NP.exp(1j * Ef_phase) # nsrc x (nchan=1)
    # Ef_matrix = NP.repeat(Ef_sky, nant, axis=1) # nsrc x nant
    skypos_dot_antpos = NP.dot(skypos-ref_point, antpos.T) # nsrc x nant
    k_dot_r_phase = 2.0 * NP.pi * (freq/FCNST.c) * skypos_dot_antpos # nsrc x nant
    Ef_2D = voltage_pattern * Ef_sky * NP.exp(1j * k_dot_r_phase) # nsrc x nant
    Ef = NP.sum(Ef_2D, axis=0) # nant
    if verbose:
        print '\tPerformed linear superposition of electric fields from source(s).'
    dictout = {}
    dictout['f'] = freq # nchan=1
    dictout['Ef'] = Ef # nant
    dictout['antpos'] = antpos # nant x 3

    if verbose:
        print 'monochromatic_E_spectrum() executed successfully.\n'

    return dictout

#################################################################################

def monochromatic_E_timeseries(freq_center, nchan, channel_width, flux_ref=1.0,
                               freq_ref=None, spectral_index=0.0, skypos=None, 
                               ref_point=None, antpos=[0.0,0.0,0.0],
                               spectrum=True, voltage_pattern=None,
                               verbose=True):

    """
    -----------------------------------------------------------------------------
    Compute a monochromatic electric field timeseries obtained from sources with 
    given flux densities and spectral indices at given positions at specified 
    antenna locations. It is computed from FFT of monochromatic electric field
    spectra by calling monochromatic_E_spectrum().

    Inputs:

    freq_center      [float] Center frequency in Hz. Center frequency must be
                     greater than half the bandwidth.

    nchan            [integer] Number of frequency channels in spectrum

    channel_width    [float] Channel width in Hz

    Keyword Inputs:

    flux_ref         [list or numpy array of float] Flux densities of sources
                     at the respective reference frequencies. Units are 
                     arbitrary. Values have to be positive. Default = 1.0. 

    freq_ref         [list or numpy array of float] Reference frequency (Hz). 
                     If not provided, default is set to center frequency given
                     in freq_center for each of the sources. If a single value 
                     is provided, it will be applicable to all the sources. If a 
                     list or numpy array is provided, it should be of size equal
                     to that of flux_ref. 

    spectral_index   [list or numpy array of float] Spectral Index 
                     (flux ~ freq ** alpha). If not provided, default is set to 
                     zero, a flat spectrum, for each of the sources. If a single 
                     value is provided, it will be applicable to all the sources. 
                     If a list or numpy array is provided, it should be of size 
                     equal to that of flux_ref. 

    skypos           [list, tuple, list of lists, list of tuples, numpy array]
                     Sky positions of sources provided in direction cosine
                     coordinates aligned with local ENU axes. It should be a
                     3-element list, a 3-element tuple, a list of 3-element
                     lists, list of 3-element tuples, or a 3-column numpy array.
                     Each 3-element entity corresponds to a source position. 
                     Number of 3-element entities should equal the number of 
                     sources as specified by the size of flux_ref. Rules of 
                     direction cosine quantities should be followed. If only 
                     one  source is specified by flux_ref and skypos is not
                     specified, skypos defaults to the zenith (0.0, 0.0, 1.0)

    ref_point        [3-element list, tuple, or numpy vector] Point on sky used
                     as a phase reference. Same units as skypos (which is
                     direction cosines and must satisfy rules of direction
                     cosines). If None provided, it defaults to zenith
                     (0.0, 0.0, 1.0)

    antpos           [list, tuple, list of lists, list of tuples, numpy array]
                     Antenna positions provided along local ENU axes. 
                     It should be a 3-element list, a 3-element tuple, a list of 
                     3-element lists, list of 3-element tuples, or a 3-column 
                     numpy array. Each 3-element entity corresponds to an
                     antenna position. If not specified, antpos by default is 
                     assigned the origin (0.0, 0.0, 0.0).

    spectrum         [boolean] If set to True, returns the spectrum from which
                     the tiemseries was created. The spectral information 
                     (frequency and electric field spectrum) is returned with 
                     keys 'f' and 'Ef' in the returned dictionary dictout.

    voltage_pattern  [numpy array] Voltage pattern for given frequency channel
                     at each source location for each antenna. It must be of
                     shape nsrc x nant. If any of these dimensions 
                     are 1, it is assumed to be identical along that direction. 
                     If specified as None (default), it is assumed to be unity
                     and identical across antennas, and sky locations

    verbose          [boolean] If set to True, prints progress and diagnostic
                     messages. Default = True.
    
    Output:

    dictout          [dictionary] Consists of the following tags and info:
                     't'          [numpy array] time stamps in the timeseries of
                                  size nchan
                     'Et'         [complex numpy array] nchan x nant numpy array 
                                  consisting of complex monochromatic electric field
                                  timeseries. nchan is the number of time steps 
                                  in the timeseries and nant is the number of
                                  antennas
                     'antpos'     [numpy array] 3-column array of antenna
                                  positions (same as the input argument antpos)
                     'f'          [numpy array] frequencies in the electric field
                                  spectrum. Same size as the timeseries. Set only
                                  if keyword input spectrum is set to True
                     'Ef'         [complex numpy array] nchan x nant numpy array 
                                  consisting of complex monochromatic electric field
                                  spectrum. nchan is the number of frequency  
                                  channels in the spectrum and nant is the number 
                                  of antennas. Set only if keyword input spectrum
                                  is set to True

    -----------------------------------------------------------------------------
    """

    if verbose:
        print '\nExecuting monochromatic_E_timeseries()...'

    if verbose:
        print '\tCalling monochromatic_E_spectrum() to compute monochromatic electric \n\t\tfield spectra...'

    spectrum_info = monochromatic_E_spectrum(freq_center, flux_ref, freq_ref, 
                                             spectral_index, skypos=skypos,
                                             ref_point=ref_point, antpos=antpos,
                                             voltage_pattern=voltage_pattern,
                                             verbose=verbose)

    if verbose:
        print '\tContinuing to execute monochromatic_E_timeseries()...'
        print '\tComputing timeseries from spectrum using inverse FFT.'

    center_channel = int(NP.floor(0.5*nchan))
    Ef = spectrum_info['Ef']
    Ef_2D = NP.zeros((nchan, Ef.size), dtype=NP.complex_)
    Ef_2D[center_channel] = Ef

    Ef_2D_shifted = NP.fft.ifftshift(Ef_2D, axes=0)
    Et_2D = NP.fft.ifft(Ef_2D_shifted, axis=0)
    f = freq_center + (NP.arange(nchan)-center_channel) * channel_width
    t = NP.fft.fftshift(NP.fft.fftfreq(nchan, f[1]-f[0]))
    t = t - NP.amin(t)

    dictout = {}
    dictout['t'] = t
    dictout['Et'] = Et_2D
    dictout['antpos'] = spectrum_info['antpos']
    if spectrum:
        dictout['f'] = f
        dictout['Ef'] = Ef_2D
    
    if verbose:
        print 'monochromatic_E_timeseries() executed successfully.\n'

    return dictout

#################################################################################
 
class AntennaArraySimulator(object):

    """
    ------------------------------------------------------------------------
    Class to manage simulation information generated by an array of antennas

    Attributes:

    antenna_array   [Instance of class AntennaArray] An instance of class
                    AntennaArray which the simulator instance will use in 
                    simulating data

    skymodel        [Instance of class SkyModel] An instance of class 
                    SkyModel which the simulator will use in simulating 
                    data

    latitude        [float] Latitude (in degrees) of Observatory (antenna 
                    locations)

    longitude       [float] Longitude (in degrees) of Observatory (antenna
                    locations)

    identical_antennas
                    [boolean] If False, antennas are not assumed to be 
                    identical, otherwise they are

    f               [Numpy array] Frequency channels (in Hz)

    f0              [Scalar] Center frequency of the observing band (in Hz)

    t               [Numpy array] Time samples in a single Nyquist sampled 
                    series (in sec)

    timestamp       [float] Dublin Julian Date will be used as the timestamp
                    of the observation

    timestamps      [list] List of Dublian Julian Dates one for each nyquist
                    timeseries in the contiguous observation

    obsmode         [string] Specifies observing mode. Accepted values are
                    'drift', 'track' and 'custom' (default)

    antinfo         [dictionary] contains the following keys and 
                    information:
                    'labels':    list of strings of antenna labels
                    'positions': position vectors of antennas (3-column 
                                 array) in local ENU coordinates

    observer        [instance of class Observer in module ephem] Instance 
                    class Observer in ephem module to hold information 
                    about LST, transit time, etc.

    Ef_info         [dictionary] Consists of E-field spectral info under two 
                    keys 'P1' and 'P2', one for each polarization. Under each 
                    of these keys is a nchan x nant complex numpy array 
                    consisting of complex stochastic electric field
                    spectra. nchan is the number of channels in the 
                    spectrum and nant is the number of antennas.

    Ef_stack        [dictionary] contains the E-field spectrum under keys
                    'P1' and 'P2' for each polarization. The value under
                    each key is a complex numpy array of shape 
                    nchan x nant x ntimes. Absent data are represented by 
                    NaN values

    Et_info         [dictionary] Consists of E-field timeseries info under 
                    two keys 'P1' and 'P2', one for each polarization. Under 
                    each of these keys is a nchan x nant complex numpy array 
                    consisting of complex stochastic electric field
                    timeseries. nchan is the number of channels in the 
                    spectrum and nant is the number of antennas.

    Et_stack        [dictionary] contains the E-field timeseries under keys
                    'P1' and 'P2' for each polarization. The value under
                    each key is a complex numpy array of shape 
                    nchan x nant x ntimes. Absent data are represented by 
                    NaN values

    Member function:

    __init__()      Initialize the AntennaArraySimulator class which manages 
                    information about the simulation of Electrc fields by 
                    the antennas

    upper_hemisphere()
                    Return the indices of locations in the catalog that are 
                    in the upper celestial hemisphere for a given LST on a 
                    given date of observation

    load_voltage_pattern()
                    Generates (by interpolating if necessary) voltage 
                    pattern at the location of catalog sources based on 
                    external voltage pattern files specified. Parallel 
                    processing can be performed.

    generate_E_spectrum()
                    Compute a stochastic electric field spectrum obtained 
                    from sources in the catalog. It can be parallelized.

    generate_E_spectrum()
                    Compute a stochastic electric field spectrum obtained 
                    from a sky model using aperture plane computations. The 
                    antenna kernel is not applied here. It is a component 
                    in creating an aperture plane alternative to the member 
                    function generate_E_spectrum() but without application 
                    of the individual antenna pattern

    stack_E_spectrum()
                    Stack E-field spectra along time-axis

    generate_E_timeseries()
                    Generate E-field timeseries from their spectra. It can 
                    be done on current or stacked spectra

    generate_voltage_pattern()
                    Generate voltage pattern analytically based on antenna 
                    shapes. Can be parallelized

    observe()       Simulate a single observation and record antenna 
                    electric fields as a function of polarization, 
                    frequencies and antennas.

    observing_run() Simulate a observing run made of multiple contiguous 
                    observations and record antenna electric fields as a 
                    function of polarization, frequencies, antennas, and 
                    time.

    save()          Save information instance of class 
                    AntennaArraySimulator to external file in HDF5 format
    ------------------------------------------------------------------------
    """

    def __init__(self, antenna_array, skymodel, identical_antennas=False):

        """
        ------------------------------------------------------------------------
        Initialize the AntennaArraySimulator class which manages information 
        about the simulation of Electrc fields by the antennas

        Class attributes initialized are:
        antenna_array, skymodel, latitude, f, f0, antinfo, observer, Ef_stack,
        Ef_info, t, timestamp, timestamps

        Read docstring of class AntennaArray for details on these attributes.

        Inputs:
    
        antenna_array 
                   [Instance of class AntennaArray] An instance of class
                   AntennaArray which the simulator instance will be 
                   initialized with

        skymodel   [Instance of class SkyModel] An instance of class SkyModel
                   which the simulator will be initialized with. 

        identical_antennas
                   [boolean] If False (default), antennas will not be assumed 
                   to be identical, otherwise they are identical
        ------------------------------------------------------------------------
        """

        try:
            antenna_array
        except NameError:
            raise NameError('Input antenna_array must be specified')

        try:
            skymodel
        except NameError:
            raise NameError('Input sky model must be specified')
        
        if not isinstance(antenna_array, AA.AntennaArray):
            raise TypeError('Input antenna_array must be an instance of class AntennaArray')

        if not isinstance(skymodel, SM.SkyModel):
            raise TypeError('Input skymodel must be an instance of class SkyModel')

        if not isinstance(identical_antennas, bool):
            raise TypeError('Whether antennas are identical or not must be specified as a boolean value')

        self.antenna_array = antenna_array
        self.skymodel = skymodel
        self.identical_antennas = identical_antennas
        self.Ef_info = {}
        self.Et_info = {}
        self.Ef_stack = {}
        self.Et_stack = {}
        self.timestamp = None
        self.timestamps = []
        self.obsmode = 'custom'

        self.latitude = self.antenna_array.latitude
        self.longitude = self.antenna_array.longitude
        self.f = self.antenna_array.f[::2]
        self.f0 = self.antenna_array.f0
        t = NP.fft.fftshift(NP.fft.fftfreq(self.f.size, self.f[1]-self.f[0]))
        self.t = t - NP.amin(t)
        
        self.antinfo = self.antenna_array.antenna_positions(pol=None, flag=False, sort=True, centering=True)
        self.observer = EP.Observer()
        self.observer.lat = NP.radians(self.latitude)
        self.observer.lon = NP.radians(self.longitude)
        self.observer.date = self.skymodel.epoch.strip('J')

    ############################################################################

    def upper_hemisphere(self, lst, obs_date=None):

        """
        ------------------------------------------------------------------------
        Return the indices of locations in the catalog that are in the upper
        celestial hemisphere for a given LST on a given date of observation

        Inputs:

        lst        [scalar] Local Sidereal Time (in hours) in the range 0--24
                   on the date specified by obs_date.

        obs_date   [string] Date of observation in YYYY/MM/DD format. If set to
                   None (default), the epoch in the sky model will be assumed
                   to be the date of observation. 

        Outputs:

        hemind     [numpy array] indices of object locations in the sky model
                   which lie in the upper celestial hemisphere that will 
                   contribute to the simulated signal
        ------------------------------------------------------------------------
        """

        try:
            lst
        except NameError:
            raise NameError('Input LST must be specified')

        if obs_date is None:
            obs_date = self.observer.date

        lstobj = EP.FixedBody()
        lstobj._epoch = obs_date
        lstobj._ra = NP.radians(lst * 15.0)
        lstobj._dec = NP.radians(self.latitude)
        lstobj.compute(self.observer)
        
        ha = NP.degrees(lstobj.ra) - self.skymodel.location[:,0]
        dec = self.skymodel.location[:,1]
        altaz = GEOM.hadec2altaz(NP.hstack((ha.reshape(-1,1), dec.reshape(-1,1))), self.latitude, units='degrees')
        hemind, = NP.where(altaz[:,0] >= 0.0)
        return (hemind, altaz[hemind,:])

    ############################################################################
    
    def load_voltage_pattern(self, vbeam_files, altaz, parallel=False,
                             nproc=None):

        """
        ------------------------------------------------------------------------
        Generates (by interpolating if necessary) voltage pattern at the 
        location of catalog sources based on external voltage pattern files
        specified. Parallel processing can be performed.

        Inputs:

        vbeam_files [dictionary] Dictionary containing file locations of 
                    far-field voltage patterns. It is specified under keys
                    'P1' and 'P2' denoting the two polarizations. Under each
                    polarization key is another dictionary with keys for 
                    individual antennas denoted by antenna labels (string). 
                    If there is only one antenna key it will be assumed to be 
                    identical for all antennas. If multiple voltage beam file 
                    locations are specified, it must be the same as number of 
                    antennas 

        altaz       [numpy array] The altitudes and azimuths (in degrees) at 
                    which the voltage pattern is to be estimated. It must be
                    a nsrc x 2 array. 

        parallel    [boolean] specifies if parallelization is to be invoked. 
                    False (default) means only serial processing

        nproc       [integer] specifies number of independent processes to 
                    spawn. Default = None, means automatically determines the 
                    number of process cores in the system and use one less 
                    than that to avoid locking the system for other processes. 
                    Applies only if input parameter 'parallel' (see above) is 
                    set to True. If nproc is set to a value more than the 
                    number of process cores in the system, it will be reset to 
                    number of process cores in the system minus one to avoid 
                    locking the system out for other processes

        Outputs:

        Dictionary containing antenna voltage beams under each polarization key
        'P1' and 'P2' at the object locations in the upper hemisphere.
        The voltage beams under each polarization key are a numpy array of 
        shape nsrc x nchan x nant in case of non-indentical antennas or 
        nsrc x nchan x 1 in case of identical antennas. 
        ------------------------------------------------------------------------
        """

        try:
            vbeam_files
        except NameError:
            raise NameError('Input vbeam_files must be specified')

        try:
            altaz
        except NameError:
            raise NameError('Input altitude-azimuth must be specified')

        if not isinstance(vbeam_files, dict):
            raise TypeError('Input vbeam_files must be a dictionary')

        if not isinstance(altaz, NP.ndarray):
            raise TypeError('Input altaz must be a numpy array')

        if altaz.ndim != 2:
            raise ValueError('Input lataz must be a nsrc x 2 numpy array')
        if altaz.shape[1] != 2:
            raise ValueError('Input lataz must be a nsrc x 2 numpy array')

        theta_phi = NP.hstack((altaz[:,0].reshape(-1,1), altaz[:,1].reshape(-1,1)))
        theta_phi = NP.radians(theta_phi)

        antkeys = NP.asarray(self.antenna_array.antennas.keys())
        vbeams = {}
        for pol in ['P1', 'P2']:
            vbeams[pol] = None
            if pol in vbeam_files:
                vbeamkeys = NP.asarray(vbeam_files[pol].keys())
                commonkeys = NP.intersect1d(antkeys, vbeamkeys)
                commonkeys = NP.sort(commonkeys)
        
                if (commonkeys.size != 1) and (commonkeys.size != antkeys.size):
                    raise ValueError('Number of voltage pattern files incompatible with number of antennas')
        
                if (commonkeys.size == 1) or self.identical_antennas:
                    vbeams[pol] = interp_beam(vbeam_files[pol][commonkeys[0]], theta_phi, self.f)
                    vbeams[pol] = vbeams[pol][:,:,NP.newaxis] # nsrc x nchan x 1
                else:
                    if parallel or (nproc is not None):
                        list_of_keys = commonkeys.tolist()
                        list_of_vbeam_files = [vbeam_files[pol][akey] for akey in list_of_keys]
                        list_of_zaaz = [theta_phi] * commonkeys.size
                        list_of_obsfreqs = [self.f] * commonkeys.size
                        if nproc is None:
                            nproc = max(MP.cpu_count()-1, 1) 
                        else:
                            nproc = min(nproc, max(MP.cpu_count()-1, 1))
                        pool = MP.Pool(processes=nproc)
                        list_of_vbeams = pool.map(interp_beam_arg_splitter, IT.izip(list_of_vbeam_files, list_of_zaaz, list_of_obsfreqs))
                        vbeams[pol] = NP.asarray(list_of_vbeams) # nsrc x nchan x nant
                        del list_of_vbeams
                    else:
                        for key in commonkeys:
                            vbeam = interp_beam(vbeam_files[pol][key], theta_phi, self.f)
                            if vbeams[pol] is None:
                                vbeams[pol] = vbeam[:,:,NP.newaxis] # nsrc x nchan x 1
                            else:
                                vbeams[pol] = NP.dstack((vbeams[pol], vbeam[:,:,NP.newaxis])) # nsrc x nchan x nant

        return vbeams
        
    ############################################################################

    def generate_voltage_pattern(self, altaz, pointing_center=None,
                                 pointing_info=None, short_dipole_approx=False,
                                 half_wave_dipole_approx=False, parallel=False,
                                 nproc=None):

        """
        ------------------------------------------------------------------------
        Generate voltage pattern analytically based on antenna shapes. Can be
        parallelized

        Inputs:

        altaz       [numpy array] The altitudes and azimuths (in degrees) at 
                    which the voltage pattern is to be estimated. It must be
                    a nsrc x 2 array. 

        pointing_center
                    [list or numpy array] coordinates of pointing center (in 
                    the same coordinate system as that of sky coordinates 
                    specified by skyunits). 2-element vector if 
                    skyunits='altaz'. 2- or 3-element vector if 
                    skyunits='dircos'. Only used with phased array primary 
                    beams, dishes excluding VLA and GMRT, or uniform rectangular 
                    or square apertures. For all telescopes except MWA, 
                    pointing_center is used in place of pointing_info. For MWA, 
                    this is used if pointing_info is not provided.

        pointing_info 
                    [dictionary] A dictionary consisting of information 
                    relating to pointing center in case of a phased array. 
                    The pointing center can be specified either via element 
                    delay compensation or by directly specifying the pointing 
                    center in a certain coordinate system. Default = None 
                    (pointing centered at zenith). This dictionary consists of 
                    the following tags and values:
                    'gains'           [numpy array] Complex element gains. 
                                      Must be of size equal to the number of 
                                      elements as specified by the number of 
                                      rows in antpos. If set to None (default), 
                                      all element gains are assumed to be unity. 
                                      Used only in phased array mode.
                    'gainerr'         [int, float] RMS error in voltage 
                                      amplitude in dB to be used in the 
                                      beamformer. Random jitters are drawn from 
                                      a normal distribution in logarithm units 
                                      which are then converted to linear units. 
                                      Must be a non-negative scalar. If not 
                                      provided, it defaults to 0 (no jitter).
                                      Used only in phased array mode.
                    'delays'          [numpy array] Delays (in seconds) to be 
                                      applied to the tile elements. Size should 
                                      be equal to number of tile elements 
                                      (number of rows in antpos). Default=None 
                                      will set all element delays to zero 
                                      phasing them to zenith. Used only in 
                                      phased array mode. 
                    'pointing_center' [numpy array] This will apply in the 
                                      absence of key 'delays'. This can be 
                                      specified as a row vector. Should have 
                                      two-columns if using Alt-Az coordinates, 
                                      or two or three columns if using direction 
                                      cosines. There is no default. The
                                      coordinate system must be specified in
                                      'pointing_coords' if 'pointing_center' is 
                                      to be used.
                    'pointing_coords' [string scalar] Coordinate system in which 
                                      the pointing_center is specified. Accepted 
                                      values are 'altaz' or 'dircos'. Must be 
                                      provided if 'pointing_center' is to be 
                                      used. No default.
                    'delayerr'        [int, float] RMS jitter in delays used in 
                                      the beamformer. Random jitters are drawn 
                                      from a normal distribution with this rms. 
                                      Must be a non-negative scalar. If not 
                                      provided, it defaults to 0 (no jitter). 
                                      Used only in phased array mode.
        
        short_dipole_approx
                    [boolean] if True, indicates short dipole approximation
                    is to be used. Otherwise, a more accurate expression is 
                    used for the dipole pattern. Default=False. Both
                    short_dipole_approx and half_wave_dipole_approx cannot be 
                    set to True at the same time
        
        half_wave_dipole_approx
                    [boolean] if True, indicates half-wave dipole approximation
                    is to be used. Otherwise, a more accurate expression is 
                    used for the dipole pattern. Default=False

        parallel    [boolean] specifies if parallelization is to be invoked. 
                    False (default) means only serial processing

        nproc       [integer] specifies number of independent processes to 
                    spawn. Default = None, means automatically determines the 
                    number of process cores in the system and use one less 
                    than that to avoid locking the system for other processes. 
                    Applies only if input parameter 'parallel' (see above) is 
                    set to True. If nproc is set to a value more than the 
                    number of process cores in the system, it will be reset to 
                    number of process cores in the system minus one to avoid 
                    locking the system out for other processes

        Outputs:

        Dictionary containing antenna voltage beams under each polarization key
        'P1' and 'P2' at the object locations in the upper hemisphere.
        The voltage beams under each polarization key are a numpy array of 
        shape nsrc x nchan x nant in case of non-indentical antennas or 
        nsrc x nchan x 1 in case of identical antennas. 
        ------------------------------------------------------------------------
        """

        try:
            altaz
        except NameError:
            raise NameError('Input altitude-azimuth must be specified')

        if not isinstance(altaz, NP.ndarray):
            raise TypeError('Input altaz must be a numpy array')

        if altaz.ndim != 2:
            raise ValueError('Input lataz must be a nsrc x 2 numpy array')
        if altaz.shape[1] != 2:
            raise ValueError('Input lataz must be a nsrc x 2 numpy array')

        telescopes = {}
        for pol in ['P1', 'P2']:
            telescopes[pol] = {}
            if self.identical_antennas:
                telescopes[pol][self.antenna_array.antennas.itervalues().next().label] = {}
                # telescopes[pol][self.antenna_array.antennas.itervalues().next().label]['id'] = 'custom'
                if self.antenna_array.antennas.itervalues().next().aperture.shape[pol] == 'circular':
                    telescopes[pol][self.antenna_array.antennas.itervalues().next().label]['shape'] = 'dish'
                    telescopes[pol][self.antenna_array.antennas.itervalues().next().label]['size'] = 2.0 * self.antenna_array.antennas.itervalues().next().aperture.rmax[pol]
                    telescopes[pol][self.antenna_array.antennas.itervalues().next().label]['orientation'] = NP.asarray([90.0, 270.0])
                    telescopes[pol][self.antenna_array.antennas.itervalues().next().label]['ocoords'] = 'altaz'
                elif (self.antenna_array.antennas.itervalues().next().aperture.shape[pol] == 'rect') or (self.antenna_array.antennas.itervalues().next().aperture.shape[pol] == 'square'):
                    telescopes[pol][self.antenna_array.antennas.itervalues().next().label]['shape'] = 'rect'
                    telescopes[pol][self.antenna_array.antennas.itervalues().next().label]['size'] = 2.0 * NP.asarray([self.antenna_array.antennas.itervalues().next().aperture.xmax[pol], self.antenna_array.antennas.itervalues().next().aperture.ymax[pol]])
                    telescopes[pol][self.antenna_array.antennas.itervalues().next().label]['orientation'] = NP.degrees(self.antenna_array.antennas.itervalues().next().aperture.rotangle[pol])
                    telescopes[pol][self.antenna_array.antennas.itervalues().next().label]['ocoords'] = 'degrees'
                else:
                    raise ValueError('Antenna aperture shape currently not supported for analytic antenna beam estimation')
            else:
                for antkey in sorted(self.antenna_array.antennas.keys()):
                    telescopes[pol][self.antenna_array.antennas[antkey].label] = {}
                    # telescopes[pol][self.antenna_array.antennas[antkey].label]['id'] = 'custom'
                    if self.antenna_array.antennas[antkey].aperture.shape[pol] == 'circular':
                        telescopes[pol][self.antenna_array.antennas[antkey].label]['shape'] = 'dish'
                        telescopes[pol][self.antenna_array.antennas[antkey].label]['size'] = 2.0 * self.antenna_array.antennas[antkey].aperture.rmax[pol]
                        telescopes[pol][self.antenna_array.antennas[antkey].label]['orientation'] = NP.asarray([90.0, 270.0])
                        telescopes[pol][self.antenna_array.antennas[antkey].label]['ocoords'] = 'altaz'
                    elif (self.antenna_array.antennas[antkey].aperture.shape[pol] == 'rect') or (self.antenna_array.antennas[antkey].aperture.shape[pol] == 'square'):
                        telescopes[pol][self.antenna_array.antennas[antkey].label]['shape'] = 'rect'
                        telescopes[pol][self.antenna_array.antennas[antkey].label]['size'] = 2.0 * NP.asarray([self.antenna_array.antennas[antkey].aperture.xmax[pol], self.antenna_array.antennas[antkey].aperture.ymax[pol]])
                        telescopes[pol][self.antenna_array.antennas[antkey].label]['orientation'] = NP.degrees(self.antenna_array.antennas[antkey].aperture.rotangle[pol])
                        telescopes[pol][self.antenna_array.antennas[antkey].label]['ocoords'] = 'degrees'
                    else:
                        raise ValueError('Antenna aperture shape currently not supported for analytic antenna beam estimation')
                    
        vbeams = {}
        for pol in ['P1', 'P2']:
            vbeams[pol] = None
            antkeys = sorted(telescopes[pol].keys())
            if len(antkeys) == 1:
                vbeams[pol] = AB.antenna_beam_generator(altaz, self.f, telescopes[pol][antkeys[0]], freq_scale='Hz', skyunits='altaz', east2ax1=telescopes[pol][antkeys[0]]['orientation'], pointing_info=pointing_info, pointing_center=pointing_center, short_dipole_approx=short_dipole_approx, half_wave_dipole_approx=half_wave_dipole_approx, power=False)
                vbeams[pol] = vbeams[pol][:,:,NP.newaxis] # nsrc x nchan x 1
            else:
                if parallel or (nproc is not None):
                    list_of_keys = antkeys
                    list_of_telescopes = [telescopes[pol][akey] for akey in list_of_keys]
                    list_of_altaz = [altaz] * len(antkeys)
                    list_of_obsfreqs = [self.f] * len(antkeys)
                    list_of_freqscale = ['Hz'] * len(antkeys)
                    list_of_skyunits = ['altaz'] * len(antkeys)
                    list_of_east2ax1 = [telescopes[pol][antkey]['orientation'] for antkey in antkeys]
                    list_of_pointing_info = [pointing_info] * len(antkeys)
                    list_of_pointing_center = [pointing_center] * len(antkeys)
                    list_of_short_dipole_approx = [short_dipole_approx] * len(antkeys)
                    list_of_half_wave_dipole_approx = [half_wave_dipole_approx] * len(antkeys)
                    list_of_powertrue = [False] * len(antkeys)
                    if nproc is None:
                        nproc = max(MP.cpu_count()-1, 1) 
                    else:
                        nproc = min(nproc, max(MP.cpu_count()-1, 1))
                    pool = MP.Pool(processes=nproc)
                    list_of_vbeams = pool.map(AB.antenna_beam_arg_splitter, IT.izip(list_of_altaz, list_of_obsfreqs, list_of_telescopes, list_of_freqscale, list_of_skyunits, list_of_east2ax1, list_of_pointing_info, list_of_pointing_center, list_of_short_dipole_approx, list_of_half_wave_dipole_approx, list_of_powertrue))
                    vbeams[pol] = NP.asarray(list_of_vbeams) # nant x nsrc x nchan
                    vbeams[pol] = NP.rollaxis(vbeams[pol], 0, start=3) # nsrc x nchan x nant
                    del list_of_vbeams
                else:
                    for key in antkeys:
                        vbeam = AB.antenna_beam_generator(altaz, self.f, telescopes[pol][key], freq_scale='Hz', skyunits='altaz', east2ax1=telescopes[pol][key]['orientation'], pointing_info=pointing_info, pointing_center=pointing_center, short_dipole_approx=short_dipole_approx, half_wave_dipole_approx=half_wave_dipole_approx, power=False)
                        if vbeams[pol] is None:
                            vbeams[pol] = vbeam[:,:,NP.newaxis] # nsrc x nchan x 1
                        else:
                            vbeams[pol] = NP.dstack((vbeams[pol], vbeam[:,:,NP.newaxis])) # nsrc x nchan x nant

        return vbeams

    ############################################################################

    def generate_E_spectrum(self, altaz, vbeams, vbeamkeys=None, ctlgind=None,
                            pol=None, ref_point=None, randomseed=None,
                            parallel=False, nproc=None, action=None,
                            verbose=True):

        """
        ------------------------------------------------------------------------
        Compute a stochastic electric field spectrum obtained from sources in 
        the catalog. It can be parallelized.
    
        Inputs:
    
        altaz     [numpy array] Alt-az sky positions (in degrees) of sources 
                  It should be a 2-column numpy array. Each 2-column entity 
                  corresponds to a source position. Number of 2-column 
                  entities should equal the number of sources as specified 
                  by the size of flux_ref. It is of size nsrc x 2
    
        vbeams    [dictionary] Complex Voltage pattern for each each antenna 
                  and each polarization at each frequency channel at each 
                  source location. It must be specified as a dictionary with 
                  keys denoting antenna labels. Under each antenna label as key 
                  it contains a dictionary with keys 'P1' and 'P2' denoting
                  the two polarizations. Under each of these keys the voltage 
                  pattern is specified as a numpy array of size nsrc x nchan. 
                  If only one antenna label is specified as key, it will be 
                  assumed to be identical for all antennas. Also if nchan 
                  is 1, it will be assumed to be achromatic and identical 
                  across frequency. No default.

        Keyword Inputs:

        ctlgind   [numpy array] Indices of sources in the attribute skymodel 
                  that will be used in generating the E-field spectrum. If
                  specified as None (default), all objects in the attribute
                  skymodel will be used. It size must be of size nsrc as 
                  described in input altaz

        pol       [list] List of polarizations to process. The polarizations
                  are specified as strings 'P1' and 'P2. If set to None
                  (default), both polarizations are processed
    
        ref_point [3-element list, tuple, or numpy vector] Point on sky used
                  as a phase reference in direction cosines and must satisfy 
                  rules of direction cosines. If None provided, it defaults 
                  to zenith (0.0, 0.0, 1.0)
    
        randomseed
                  [integer] Seed to initialize the randon generator. If set
                  to None (default), the random sequences generated are not
                  reproducible. Set to an integer to generate reproducible
                  random sequences

        parallel  [boolean] specifies if parallelization is to be invoked. 
                  False (default) means only serial processing. Highly 
                  recommended to set to False as overheads in parallelization
                  slow it down.

        nproc     [integer] specifies number of independent processes to spawn.
                  Default = None, means automatically determines the number of 
                  process cores in the system and use one less than that to 
                  avoid locking the system for other processes. Applies only 
                  if input parameter 'parallel' (see above) is set to True. 
                  If nproc is set to a value more than the number of process
                  cores in the system, it will be reset to number of process 
                  cores in the system minus one to avoid locking the system out 
                  for other processes

        action    [string or None] If set to 'return' the computed E-field
                  spectrum is returned. If None or anything else, the computed
                  E-field spectrum is stored as an attribute but not returned

        verbose   [Boolean] Default = False. If set to True, prints some 
                  diagnotic or progress messages.

        Output:
    
        Ef_info   [dictionary] Consists of E-field info under two keys 'P1' and
                  'P2', one for each polarization. Under each of these keys
                  the complex electric fields spectra of shape nchan x nant are 
                  stored. nchan is the number of channels in the spectrum and 
                  nant is the number of antennas.
        ------------------------------------------------------------------------
        """

        try:
            altaz
        except NameError:
            raise NameError('Input altaz must be specified')

        try:
            vbeams
        except NameError:
            raise NameError('Input vbeams must be specified')
       
        if not isinstance(vbeams, dict):
            raise TypeError('Input vbeams must be a dictionary')

        if vbeamkeys is not None:
            if not isinstance(vbeamkeys, dict):
                raise TypeError('Input vbeamkeys must be a dictionary')
            for apol in ['P1', 'P2']:
                if apol not in vbeamkeys:
                    vbeamkeys[apol] = []
                if not isinstance(vbeamkeys[apol], list):
                    raise TypeError('vbeamkeys under each polarization must be a list of antenna keys')
        else:
            vbeamkeys = {}
            for apol in ['P1', 'P2']:
                vbeamkeys[apol] = []

        for apol in ['P1', 'P2']:
            nant = vbeams[apol].shape[2]
            if vbeams[apol].shape[2] > 1:
                if vbeams[apol].shape[2] != len(self.antenna_array.antennas):
                    raise ValueError('Number of antennas in vbeams incompatible with that in attribute antenna_array')

            if len(vbeamkeys[apol]) == 0:
                if nant == 1:
                    vbeamkeys[apol] = [self.antenna_array.antennas.iterkeys().next()]
                else:
                    vbeamkeys[apol] = sorted(self.antenna_array.antennas.keys())
            elif len(vbeamkeys[apol] != nant):
                raise ValueError('Number of antennas in vbeams and vbeamkeys mismatch')
            vbkeys_sortind = NP.argsort(NP.asarray(vbeamkeys[apol]))
            vbeams[apol] = vbeams[apol][:,:,vbkeys_sortind]
            vbeamkeys[apol] = NP.asarray(vbeamkeys[apol])[vbkeys_sortind]

        srcdircos = GEOM.altaz2dircos(altaz, units='degrees')

        if ctlgind is None:
            ctlgind = NP.arange(self.skymodel.location.shape[0])
        elif isinstance(ctlgind, list):
            ctlgind = NP.asarray(ctlgind)
        elif isinstance(ctlgind, NP.ndarray):
            ctlgind = ctlgind.ravel()
        else:
            raise TypeError('Input ctlgind must be a list, numpy array or set to None')
        if ctlgind.size != altaz.shape[0]:
            raise ValueError('Input ctlgind must contain same number of elements as number of objects in input altaz.')
        skymodel = self.skymodel.subset(ctlgind, axis='position')
        nsrc = ctlgind.size
        nchan = self.f.size

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

        if randomseed is None:
            randomseed = NP.random.randint(1000000)
        elif not isinstance(randomseed, int):
            raise TypeError('If input randomseed is not None, it must be an integer')

        antkeys_sortind = NP.argsort(NP.asarray(self.antinfo['labels'], dtype='|S0'))
        antpos = self.antinfo['positions']
        antkeys_sorted = NP.asarray(self.antinfo['labels'], dtype='|S0')[antkeys_sortind]
        antpos_sorted = antpos[antkeys_sortind,:]
        Ef_info = {}
        for apol in ['P1', 'P2']:
            commonkeys = NP.intersect1d(antkeys_sorted, vbeamkeys[apol])
            commonkeys = NP.sort(commonkeys)

            if (commonkeys.size != 1) and (commonkeys.size != antkeys_sorted.size):
                raise ValueError('Number of voltage pattern files incompatible with number of antennas')
        
            if apol == 'P2':
                randomseed = randomseed + 1000000

            randstate = NP.random.RandomState(randomseed)
            randvals = randstate.normal(loc=0.0, scale=1.0, size=(nsrc,nchan)) + 1j * randstate.normal(loc=0.0, scale=1.0, size=(nsrc,nchan)) # nsrc x nchan

            if parallel or (nproc is not None):
                if nproc is None:
                    nproc = max(MP.cpu_count()-1, 1) 
                else:
                    nproc = min(nproc, max(MP.cpu_count()-1, 1))
                split_ind = NP.arange(nproc, nchan, nproc)
                list_split_freqs = NP.array_split(self.f, split_ind, axis=0)
                list_split_vbeams = NP.array_split(vbeams[apol], split_ind, axis=1)
                list_antpos = [antpos_sorted] * (len(split_ind) + 1)
                list_skypos = [srcdircos] * (len(split_ind) + 1)
                list_flux_ref = [skymodel.spec_parms['flux-scale']] * (len(split_ind) + 1)
                list_freq_ref = [skymodel.spec_parms['freq-ref']] * (len(split_ind) + 1)
                list_spindex = [skymodel.spec_parms['power-law-index']] * (len(split_ind) + 1)
                list_spectrum = [None] * (len(split_ind) + 1)
                list_refpoint = [ref_point] * (len(split_ind) + 1)
                list_randomseed = [None] * (len(split_ind) + 1)
                list_randvals = NP.array_split(randvals, split_ind, axis=1)
                list_verbose = [verbose] * (len(split_ind) + 1)
                
                pool = MP.Pool(processes=nproc)
                Ef_info_list = pool.map(generate_E_spectrum_arg_splitter, IT.izip(list_split_freqs, list_skypos, list_flux_ref, list_freq_ref, list_spindex, list_spectrum, list_antpos, list_split_vbeams, list_refpoint, list_randomseed, list_randvals, list_verbose))
                Ef_info[apol] = None
                for chunk,item in enumerate(Ef_info_list):
                    if Ef_info[apol] is None:
                        Ef_info[apol] = item['Ef']
                    else:
                        Ef_info[apol] = NP.vstack((Ef_info[apol], item['Ef']))
                del Ef_info_list
                self.Ef_info[apol] = Ef_info[apol]
            else:
                Ef_info[apol] = generate_E_spectrum(self.f, skypos=srcdircos, flux_ref=skymodel.spec_parms['flux-scale'], freq_ref=skymodel.spec_parms['freq-ref'], spectral_index=skymodel.spec_parms['power-law-index'], spectrum=None, antpos=antpos_sorted, voltage_pattern=vbeams[apol], ref_point=ref_point, randomseed=randomseed, randvals=randvals, verbose=verbose)
                self.Ef_info[apol] = Ef_info[apol]['Ef']
        if action == 'return':
            return self.Ef_info

    ############################################################################
    
    def generate_sky_E_spectrum(self, altaz, ctlgind=None, uvlocs=None, 
                                pol=None, randomseed=None, randvals=None):

        """
        ------------------------------------------------------------------------
        Compute a stochastic electric field spectrum obtained from a sky model 
        using aperture plane computations. The antenna kernel is not applied 
        here. It is a component in creating an aperture plane alternative to 
        the member function generate_E_spectrum() but without application of 
        the individual antenna pattern
    
        Inputs: 
    
        altaz       [numpy array] Alt-az sky positions (in degrees) of sources 
                    It should be a 2-column numpy array. Each 2-column entity 
                    corresponds to a source position. Number of 2-column 
                    entities should equal the number of sources as specified 
                    by the size of flux_ref. It is of size nsrc x 2
    
        ctlgind     [numpy array] Indices of sources in the attribute skymodel 
                    that will be used in generating the E-field spectrum. If
                    specified as None (default), all objects in the attribute
                    skymodel will be used. It size must be of size nsrc as 
                    described in input altaz
    
        uvlocs      [numpy array] Locations in the UV-plane at which electric
                    fields are to be computed. It must be of size nuv x 2. If
                    set to None (default), it will be automatically determined
                    from the antenna aperture attribute

        pol         [list] List of polarizations to process. The polarizations
                    are specified as strings 'P1' and 'P2. If set to None
                    (default), both polarizations are processed
    
        randomseed  [integer] Seed to initialize the randon generator. If set
                    to None (default), the random sequences generated are not
                    reproducible. Set to an integer to generate reproducible
                    random sequences. Will be used only if the other input 
                    randvals is set to None
    
        randvals    [numpy array] Externally generated complex random numbers.
                    Both real and imaginary parts must be drawn from a normal
                    distribution (mean=0, var=1). Always must have size equal 
                    to nsrc x nchan x npol. If specified as a vector, it must be 
                    of size nsrc x nchan x npol. Either way it will be resphaed 
                    to size nsrc x nchan x npol. If randvals is specified, no 
                    fresh random numbers will be generated and the input 
                    randomseed will be ignored.
    
        Output:
    
        sky_Ef_info [dictionary] Consists of E-field info under two keys 'P1' 
                    and 'P2', one for each polarization. Under each of these 
                    keys the complex electric fields spectra of shape 
                    nuv x nchan are stored. nchan is the number of channels in 
                    the spectrum and nuv is the number of gridded points in the 
                    aperture footprint
        ------------------------------------------------------------------------
        """
        
        try:
            altaz
        except NameError:
            raise NameError('Input altaz must be specified')

        srcdircos = GEOM.altaz2dircos(altaz, units='degrees')

        if ctlgind is None:
            ctlgind = NP.arange(self.skymodel.location.shape[0])
        elif isinstance(ctlgind, list):
            ctlgind = NP.asarray(ctlgind)
        elif isinstance(ctlgind, NP.ndarray):
            ctlgind = ctlgind.ravel()
        else:
            raise TypeError('Input ctlgind must be a list, numpy array or set to None')
        if ctlgind.size != altaz.shape[0]:
            raise ValueError('Input ctlgind must contain same number of elements as number of objects in input altaz.')
        skymodel = self.skymodel.subset(ctlgind, axis='position')
        nsrc = ctlgind.size
        nchan = self.f.size
        spectrum = skymodel.generate_spectrum(frequency=self.f)

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
        npol = len(pol)

        if randomseed is None:
            randomseed = NP.random.randint(1000000)
        elif not isinstance(randomseed, int):
            raise TypeError('If input randomseed is not None, it must be an integer')

        if uvlocs is None:
            typetags = self.antenna_array.typetags
            antwts = {}
            antlabels = []
            aprtrs = []
            max_aprtr_size = []
            for typetag in typetags:
                antlabel = list(self.antenna_array.typetags[typetag])[0]
                antlabels += [antlabel]
                aprtr = self.antenna_array.antennas[antlabel].aperture
                max_aprtr_size += [max([NP.sqrt(aprtr.xmax['P1']**2 + NP.sqrt(aprtr.ymax['P1']**2)), NP.sqrt(aprtr.xmax['P2']**2 + NP.sqrt(aprtr.ymax['P2']**2)), aprtr.rmax['P1'], aprtr.rmax['P2']])]
                
            max_aprtr_halfwidth = NP.amax(NP.asarray(max_aprtr_size))
            wl = FCNST.c / self.f
            trc = max_aprtr_halfwidth / wl.min()
            blc = -trc
            uvspacing = 0.5
            gridu, gridv = GRD.grid_2d([(blc, trc), (blc, trc)], pad=0.0, spacing=uvspacing, pow2=True)
            uvlocs = NP.hstack((gridu.reshape(-1,1), gridv.reshape(-1,1)))
        else:
            if not isinstance(uvlocs, NP.ndarray):
                raise TypeError('Input uvlocs is numpy array')
            if uvlocs.ndim != 2:
                raise ValueError('Input uvlocs must be a 2D numpy array')
            if uvlocs.shape[1] != 2:
                raise ValueError('Input uvlocs must be a 2-column array')

        if randvals is not None:
            if not isinstance(randvals, NP.ndarray):
                raise TypeError('Input randvals must be a numpy array')
            if randvals.size != nsrc * nchan * npol:
                raise ValueError('Input randvals found to be of invalid size')
            randvals = randvals.reshape(nsrc,nchan,npol)

        sigmas = NP.sqrt(spectrum) # nsrc x nchan
        sky_Ef_info = {}
        if randvals is None:
            randstate = NP.random.RandomState(randomseed)
            randvals = randstate.normal(loc=0.0, scale=1.0, size=(nsrc,nchan,npol)) + 1j * randstate.normal(loc=0.0, scale=1.0, size=(nsrc,nchan,npol)) # nsrc x nchan x npol
        for polind, p in enumerate(pol):
            Ef_amp = sigmas/NP.sqrt(2) * randvals[:,:,polind] # nsrc x nchan
            Ef_phase = 1.0
            Ef = Ef_amp * Ef_phase # nsrc x nchan
            
            srcdircos_2d = srcdircos[:,:2] # nsrc x 2
            u_dot_l = NP.dot(uvlocs, srcdircos_2d.T) # nuv x nsrc
            matDFT = NP.exp(1j * 2 * NP.pi * u_dot_l) # nuv x nsrc
    
            sky_Ef_info[p] = NP.dot(matDFT, Ef) # nuv x nchan

        return sky_Ef_info

    ############################################################################
    
    def applyApertureWts(self, sky_Ef_info, uvlocs=None, pol=None):

        """
        ------------------------------------------------------------------------
        Apply aperture weights and estimate measurements of antenna electric 
        fields (assuming they are centered at origin). Aperture illumination
        weights are estimated and applied only for the unique antenna typetags.

        Inputs:

        sky_Ef_info [dictionary] Consists of E-field info under two keys 'P1' and
                    'P2', one for each polarization. Under each of these keys
                    the complex electric fields spectra of shape nuv x nchan are 
                    stored. nchan is the number of channels in the spectrum and 
                    nuv is the number of gridded points in the aperture footprint

        uvlocs      [numpy array] Locations in the UV-plane at which electric
                    fields are to be computed. It must be of size nuv x 2. If
                    set to None (default), it will be automatically determined
                    from the antenna aperture attribute

        Outputs:

        ant_Ef_info [dictionary] Contains antenna electric fields obtained by 
                    summing the electric fields on the grid locations that
                    come under the aperture illumination footprint of that
                    antenna. It consists of keys which are unique antenna 
                    typetags. Under each of these keys is another dictionary 
                    with two keys 'P1' and 'P2' for the two polarizations. The
                    value under each of these keys is a numpy array of size 
                    nchan where nchan is the number of frequency channels
        ------------------------------------------------------------------------
        """

        try:
            sky_Ef_info
        except NameError:
            raise NameError('Inputs sky_Ef_info must be specified')

        if not isinstance(sky_Ef_info, dict):
            raise TypeError('Input sky_Ef_info must be a dictionary')

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
        npol = len(pol)

        if len(set(pol).intersection(sky_Ef_info.keys())) == 0:
            raise KeyError('Input sky_Ef_info does not contain any of the accepted polarizations')
        
        typetags = self.antenna_array.typetags.keys()
        antlabels = []
        aprtrs = []
        max_aprtr_size = []
        for typetag in typetags:
            antlabel = list(self.antenna_array.typetags[typetag])[0]
            antlabels += [antlabel]
            aprtr = self.antenna_array.antennas[antlabel].aperture
            aprtrs += [aprtr]
            max_aprtr_size += [max([NP.sqrt(aprtr.xmax['P1']**2 + NP.sqrt(aprtr.ymax['P1']**2)), NP.sqrt(aprtr.xmax['P2']**2 + NP.sqrt(aprtr.ymax['P2']**2)), aprtr.rmax['P1'], aprtr.rmax['P2']])]
        max_aprtr_halfwidth = NP.amax(NP.asarray(max_aprtr_size))
        
        if uvlocs is None:
            wl = FCNST.c / self.f
            trc = max_aprtr_halfwidth / wl.min()
            blc = -trc
            uvspacing = 0.5
            gridu, gridv = GRD.grid_2d([(blc, trc), (blc, trc)], pad=0.0, spacing=uvspacing, pow2=True)
            uvlocs = NP.hstack((gridu.reshape(-1,1), gridv.reshape(-1,1)))
        else:
            if not isinstance(uvlocs, NP.ndarray):
                raise TypeError('Input uvlocs is numpy array')
            if uvlocs.ndim != 2:
                raise ValueError('Input uvlocs must be a 2D numpy array')
            if uvlocs.shape[1] != 2:
                raise ValueError('Input uvlocs must be a 2-column array')
        
        for p in pol:
            if p in sky_Ef_info:
                if not isinstance(sky_Ef_info[p], NP.ndarray):
                    raise TypeError('Input sky_Ef_info under polarization key {0} must be a numpy array'.format(p))
                if sky_Ef_info[p].shape != (uvlocs.shape[0], self.f.size):
                    raise ValueError('Input sky_Ef_info under polarization key {0} has incompatible dimensions'.format(p))

        wl = FCNST.c / self.f
        wavelength = NP.zeros(uvlocs.shape[0]).reshape(-1,1) + wl.reshape(1,-1)
        xlocs = uvlocs[:,0].reshape(-1,1) * wl.reshape(1,-1)
        ylocs = uvlocs[:,1].reshape(-1,1) * wl.reshape(1,-1)
        xylocs = NP.hstack((xlocs.reshape(-1,1), ylocs.reshape(-1,1)))

        du = NP.diff(uvlocs[:,0]).max()
        dv = NP.diff(uvlocs[:,1]).max()
        rmaxNN = 0.5 * NP.sqrt(du**2 + dv**2) * wl.min()
        distNN = 2.0 * max_aprtr_halfwidth
        indNN_list, blind, vuf_gridind = LKP.find_NN(NP.zeros(2).reshape(1,-1), xylocs, distance_ULIM=distNN, flatten=True, parallel=False)
        dxy = xylocs[vuf_gridind,:]
        unraveled_vuf_ind = NP.unravel_index(vuf_gridind, (uvlocs.shape[0],self.f.size,))
        ant_Ef_info = {}
        for aprtrind, aprtr in enumerate(aprtrs):
            typetag = typetags[aprtrind]
            ant_Ef_info[typetag] = {}
            for p in pol:
                krn = aprtr.compute(dxy, wavelength=wavelength.ravel()[vuf_gridind], pol=p, rmaxNN=rmaxNN, load_lookup=False)
                krn_sparse = SpM.csr_matrix((krn[p], unraveled_vuf_ind), shape=(uvlocs.shape[0], self.f.size), dtype=NP.complex64)
                krn_sparse_sumuv = krn_sparse.sum(axis=0)
                krn_sparse_norm = krn_sparse.A / krn_sparse_sumuv.A
                spval = krn_sparse_norm[unraveled_vuf_ind]
                antwts = SpM.csr_matrix((spval, unraveled_vuf_ind), shape=(uvlocs.shape[0],self.f.size), dtype=NP.complex64)
                weighted_Ef = antwts.A * sky_Ef_info[p]
                ant_Ef_info[typetag][p] = NP.sum(weighted_Ef, axis=0)
                
        return ant_Ef_info

    ############################################################################
    
    def stack_E_spectrum(self, Ef_info=None):

        """
        ------------------------------------------------------------------------
        Stack E-field spectra along time-axis

        Inputs:

        Ef_info         [dictionary] Consists of E-field info under two keys 
                        'P1' and 'P2', one for each polarization. Under each of 
                        these keys is a nchan x nant complex numpy array 
                        consisting of complex stochastic electric field
                        spectra. nchan is the number of channels in the 
                        spectrum and nant is the number of antennas. If set to
                        None (default), the existing and the most recent 
                        attribute Ef_info will be used in its place and will 
                        get stacked
        ------------------------------------------------------------------------
        """

        if Ef_info is None:
            Ef_info = self.Ef_info

        if not isinstance(Ef_info, dict):
            raise TypeError('Input Ef_info must be a dictionary')

        if Ef_info:
            for pol in ['P1', 'P2']:
                if pol in Ef_info:
                    if Ef_info[pol].shape[0] != self.f.size:
                        raise ValueError('Dimensions of input Ef_info incompatible with number of frequency channels')
                    if Ef_info[pol].shape[1] != len(self.antinfo['labels']):
                        raise ValueError('Dimensions of input Ef_info incompatible with number of antennas')
    
                if not self.Ef_stack:
                    self.Ef_stack[pol] = NP.empty((self.f.size,len(self.antinfo['labels'])), dtype=NP.complex)
                    self.Ef_stack[pol].fill(NP.nan)
                    if pol in Ef_info:
                        self.Ef_stack[pol] = Ef_info[pol]
                    self.Ef_stack[pol] = self.Ef_stack[pol][:,:,NP.newaxis]
                else:
                    if pol not in self.Ef_stack:
                        self.Ef_stack[pol] = NP.empty((self.f.size,len(self.antinfo['labels'])), dtype=NP.complex)
                        self.Ef_stack[pol].fill(NP.nan)
                        if pol in Ef_info:
                            self.Ef_stack[pol] = Ef_info[pol]
                        self.Ef_stack[pol] = self.Ef_stack[pol][:,:,NP.newaxis]
                    else:
                        if pol in Ef_info:
                            self.Ef_stack[pol] = NP.dstack((self.Ef_stack[pol], Ef_info[pol][:,:,NP.newaxis]))
                        else:
                            nanvalue = NP.empty((self.f.size,len(self.antinfo['labels'])), dtype=NP.complex)
                            nanvalue.fill(NP.nan)
                            self.Ef_stack[pol] = NP.dstack((self.Ef_stack[pol], nanvalue[:,:,NP.newaxis]))

    ############################################################################
    
    def generate_E_timeseries(self, operand='recent'):

        """
        ------------------------------------------------------------------------
        Generate E-field timeseries from their spectra. It can be done on 
        current or stacked spectra

        Inputs:

        operand         [string] Parameter to decide if the timeseries is to
                        be produced from current E-field spectrum or from the
                        stacked spectra. If set to 'recent' (default), the most
                        recent spectra will be used. If set to 'stack' then the
                        stacked spectra will be used to create the timeseries
        ------------------------------------------------------------------------
        """

        if not isinstance(operand, str):
            raise TypeError('Input keyword operand must be a string')

        if operand not in ['recent', 'stack']:
            raise ValueError('Input keyword operand must be set to "recent" or "stack"')

        for pol in ['P1', 'P2']:
            if operand == 'recent':
                if self.Ef_info:
                    if pol in self.Ef_info:
                        Ef_shifted = NP.fft.ifftshift(self.Ef_info[pol], axes=0)
                        self.Et_info[pol] = NP.fft.ifft(Ef_shifted, axis=0)
            else:
                if self.Ef_stack:
                    if pol in self.Ef_stack:
                        Ef_shifted = NP.fft.ifftshift(self.Ef_stack[pol], axes=0)
                        self.Et_stack[pol] = NP.fft.ifft(Ef_shifted, axis=0)
                
    ############################################################################
    
    def observe(self, lst, phase_center_coords, pointing_center_coords,
                obs_date=None, phase_center=None, pointing_center=None,
                pointing_info=None, vbeam_files=None, obsmode=None, 
                randomseed=None, stack=False, short_dipole_approx=False,
                half_wave_dipole_approx=False, parallel_genvb=False,
                parallel_genEf=False, nproc=None):

        """
        ------------------------------------------------------------------------
        Simulate a single observation and record antenna electric fields as a
        function of polarization, frequencies and antennas.

        Inputs:

        lst        [scalar] Local Sidereal Time (in hours) in the range 0--24
                   on the date specified by obs_date.

        phase_center_coords
                   [string] Coordinate system describing the phase center. 
                   Accepted values are 'altaz', 'radec', 'hadec' and 'dircos'
                   for Alt-Az, RA-dec, HA-dec and direction cosines 
                   respectively. If set to 'altaz', 'radec' or 'hadec', the
                   coordinates must be specified in degrees. 

        pointing_center_coords
                   [string] Coordinate system describing the pointing center. 
                   Accepted values are 'altaz', 'radec', 'hadec' and 'dircos'
                   for Alt-Az, RA-dec, HA-dec and direction cosines 
                   respectively. If set to 'altaz', 'radec' or 'hadec', the
                   coordinates must be specified in degrees. 

        Keyword Inputs:

        obs_date   [string] Date of observation in YYYY/MM/DD format. If set to
                   None (default), the epoch in the sky model will be assumed
                   to be the date of observation. 

        phase_center
                   [numpy array] Phase center of the observation in the 
                   coordinate system specified by phase_center_coords. If 
                   phase_center_coords is set to 'altaz', 'radec' or 'hadec'
                   the phase center must be a 2-element numpy array with values
                   in degrees. If phase_center_coords is set to 'dircos' it 
                   must be a 3-element direction cosine vector

        pointing_center
                   [numpy array] Pointing center of the observation in the
                   coordinate system specified by pointing_center_coords. If 
                   pointing_center_coords is set to 'altaz', 'radec' or 'hadec'
                   the pointing center must be a 2-element numpy array with 
                   values in degrees. If pointing_center_coords is set to 
                   'dircos' it must be a 3-element direction cosine vector

        pointing_info 
                   [dictionary] A dictionary consisting of information 
                   relating to pointing center in case of a phased array. 
                   The pointing center can be specified either via element 
                   delay compensation or by directly specifying the pointing 
                   center in a certain coordinate system. Default = None 
                   (pointing centered at zenith). This dictionary consists of 
                   the following tags and values:
                   'gains'           [numpy array] Complex element gains. 
                                     Must be of size equal to the number of 
                                     elements as specified by the number of 
                                     rows in antpos. If set to None (default), 
                                     all element gains are assumed to be unity. 
                                     Used only in phased array mode.
                   'gainerr'         [int, float] RMS error in voltage 
                                     amplitude in dB to be used in the 
                                     beamformer. Random jitters are drawn from 
                                     a normal distribution in logarithm units 
                                     which are then converted to linear units. 
                                     Must be a non-negative scalar. If not 
                                     provided, it defaults to 0 (no jitter).
                                     Used only in phased array mode.
                   'delays'          [numpy array] Delays (in seconds) to be 
                                     applied to the tile elements. Size should 
                                     be equal to number of tile elements 
                                     (number of rows in antpos). Default=None 
                                     will set all element delays to zero 
                                     phasing them to zenith. Used only in 
                                     phased array mode. 
                   'pointing_center' [numpy array] This will apply in the 
                                     absence of key 'delays'. This can be 
                                     specified as a row vector. Should have 
                                     two-columns if using Alt-Az coordinates, 
                                     or two or three columns if using direction 
                                     cosines. There is no default. The
                                     coordinate system must be specified in
                                     'pointing_coords' if 'pointing_center' is 
                                     to be used.
                   'pointing_coords' [string scalar] Coordinate system in which 
                                     the pointing_center is specified. Accepted 
                                     values are 'altaz' or 'dircos'. Must be 
                                     provided if 'pointing_center' is to be 
                                     used. No default.
                   'delayerr'        [int, float] RMS jitter in delays used in 
                                     the beamformer. Random jitters are drawn 
                                     from a normal distribution with this rms. 
                                     Must be a non-negative scalar. If not 
                                     provided, it defaults to 0 (no jitter). 
                                     Used only in phased array mode.
        
        vbeam_files 
                   [dictionary] Dictionary containing file locations of 
                   far-field voltage patterns. It is specified under keys
                   'P1' and 'P2' denoting the two polarizations. Under each
                   polarization key is another dictionary with keys for 
                   individual antennas denoted by antenna labels (string). 
                   If there is only one antenna key it will be assumed to be 
                   identical for all antennas. If multiple voltage beam file 
                   locations are specified, it must be the same as number of 
                   antennas 

        obsmode    [string] Specifies observing mode. Accepted values are
                   'drift', 'track' or None (default)

        randomseed
                   [integer] Seed to initialize the randon generator. If set
                   to None (default), the random sequences generated are not
                   reproducible. Set to an integer to generate reproducible
                   random sequences

        stack      [boolean] If set to True, stack the generated E-field
                   spectrum to the attribute Ef_stack. If set to False 
                   (default), no such action is performed.

        short_dipol_approx
                   [boolean] if True, indicates short dipole approximation
                   is to be used. Otherwise, a more accurate expression is 
                   used for the dipole pattern. Default=False. Both
                   short_dipole_approx and half_wave_dipole_approx cannot be 
                   set to True at the same time
        
        half_wave_dpole_approx
                   [boolean] if True, indicates half-wave dipole approximation
                   is to be used. Otherwise, a more accurate expression is 
                   used for the dipole pattern. Default=False

        parallel_genvb
                   [boolean] specifies if parallelization is to be invoked in
                   generating voltage beams. If False (default) means only 
                   serial processing. Highly recommended to set to False as 
                   overheads in parallelization slow it down.

        parallel_genEf  
                   [boolean] specifies if parallelization is to be invoked in
                   generating E-field spectra. If False (default) means only 
                   serial processing. Highly recommended to set to False as 
                   overheads in parallelization slow it down.

        nproc      [integer] specifies number of independent processes to 
                   spawn. Default = None, means automatically determines the 
                   number of process cores in the system and use one less 
                   than that to avoid locking the system for other processes. 
                   Applies only if input parameter 'parallel' (see above) is 
                   set to True. If nproc is set to a value more than the 
                   number of process cores in the system, it will be reset to 
                   number of process cores in the system minus one to avoid 
                   locking the system out for other processes

        ------------------------------------------------------------------------
        """

        try:
            lst, phase_center_coords, pointing_center_coords
        except NameError:
            raise NameError('Input LST must be specified')

        if not isinstance(lst, (int,float)):
            raise TypeError('Input LST must be a scalar')
        lst = float(lst)

        if phase_center_coords not in ['hadec', 'radec', 'altaz', 'dircos']:
            raise ValueError('Input phase_center_coords must be set tp "radec", "hadec", "altaz" or "dircos"')

        if pointing_center_coords not in ['hadec', 'radec', 'altaz', 'dircos']:
            raise ValueError('Input pointing_center_coords must be set tp "radec", "hadec", "altaz" or "dircos"')
        
        if obs_date is None:
            obs_date = self.observer.date

        lstobj = EP.FixedBody()
        lstobj._epoch = obs_date
        lstobj._ra = NP.radians(lst * 15.0)
        lstobj._dec = NP.radians(self.latitude)
        lstobj.compute(self.observer)
        lst_temp = NP.degrees(lstobj.ra) # in degrees
        dec_temp = NP.degrees(lstobj.dec) # in degrees

        obsrvr = EP.Observer()
        obsrvr.lat = self.observer.lat
        obsrvr.lon = self.observer.lon
        obsrvr.date = obs_date
        self.timestamp = obsrvr.next_transit(lstobj)

        if phase_center is None:
            phase_center_dircos = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
        else:
            if phase_center_coords == 'dircos':
                phase_center_dircos = phase_center
            elif phase_center_coords == 'altaz':
                phase_center_dircos = GEOM.altaz2dircos(phase_center, units='degrees')
            elif phase_center_coords == 'hadec':
                phase_center_altaz = GEOM.hadec2altaz(phase_center, self.latitude, units='degrees')
                phase_center_dircos = GEOM.altaz2dircos(phase_center_altaz, units='degrees')
            elif phase_center_coords == 'radec':
                phase_center_hadec = NP.asarray([lst_temp - phase_center[0,0], phase_center[0,1]]).reshape(1,-1)
                phase_center_altaz = GEOM.hadec2altaz(phase_center_hadec, self.latitude, units='degrees')
                phase_center_dircos = GEOM.altaz2dircos(phase_center_altaz, units='degrees')
            else:
                raise ValueError('Invalid value specified in phase_center_coords')

        if pointing_center is None:
            pointing_center_altaz = NP.asarray([90.0, 270.0]).reshape(1,-1)
        else:
            if pointing_center_coords == 'altaz':
                pointing_center_altaz = pointing_center
            elif pointing_center_coords == 'dircos':
                pointing_center_altaz = GEOM.dircos2altaz(pointing_center, units='degrees')
            elif pointing_center_coords == 'hadec':
                pointing_center_altaz = GEOM.hadec2altaz(pointing_center, self.latitude, units='degrees')
            elif pointing_center_coords == 'radec':
                pointing_center_hadec = NP.asarray([lst_temp - pointing_center[0,0], pointing_center[0,1]]).reshape(1,-1)
                pointing_center_altaz = GEOM.hadec2altaz(pointing_center_hadec, self.latitude, units='degrees')
            else:
                raise ValueError('Invalid value specified in pointing_center_coords')
            
        hemind, altaz = self.upper_hemisphere(lst, obs_date=obs_date)
        if hemind.size == 0:
            self.Ef_info = {}
            for apol in ['P1', 'P2']:
                self.Ef_info[apol] = NP.zeros((self.f.size, len(self.antenna_array.antennas)), dtype=NP.complex)
        else:
            if vbeam_files is not None:
                vbeams = self.load_voltage_patterns(vbeam_files, altaz, parallel=parallel, nproc=nproc)
            else:
                vbeams = self.generate_voltage_pattern(altaz, pointing_center=pointing_center_altaz, pointing_info=pointing_info, short_dipole_approx=short_dipole_approx, half_wave_dipole_approx=half_wave_dipole_approx, parallel=parallel_genvb, nproc=nproc)
            self.generate_E_spectrum(altaz, vbeams, ctlgind=hemind, pol=['P1','P2'], ref_point=phase_center_dircos, randomseed=randomseed, parallel=parallel_genEf, nproc=nproc, action='store')
            # sky_Ef_info = self.generate_sky_E_spectrum(altaz, ctlgind=hemind, uvlocs=None, pol=None, randomseed=randomseed, randvals=None)
            # ant_Ef_info = self.applyApertureWts(sky_Ef_info, uvlocs=None, pol=None)

        if obsmode is not None:
            if obsmode in ['drift', 'track']:
                self.obsmode = obsmode
            else:
                raise ValueError('Invalid value specified for input obsmode')

        if stack:
            self.stack_E_spectrum()
            self.timestamps += [self.timestamp]

    ############################################################################

    def observing_run(self, init_parms, obsmode='track', duration=None,
                      pointing_info=None, vbeam_files=None, randomseed=None,
                      short_dipole_approx=False, half_wave_dipole_approx=False,
                      parallel_genvb=False, parallel_genEf=False, nproc=None):

        """
        ------------------------------------------------------------------------
        Simulate a observing run made of multiple contiguous observations and 
        record antenna electric fields as a function of polarization, 
        frequencies, antennas, and time.

        Inputs:

        init_parms [dictionary] Contains the parameters to initialize an
                   observing run. It consists of the following keys and values:
                   'obs_date'   [string] Date string in 'YYYY/MM/DD HH:MM:SS.SS' 
                                format. If not provided, will default to using 
                                the epoch of the sky model attribute. If key
                                'sidereal_time' is absent, this parameter will 
                                be used as the solar time and a sidereal time
                                will be estimated for the instant specified in
                                this parameter
                   'sidereal_time' 
                                [float] Local sidereal time (in hours) on the 
                                date of observation specified in the YYYY/MM/DD 
                                part of the value in key 'obs_date'. If not 
                                specified, a sidereal time will be estimated 
                                from the 'obs_date' parameter
                   'phase_center_coords'
                                [string] Coordinate system describing the phase 
                                center. Accepted values are 'altaz', 'radec', 
                                'hadec' and 'dircos' for Alt-Az, RA-dec, HA-dec 
                                and direction cosines respectively. If set to 
                                'altaz', 'radec' or 'hadec', the coordinates 
                                must be specified in degrees. 
                   'pointing_center_coords'
                                [string] Coordinate system describing the 
                                pointing center. Accepted values are 'altaz', 
                                'radec', 'hadec' and 'dircos' for Alt-Az, 
                                RA-dec, HA-dec and direction cosines 
                                respectively. If set to 'altaz', 'radec' or 
                                'hadec', the coordinates must be specified in 
                                degrees. 
                   'phase_center'
                                [numpy array] Phase center of the observation 
                                in the coordinate system specified by 
                                phase_center_coords. If phase_center_coords is 
                                set to 'altaz', 'radec' or 'hadec' the phase 
                                center must be a 2-element numpy array with 
                                values in degrees. If phase_center_coords is 
                                set to 'dircos' it must be a 3-element 
                                direction cosine vector
                   'pointing_center'
                                [numpy array] Pointing center of the 
                                observation in the coordinate system specified 
                                by pointing_center_coords. If 
                                pointing_center_coords is set to 'altaz', 
                                'radec' or 'hadec' the pointing center must be 
                                a 2-element numpy array with values in degrees. 
                                If pointing_center_coords is set to 'dircos' it 
                                must be a 3-element direction cosine vector

        Keyword Inputs:

        obsmode    [string] Mode of observation. Accepted values are 'drift' 
                   and 'track' (default)

        duration   [float] Total duration of the observing run (in seconds). If
                   set to None (default), one timeseries is generated

        pointing_info 
                   [dictionary] A dictionary consisting of information 
                   relating to pointing center in case of a phased array. 
                   The pointing center can be specified either via element 
                   delay compensation or by directly specifying the pointing 
                   center in a certain coordinate system. Default = None 
                   (pointing centered at zenith). This dictionary consists of 
                   the following tags and values:
                   'gains'           [numpy array] Complex element gains. 
                                     Must be of size equal to the number of 
                                     elements as specified by the number of 
                                     rows in antpos. If set to None (default), 
                                     all element gains are assumed to be unity. 
                                     Used only in phased array mode.
                   'gainerr'         [int, float] RMS error in voltage 
                                     amplitude in dB to be used in the 
                                     beamformer. Random jitters are drawn from 
                                     a normal distribution in logarithm units 
                                     which are then converted to linear units. 
                                     Must be a non-negative scalar. If not 
                                     provided, it defaults to 0 (no jitter).
                                     Used only in phased array mode.
                   'delays'          [numpy array] Delays (in seconds) to be 
                                     applied to the tile elements. Size should 
                                     be equal to number of tile elements 
                                     (number of rows in antpos). Default=None 
                                     will set all element delays to zero 
                                     phasing them to zenith. Used only in 
                                     phased array mode. 
                   'pointing_center' [numpy array] This will apply in the 
                                     absence of key 'delays'. This can be 
                                     specified as a row vector. Should have 
                                     two-columns if using Alt-Az coordinates, 
                                     or two or three columns if using direction 
                                     cosines. There is no default. The
                                     coordinate system must be specified in
                                     'pointing_coords' if 'pointing_center' is 
                                     to be used.
                   'pointing_coords' [string scalar] Coordinate system in which 
                                     the pointing_center is specified. Accepted 
                                     values are 'altaz' or 'dircos'. Must be 
                                     provided if 'pointing_center' is to be 
                                     used. No default.
                   'delayerr'        [int, float] RMS jitter in delays used in 
                                     the beamformer. Random jitters are drawn 
                                     from a normal distribution with this rms. 
                                     Must be a non-negative scalar. If not 
                                     provided, it defaults to 0 (no jitter). 
                                     Used only in phased array mode.
        
        vbeam_files 
                   [dictionary] Dictionary containing file locations of 
                   far-field voltage patterns. It is specified under keys
                   'P1' and 'P2' denoting the two polarizations. Under each
                   polarization key is another dictionary with keys for 
                   individual antennas denoted by antenna labels (string). 
                   If there is only one antenna key it will be assumed to be 
                   identical for all antennas. If multiple voltage beam file 
                   locations are specified, it must be the same as number of 
                   antennas 

        randomseed
                   [integer] Seed to initialize the randon generator. If set
                   to None (default), the random sequences generated are not
                   reproducible. Set to an integer to generate reproducible
                   random sequences

        short_dipole_approx
                   [boolean] if True, indicates short dipole approximation
                   is to be used. Otherwise, a more accurate expression is 
                   used for the dipole pattern. Default=False. Both
                   short_dipole_approx and half_wave_dipole_approx cannot be 
                   set to True at the same time
        
        half_wave_dipole_approx
                   [boolean] if True, indicates half-wave dipole approximation
                   is to be used. Otherwise, a more accurate expression is 
                   used for the dipole pattern. Default=False

        parallel_genvb
                   [boolean] specifies if parallelization is to be invoked in
                   generating voltage beams. If False (default) means only 
                   serial processing. Highly recommended to set to False as 
                   overheads in parallelization slow it down.

        parallel_genEf  
                   [boolean] specifies if parallelization is to be invoked in
                   generating E-field spectra. If False (default) means only 
                   serial processing. Highly recommended to set to False as 
                   overheads in parallelization slow it down.

        nproc      [integer] specifies number of independent processes to 
                   spawn. Default = None, means automatically determines the 
                   number of process cores in the system and use one less 
                   than that to avoid locking the system for other processes. 
                   Applies only if input parameter 'parallel' (see above) is 
                   set to True. If nproc is set to a value more than the 
                   number of process cores in the system, it will be reset to 
                   number of process cores in the system minus one to avoid 
                   locking the system out for other processes
        ------------------------------------------------------------------------
        """

        try:
            init_parms
        except NameError:
            raise NameError('Input init_parms must be specified')

        if not isinstance(init_parms, dict):
            raise TypeError('Input init_parms must be a dictionary')

        if not isinstance(obsmode, str):
            raise TypeError('Input osbmode must be a string')

        if obsmode not in ['track', 'drift']:
            raise ValueError('Input obsmode must be set to "track" or "drift"')
        self.obsmode = obsmode

        if 'obs_date' not in init_parms:
            init_parms['obs_date'] = self.skymodel.epoch.strip('J')

        if 'phase_center' not in init_parms:
            init_parms['phase_center'] = NP.asarray([90.0, 270.0]).reshape(1,-1)
            init_parms['phase_center_coords'] = 'altaz'
        else:
            init_parms['phase_center'] = NP.asarray(init_parms['phase_center']).reshape(1,-1)

        if 'pointing_center' not in init_parms:
            init_parms['pointing_center'] = NP.asarray([90.0, 270.0]).reshape(1,-1)
            init_parms['pointing_center_coords'] = 'altaz'
        else:
            init_parms['pointing_center'] = NP.asarray(init_parms['pointing_center']).reshape(1,-1)

        if duration is None:
            duration = self.t.max()

        duration = float(duration)
        if duration <= 0.0:
            raise ValueError('Observation duration must be positive')
        n_nyqseries = NP.round(duration/self.t.max()).astype(int)
        if n_nyqseries < 1:
            raise ValueError('Observation duration is too short to make a single Nyquist observation sample')

        if 'sidereal_time' in init_parms:
            if not isinstance(init_parms['sidereal_time'], (int,float)):
                raise TypeError('sidereal time must be a scalar')
            init_parms['sidereal_time'] = float(init_parms['sidereal_time'])
            if (init_parms['sidereal_time'] >= 0.0) and (init_parms['sidereal_time'] < 24.0):
                sdrltime = init_parms['sidereal_time']
            else:
                raise ValueError('sidereal time must be in the range 0--24 hours')
        else:
            if not isinstance(init_parms['obs_date'], str):
                raise TypeError('obs_date value must be a date string in YYYY/MM/DD HH:MM:SS.SSS format')
            slrtime = init_parms['obs_date']

        obsrvr = EP.Observer()
        obsrvr.lat = NP.radians(self.latitude)
        obsrvr.lon = NP.radians(self.longitude)
        obsrvr.date = init_parms['obs_date']

        lstobj = EP.FixedBody()
        lstobj._epoch = init_parms['obs_date']
        lstobj._epoch = EP.Date(NP.floor(lstobj._epoch - 0.5) + 0.5) # Round it down to beginning of the day
        if 'sidereal_time' not in init_parms:
            obsrvr.date = slrtime
            sdrltime = NP.degrees(obsrvr.sidereal_time()) / 15.0
        lstobj._ra = NP.radians(sdrltime * 15.0)
        if 'sidereal_time' in init_parms:
            lstobj.compute(obsrvr)
            slrtime = lstobj.transit_time
            obsrvr.date = slrtime

        updated_sdrltime = copy.copy(sdrltime)
        updated_slrtime = copy.copy(slrtime)
        updated_obsdate = EP.Date(NP.floor(obsrvr.date - 0.5) + 0.5) # Round it down to beginning of the day
        if obsmode == 'track':
            if init_parms['phase_center_coords'] == 'dircos':
                phase_center_altaz = GEOM.dircos2altaz(init_parms['phase_center'], units='degrees')
                phase_center_hadec = GEOM.altaz2hadec(phase_center_altaz, self.latitude, units='degrees')
                phase_center_radec = NP.asarray([15.0*sdrltime - phase_center_hadec[0,0], phase_center_hadec[0,1]]).reshape(1,-1)
                phase_center_coords = 'radec'
            elif init_parms['phase_center_coords'] == 'altaz':
                phase_center_hadec = GEOM.altaz2hadec(init_parms['phase_center'], self.latitude, units='degrees')
                phase_center_radec = NP.asarray([15.0*sdrltime - phase_center_hadec[0,0], phase_center_hadec[0,1]]).reshape(1,-1)
                phase_center_coords = 'radec'
            elif init_parms['phase_center_coords'] == 'hadec':
                phase_center_radec = NP.asarray([15.0*sdrltime - init_parms['phase_center'][0,0], init_parms['phase_center'][0,1]]).reshape(1,-1)
                phase_center_coords = 'radec'
            else:
                phase_center_radec = init_parms['phase_center']
                phase_center_coords = 'radec'

            if init_parms['pointing_center_coords'] == 'dircos':
                pointing_center_altaz = GEOM.dircos2altaz(init_parms['pointing_center'], units='degrees')
                pointing_center_hadec = GEOM.altaz2hadec(pointing_center_altaz, self.latitude, units='degrees')
                pointing_center_radec = NP.asarray([15.0*sdrltime - pointing_center_hadec[0,0], pointing_center_hadec[0,1]]).reshape(1,-1)
                pointing_center_coords = 'radec'
            elif init_parms['pointing_center_coords'] == 'altaz':
                pointing_center_hadec = GEOM.altaz2hadec(init_parms['pointing_center'], self.latitude, units='degrees')
                pointing_center_radec = NP.asarray([15.0*sdrltime - pointing_center_hadec[0,0], pointing_center_hadec[0,1]]).reshape(1,-1)
                pointing_center_coords = 'radec'
            elif init_parms['pointing_center_coords'] == 'hadec':
                pointing_center_radec = NP.asarray([15.0*sdrltime - init_parms['pointing_center'][0,0], init_parms['pointing_center'][0,1]]).reshape(1,-1)
                pointing_center_coords = 'radec'
            else:
                pointing_center_radec = init_parms['pointing_center']
                pointing_center_coords = 'radec'
            phase_center = phase_center_radec
            pointing_center = pointing_center_radec
        else:
            if init_parms['phase_center_coords'] == 'radec':
                phase_center = NP.asarray([15.0*sdrltime - init_parms['phase_center'][0,0], init_parms['phase_center'][0,1]]).reshape(1,-1)
                phase_center_coords = 'hadec'
            else:
                phase_center = init_parms['phase_center']
                phase_center_coords = init_parms['phase_center_coords']

            if init_parms['pointing_center_coords'] == 'radec':
                pointing_center = NP.asarray([15.0*sdrltime - init_parms['pointing_center'][0,0], init_parms['pointing_center'][0,1]]).reshape(1,-1)
                pointing_center_coords = 'hadec'
            else:
                pointing_center = init_parms['pointing_center']
                pointing_center_coords = init_parms['pointing_center_coords']
                
        if randomseed is None:
            randomseed = NP.random.randint(1000000)
        elif not isinstance(randomseed, int):
            raise TypeError('If input randomseed is not None, it must be an integer')

        progressbar_loc = (0, WM.term.height)
        writer = WM.Writer(progressbar_loc)
        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Iterations '.format(n_nyqseries), PGB.ETA()], maxval=n_nyqseries, fd=writer).start()
        for i in range(n_nyqseries):
            self.observe(updated_sdrltime, phase_center_coords, pointing_center_coords, obs_date=updated_obsdate, phase_center=phase_center, pointing_center=pointing_center, pointing_info=pointing_info, vbeam_files=vbeam_files, randomseed=randomseed+i, stack=True, short_dipole_approx=short_dipole_approx, half_wave_dipole_approx=half_wave_dipole_approx, parallel_genvb=parallel_genvb, parallel_genEf=parallel_genEf, nproc=nproc)
            obsrvr.date = obsrvr.date + EP.second * self.t.max()
            updated_sdrltime = NP.degrees(obsrvr.sidereal_time()) / 15.0
            updated_slrtime = copy.copy(obsrvr.date)
            updated_obsdate = EP.Date(NP.floor(obsrvr.date - 0.5) + 0.5) # Round it down to beginning of the day

            progress.update(i+1)
        progress.finish()

    ############################################################################

    def save(self, filename, compress=True):
        
        """
        ------------------------------------------------------------------------
        Save information instance of class AntennaArraySimulator to external 
        file in HDF5 format

        Input:

        filename    [string] Full path to the external file where data in the
                    instance of class AntennaArraySimulator is to be saved. 
                    The filename extension should be avoided as it will be 
                    appended automatically

        Keyword Inputs:

        compress    [boolean] If set to True (default), will compress the data
                    arrays in GZIP format
        ------------------------------------------------------------------------
        """

        with h5py.File(filename+'.hdf5', 'w') as fileobj:
            obsparm_group = fileobj.create_group('obsparm')
            obsparm_group['f0'] = self.f0
            obsparm_group['f0'].attrs['units'] = 'Hz'
            obsparm_group['frequencies'] = self.f
            obsparm_group['frequencies'].attrs['units'] = 'Hz'
            obsparm_group['tsamples'] = self.t
            obsparm_group['tsamples'].attrs['units'] = 'seconds'
            obsparm_group['timestamps'] = self.timestamps
            obsparm_group['timestamps'].attrs['units'] = 'Dublin Julian Date'
            obsparm_group['timestamp'] = self.timestamp
            obsparm_group['timestamp'].attrs['units'] = 'Dublin Julian Date'
            obsparm_group['mode'] = self.obsmode

            observatory_group = fileobj.create_group('observatory')
            observatory_group['latitude'] = self.latitude
            observatory_group['latitude'].attrs['units'] = 'degrees'
            observatory_group['longitude'] = self.longitude
            observatory_group['longitude'].attrs['units'] = 'degrees'
            observatory_group['antennas'] = self.antinfo['labels']
            observatory_group['antennas'].attrs['identical'] = self.identical_antennas
            observatory_group['antenna_positions'] = self.antinfo['positions']
            observatory_group['antenna_positions'].attrs['units'] = 'metres'

            self.skymodel.save(filename+'.skymodel', fileformat='hdf5')
            skymodel_group = fileobj.create_group('skymodel')
            skymodel_group['filename'] = filename+'.skymodel.hdf5'

            spec_group = fileobj.create_group('spectrum')
            if self.Ef_info:
                for pol in ['P1', 'P2']:
                    if pol in self.Ef_info:
                        if compress:
                            dset = spec_group.create_dataset('current/'+pol, data=self.Ef_info[pol], compression="gzip", compression_opts=9)
                        else:
                            spec_group['current/'+pol] = self.Ef_info[pol]
                spec_group['current'].attrs['timestamp'] = self.timestamp
            if self.Ef_stack:
                for pol in ['P1', 'P2']:
                    if pol in self.Ef_stack:
                        if compress:
                            dset = spec_group.create_dataset('tstack/'+pol, data=self.Ef_stack[pol], compression="gzip", compression_opts=9)
                        else:
                            spec_group['tstack/'+pol] = self.Ef_stack[pol]

            time_group = fileobj.create_group('timeseries')
            if self.Et_info:
                for pol in ['P1', 'P2']:
                    if pol in self.Et_info:
                        if compress:
                            dset = time_group.create_dataset('current/'+pol, data=self.Et_info[pol], compression="gzip", compression_opts=9)
                        else:
                            time_group['current/'+pol] = self.Et_info[pol]
                time_group['current'].attrs['timestamp'] = self.timestamp
            if self.Et_stack:
                for pol in ['P1', 'P2']:
                    if pol in self.Et_stack:
                        if compress:
                            dset = time_group.create_dataset('tstack/'+pol, data=self.Et_stack[pol], compression="gzip", compression_opts=9)
                        else:
                            time_group['tstack/'+pol] = self.Et_stack[pol]
                        
    ############################################################################
