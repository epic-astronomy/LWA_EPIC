import numpy as NP
import scipy.constants as FCNST
import ephem as EP
import multiprocessing as MP
import itertools as IT
from astropy.io import fits, ascii
import my_DSP_modules as DSP
import my_operations as OPS
import geometry as GEOM
import catalog as SM
import antenna_array as AA

################### Routines essential for parallel processing ################

def interp_beam_arg_splitter(args, **kwargs):
    return interp_beam(*args, **kwargs)

def stochastic_E_timeseries_arg_splitter(args, **kwargs):
    return stochastic_E_timeseries(*args, **kwargs)

###############################################################################

def interp_beam(beamfile, altaz, freqs):

    """
    -----------------------------------------------------------------------------
    Read and interpolate antenna pattern to the specified frequencies and 
    angular locations.

    Inputs:

    beamfile    [string] Full path to file containing antenna pattern. Must be
                specified, no default.

    altaz       [numpy array] Altitude and Azimuth as a nsrc x 2 numpy array. It
                must be specified in degrees. If not specified, no interpolation
                is performed spatially.

    freqs       [numpy array] Frequencies (in Hz) at which the antenna pattern 
                is to be interpolated. If not specified, no spectral 
                interpolation is performed

    Outputs:

    Antenna pattern interpolated at locations and frequencies specified. It will
    be a numpy array of size nsrc x nchan
    -----------------------------------------------------------------------------
    """

    try:
        beamfile
    except NameError:
        raise NameError('Input beamfile must be specified')

    try:
        altaz
    except NameError:
        theta_phi = None

    if theta_phi is not None:
        if not isinstance(altaz, NP.ndarray):
            raise TypeError('Input altaz must be a numpy array')

        if altaz.ndim != 2:
            raise ValueError('Input altaz must be a nsrc x 2 numpy array')

        theta_phi = NP.hstack((90.0-altaz[:,0].reshape(-1,1), altaz[:,1].reshape(-1,1)))
        theta_phi = NP.radians(theta_phi)

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

    Ef = Ef[:,:,NP.newaxis]
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

    antinfo         [dictionary] contains the following keys and 
                    information:
                    'labels':    list of strings of antenna labels
                    'positions': position vectors of antennas (3-column 
                                 array) in local ENU coordinates

    Member function:

    __init__()      Initialize the AntennaArraySimulator class which manages 
                    information about the simulation of Electrc fields by 
                    the antennas

    upper_hemisphere()
                    Return the indices of locations in the catalog that are 
                    in the upper celestial hemisphere for a given LST on a 
                    given date of observation

    find_voltage_pattern()
                    Generates (by interpolating if necessary) voltage 
                    pattern at the location of catalog sources based on 
                    external voltage pattern files specified. Parallel 
                    processing can be performed.
    ------------------------------------------------------------------------
    """

    def __init__(self, antenna_array, skymodel, identical_antennas=False):

        """
        ------------------------------------------------------------------------
        Initialize the AntennaArraySimulator class which manages information 
        about the simulation of Electrc fields by the antennas

        Class attributes initialized are:
        antenna_array, skymodel, latitude, f, f0, antinfo

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

        self.latitude = self.antenna_array.latitude
        self.longitude = self.antenna_array.longitude
        self.f = self.antenna_array.f
        self.f0 = self.antenna_array.f0
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
    
    def find_voltage_pattern(self, vbeam_files, parallel=False, nproc=None):

        """
        ------------------------------------------------------------------------
        Generates (by interpolating if necessary) voltage pattern at the 
        location of catalog sources based on external voltage pattern files
        specified. Parallel processing can be performed.

        Inputs:

        vbeam_files [dictionary] Dictionary containing file locations of 
                    far-field voltage pattern of individual antennas under 
                    keys denoted by antenna labels (string). If there is only
                    one item it will be assumed to be identical for all 
                    antennas. If multiple voltage beam file locations are
                    specified, it must be the same as number of antennas

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

        Antenna voltage beams at the object locations in the upper hemisphere.
        It is a numpy array of shape nsrc x nchan x nant in case of 
        non-indentical antennas or nsrc x nchan x 1 in case of identical 
        antennas. 
        ------------------------------------------------------------------------
        """

        try:
            vbeam_files
        except NameError:
            raise NameError('Input vbeam_files must be specified')

        if not isinstance(vbeam_files, dict):
            raise TypeError('Input vbeam_files must be a dictionary')

        antkeys = NP.asarray(self.antenna_array.antennas.keys())
        vbeamkeys = NP.asarray(vbeam_files.keys())

        commonkeys = NP.intersect1d(antkeys, vbeamkeys)

        if (commonkeys.size != 1) and (commonkeys.size != antkeys.size):
            raise ValueError('Number of voltage pattern files incompatible with number of antennas')

        hemind, upper_altaz = self.upper_hemisphere(lst, obs_date=obs_date)

        if (commonkeys.size == 1) or self.identical_antennas:
            vbeams = interp_beam(vbeam_files[commonkeys[0]], upper_altaz, self.f)
            vbeams = vbeams[:,:,NP.newaxis]
        else:
            if parallel or (nproc is not None):
                list_of_keys = commonkeys.tolist()
                list_of_vbeam_files = [vbfile[akey] for akey in list_of_keys]
                list_of_altaz = [upper_altaz] * commonkeys.size
                list_of_obsfreqs = [self.f] * commonkeys.size
                if nproc is None:
                    nproc = max(MP.cpu_count()-1, 1) 
                else:
                    nproc = min(nproc, max(MP.cpu_count()-1, 1))
                pool = MP.Pool(processes=nproc)
                list_of_vbeams = pool.map(interp_beam_arg_splitter, IT.izip(list_of_vbeam_files, list_of_altaz, list_of_obsfreqs))
                vbeams = NP.asarray(list_of_vbeams)
            else:
                vbeams = None
                for key in commonkeys:
                    vbeam = interp_beam(vbeam_files[key], upper_altaz, self.f)
                    if vbeams is None:
                        vbeams = vbeam[:,:,NP.newaxis]
                    else:
                        vbeams = NP.dstack((vbeams, vbeam[:,:,NP.newaxis]))

        return vbeams
        
    ############################################################################
