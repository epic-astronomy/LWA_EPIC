import numpy as NP
import my_DSP_modules as DSP
import geometry as GEOM
import scipy.constants as FCNST

#############################################################################

def stochastic_E_spectrum(freq_center, nchan, channel_width, flux_ref=1.0,
                          freq_ref=None, spectral_index=0.0, skypos=None, 
                          ref_point=None, antpos=[0.0,0.0,0.0], verbose=True):

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

    if verbose:
        print '\tArguments verified for compatibility.'
        print '\tSetting up the recipe for producing stochastic Electric field spectra...'

    center_channel = int(NP.floor(0.5*nchan))
    freqs = freq_center + channel_width * (NP.arange(nchan) - center_channel)
    alpha = spectral_index.reshape(-1,1)
    freqs = freqs.reshape(1,-1)
    freq_ratio = freqs / freq_ref.reshape(-1,1)
    fluxes = flux_ref.reshape(-1,1) * (freq_ratio ** alpha)
    sigmas = NP.sqrt(fluxes)
    Ef_amp = sigmas * NP.random.normal(loc=0.0, scale=1.0, size=(nsrc,nchan))
    Ef_phase = NP.exp(1j*NP.random.uniform(low=0.0, high=2*NP.pi, size=(nsrc,nchan)))
    # Ef_phase = 1.0
    Ef = Ef_amp * Ef_phase

    Ef = Ef[:,:,NP.newaxis]
    skypos_dot_antpos = NP.dot(skypos-ref_point, antpos.T)
    k_dot_r_phase = 2.0 * NP.pi * freqs[:,:,NP.newaxis] / FCNST.c * skypos_dot_antpos[:,NP.newaxis,:]
    Ef = Ef * NP.exp(1j * k_dot_r_phase)
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
                            tshift=True, verbose=True):

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
                      'tres'      [numpy vector] Residual delays after removal of 
                                  delays that are integral multiples of delay in 
                                  a bin of the timeseries, in the process of 
                                  phasing of antennas. It is computed only if the 
                                  input parameter 'tshift' is set to True. Length
                                  of the vector is equal to the number of
                                  antennas. If 'tshift' is set to False, the key
                                  'tres' is set to None
                       'tshift'   [numpy vector] if input parameter 'tshift' is
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
                                              verbose=verbose)
    else:
        spectrum_info = stochastic_E_spectrum(freq_center, nchan, channel_width,
                                              flux_ref, freq_ref, spectral_index,
                                              skypos=skypos, antpos=antpos,
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
                             antpos=[0.0,0.0,0.0], verbose=True):

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
                     at the specified frequencies. Units are 
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

    if verbose:
        print '\tArguments verified for compatibility.'
        print '\tSetting up the recipe for producing monochromatic Electric field...'

    nchan = 1
    alpha = spectral_index.reshape(-1)
    freq_ratio = freq / freq_ref
    flux = flux_ref * (freq_ratio ** alpha)
    sigma = NP.sqrt(flux).reshape(-1,1)
    Ef_amp = sigma
    Ef_phase = NP.random.uniform(low=0.0, high=2*NP.pi, size=(nsrc,nchan))
    Ef_sky =  Ef_amp * NP.exp(1j * Ef_phase)
    Ef_matrix = NP.repeat(Ef_sky, nant, axis=1)
    skypos_dot_antpos = NP.dot(skypos-ref_point, antpos.T)
    k_dot_r_phase = 2.0 * NP.pi * (freq/FCNST.c) * skypos_dot_antpos
    Ef_2D = Ef_sky * NP.exp(1j * k_dot_r_phase)
    Ef = NP.sum(Ef_2D, axis=0)
    if verbose:
        print '\tPerformed linear superposition of electric fields from source(s).'
    dictout = {}
    dictout['f'] = freq
    dictout['Ef'] = Ef
    dictout['antpos'] = antpos

    if verbose:
        print 'monochromatic_E_spectrum() executed successfully.\n'

    return dictout

#################################################################################

def monochromatic_E_timeseries(freq_center, nchan, channel_width, flux_ref=1.0,
                               freq_ref=None, spectral_index=0.0, skypos=None, 
                               ref_point=None, antpos=[0.0,0.0,0.0],
                               spectrum=True, verbose=True):

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
 
