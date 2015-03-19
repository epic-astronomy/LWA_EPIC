import numpy as NP
import astropy
from astropy.io import fits
import lwa_operations as LWAO
import progressbar as PGB

#################################################################################

class DataHandler(object):

    """
    ----------------------------------------------------------------------------
    Class to manage Antenna Voltage data

    Attributes:

    latitude    [scalar] Latitude of the antenna locations (in degrees)

    antid       [numpy array of strings] Unique identifier of antennas

    antpos      [numpy array] Antenna positions (in m) specified along local 
                East, North and Up as a 3-column numpy array

    timestamps  [numpy array of strings] Timestamps 

    sample_rate [scalar] Sampling rate (in Hz)

    center_freq [scalar] Center frequency (in Hz)

    freq        [numpy array] Frequencies in channels (in Hz)

    freq_resolution
                [scalar] Frequency resolution (in Hz)

    n_timestamps
                [scalar] Number of timestamps

    n_antennas  [scalar] Number of antennas

    nchan       [scalar] Number of frequency channels

    npol        [scalar] Number of polarizations (fixed to 2)

    data        [numpy array] Complex voltages of shape 
                n_timestamps x n_antennas x nchan x npol. Places where data is
                not available, it is filled with NaN.

    Methods:

    __init__()  Initialize instance of class DataHandler 

    save()      Save the antenna voltage data to disk in a "Common Data Format"
                (CDF)
    ----------------------------------------------------------------------------
    """

    def __init__(self, indata=None):

        """
        ------------------------------------------------------------------------
        Initialize the DataHandler Class which manages reads, manages and writes
        antenna voltage data. 

        Class attributes initialized are:
        latitude, antid, antpos, timestamps, sample_rate, center_freq, freq,
        freq_resolution, n_timestamps, n_antennas, nchan, npol, data
     
        Read docstring of class PolInfo for details on these attributes.

        Inputs:

        indata   [str, list of str, dict] Specifies the input data to initialize
                 from. If string, it must be the FITS file location containing 
                 data saved by this class (it contains both polarizations) or a 
                 FITS file containing reformatted LWA data (one polarization).
                 If provided as a list, it may contain one FITS file location 
                 saved by this class (contains both polarizations) or the first
                 polarization of reformatted LWA data. If it contains two 
                 strings, it must be one for each polarization of reformatted 
                 LWA data. If specified as dictionary, it must be contain the
                 following keys and values:
                 'intype'      [string] specifies type of data. Currently 
                               accepted values are 'sim' and 'LWA' for 
                               simulations and LWA respectively. 
                 'data-block'  [instance of class LWAO.LWAObs] Antenna and 
                               antenna voltage information. Read docstring of
                               class LWAO.LWAObs for more information. This is
                               associated with intype='LWA'. Needs serious
                               development for intype='sim'
        ------------------------------------------------------------------------
        """

        self.latitude = None
        self.data = None
        self.cable_delays = None
        self.antid = None
        self.antpos = None
        self.timestamps = None
        self.sample_rate = None
        self.center_freq = None
        self.freq = None
        self.n_timestamps = None
        self.n_antennas = None
        self.nchan = None
        if isinstance(indata, DataHandler):
            self.data = indata
        elif isinstance(indata, (str,list)):
            if isinstance(indata, str):
                try:
                    hdulist1 = fits.open(indata)
                except IOError:
                    raise IOError('File {0} could not be read'.format(indata))

                if 'format' in hdulist1[0].header:
                    if hdulist1[0].header['format'] == 'CDF':
                        self.latitude = hdulist1['PRIMARY'].header['latitude']
                        self.sample_rate = hdulist1['PRIMARY'].header['sample_rate']
                        self.center_freq = hdulist1['PRIMARY'].header['center_freq']
                        self.freq_resolution = hdulist1['PRIMARY'].header['freq_resolution']
                        self.n_timestamps = hdulist1['PRIMARY'].header['n_timestamps']
                        self.n_antennas = hdulist1['PRIMARY'].header['n_antennas']
                        self.nchan = hdulist1['PRIMARY'].header['nchan']
                        self.npol = hdulist1['PRIMARY'].header['npol']
                        self.timestamps = hdulist1['TIMESTAMPS'].data.field('timestamp')
                        self.antid = hdulist1['ANTENNA INFO'].data.field('labels')
                        self.antpos = hdulist1['ANTENNA INFO'].data.field('Positions')
                        self.freq = hdulist1['Frequencies'].data
                        self.data = hdulist1['real_antenna_voltages'].data + 1j * hdulist1['imag_antenna_voltages'].data
                        hdulist1.close()
                else:
                    extnames1 = [hdu.header['EXTNAME'] for hdu in hdulist1]
                    self.latitude = hdulist1[0].header['latitude']
                    self.sample_rate = hdulist1[0].header['sample_rate']
                    self.center_freq = hdulist1[0].header['center_freq']
                    self.freq = hdulist1['FREQS']
                    antid_P1 = hdulist1['Antenna Positions'].data['Antenna']
                    antpos_P1 = hdulist1['Antenna Positions'].data['Position']
                    timestamps = extnames1[5:]
                    self.freq_resolution = self.freq[1] - self.freq[0]
                    antid_P1 = map(str, antid_P1)
                    antid_P1 = NP.asarray(antid_P1)
                    self.antid = NP.copy(antid_P1)
            else:
                try:
                    hdulist1 = fits.open(indata[0])
                except IOError:
                    raise IOError('File {0} could not be read'.format(indata[0]))

                if 'format' in hdulist1[0].header:
                    if hdulist1[0].header['format'] == 'CDF':
                        self.latitude = hdulist1['PRIMARY'].header['latitude']
                        self.sample_rate = hdulist1['PRIMARY'].header['sample_rate']
                        self.center_freq = hdulist1['PRIMARY'].header['center_freq']
                        self.freq_resolution = hdulist1['PRIMARY'].header['freq_resolution']
                        self.n_timestamps = hdulist1['PRIMARY'].header['n_timestamps']
                        self.n_antennas = hdulist1['PRIMARY'].header['n_antennas']
                        self.nchan = hdulist1['PRIMARY'].header['nchan']
                        self.npol = hdulist1['PRIMARY'].header['npol']
                        self.timestamps = hdulist1['TIMESTAMPS'].data.field('timestamp')
                        self.antid = hdulist1['ANTENNA INFO'].data.field('labels')
                        self.antpos = hdulist1['ANTENNA INFO'].data.field('Positions')
                        self.freq = hdulist1['Frequencies'].data
                        self.data = hdulist1['real_antenna_voltages'].data + 1j * hdulist1['imag_antenna_voltages'].data
                        hdulist1.close()
                else:
                    extnames1 = [hdu.header['EXTNAME'] for hdu in hdulist1]
                    self.latitude = hdulist1[0].header['latitude']
                    self.sample_rate = hdulist1[0].header['sample_rate']
                    self.center_freq = hdulist1[0].header['center_freq']
                    self.freq = hdulist1['FREQS'].data
                    antid_P1 = hdulist1['Antenna Positions'].data['Antenna']
                    antpos_P1 = hdulist1['Antenna Positions'].data['Position']
                    self.timestamps = extnames1[5:]
                    self.freq_resolution = self.freq[1] - self.freq[0]
                    antid_P1 = map(str, antid_P1)
                    antid_P1 = NP.asarray(antid_P1)
                    self.antid = NP.copy(antid_P1)
                    
                    extnames2 = None
                    antid_P2 = None
                    if len(indata) > 1:
                        try:
                            hdulist2 = fits.open(indata[1])
                        except IOError:
                            raise IOError('File {0} could not be read'.format(indata[1]))
                        extnames2 = [hdu.header['EXTNAME'] for hdu in hdulist2]
                        antid_P2 = hdulist2['Antenna Positions'].data['Antenna']
                        antpos_P2 = hdulist2['Antenna Positions'].data['Position']
                        self.timestamps = NP.union1d(extnames1[5:], extnames2[5:])
                        antid_P2 = map(str, antid_P2)
                        antid_P2 = NP.asarray(antid_P2)
                        self.antid = NP.union1d(antid_P1, antid_P2)

                    for antid in self.antid:
                        if antid in antid_P1:
                            if self.antpos is None:
                                self.antpos = antpos_P1[antid_P1==antid,:].reshape(1,-1)
                            else:
                                self.antpos = NP.vstack((self.antpos, antpos_P1[antid_P1==antid,:].reshape(1,-1)))
                        else:
                            if self.antpos is None:
                                self.antpos = antpos_P2[antid_P2==antid,:].reshape(1,-1)
                            else:
                                self.antpos = NP.vstack((self.antpos, antpos_P2[antid_P2==antid,:].reshape(1,-1)))
        
                    self.n_timestamps = len(self.timestamps)
                    self.n_antennas = self.antid.size
                    self.nchan = self.freq.size
        
                    self.data = NP.empty((self.n_timestamps, self.n_antennas, self.nchan, 2), dtype=NP.complex64)
                    self.data.fill(NP.nan)
        
                    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Timestamps '.format(self.n_timestamps), PGB.ETA()], maxval=self.n_timestamps).start()
        
                    for hdu in hdulist1[5:]:
                        timestamp = hdu.header['EXTNAME']
                        it = NP.where(self.timestamps == timestamp)[0]
                        cols = hdu.columns
                        for ic, col in enumerate(cols):
                            ia = NP.where(self.antid == col.name)[0]
                            Et = hdu.data.field(ic)
                            self.data[it,ia,:,0] = Et[:,0] + 1j * Et[:,1]
                        progress.update(it+1)
        
                    progress.finish()
        
                    if len(indata) > 1:
                        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Timestamps '.format(self.n_timestamps), PGB.ETA()], maxval=self.n_timestamps).start()
            
                        for hdu in hdulist2[5:]:
                            timestamp = hdu.header['EXTNAME']
                            it = NP.where(self.timestamps == timestamp)[0]
                            cols = hdu.columns
                            for ic, col in enumerate(cols):
                                ia = NP.where(self.antid == col.name)[0]
                                Et = hdu.data.field(ic)
                                self.data[it,ia,:,1] = Et[:,0] + 1j * Et[:,1]
                            progress.update(it+1)
            
                        progress.finish()
                    
                    hdulist1.close()
                    if len(indata) > 1:
                        hdulist2.close()

        elif isinstance(indata, dict):
            if 'intype' not in indata:
                raise KeyError('Key "intype" not found in input parameter indata')
            elif indata['intype'] not in ['sim', 'LWA']:
                raise ValueError('Value in key "intype" is not currently accepted.')
            elif indata['intype'] == 'LWA':
                if not isinstance(indata['data-block'], LWAO.LWAObs):
                    raise TypeError('Data type in input should be an instance of class LWAO.LWAObs')
                self.data['type'] = indata['intype']
                self.data['pol'] = [indata['data-block'].P1.pol, indata['data-block'].P1.pol]
                self.sample_rate = indata['data-block'].sample_rate
                self.center_freq = indata['data-block'].center_freq
                self.freq = indata['data-block'].freq
                self.freq_resolution = self.freq[1] - self.freq[0]
                self.nchan = self.freq.size
                self.latitude = indata['data-block'].latitude
                antid_P1 = None
                antid_P2 = None                
                if indata['data-block'].P1.stands:
                    antid_P1 = indata['data-block'].P1.stands
                    antpos_P1 = indata['data-block'].P1.antpos
                if indata['data-block'].P2.stands:
                    antid_P2 = indata['data-block'].P2.stands
                    antpos_P2 = indata['data-block'].P2.antpos
                antid_P1 = map(str, antid_P1)
                antid_P2 = map(str, antid_P2)                
                self.antid = NP.union1d(antid_P1, antid_P2)
                self.n_antennas = self.antid.size
                for antid in self.antid:
                    if antid in antid_P1:
                        if self.antpos is None:
                            self.antpos = antpos_P1[antid_P1==antid,:].reshape(1,-1)
                        else:
                            self.antpos = NP.vstack((self.antpos, antpos_P1[antid_P1==antid,:].reshape(1,-1)))
                    else:
                        if self.antpos is None:
                            self.antpos = antpos_P2[antid_P2==antid,:].reshape(1,-1)
                        else:
                            self.antpos = NP.vstack((self.antpos, antpos_P2[antid_P2==antid,:].reshape(1,-1)))

                if indata['data-block'].P1.timestamps:
                    timestamps_P1 = indata['data-block'].P1.timestamps
                if indata['data-block'].P2.timestamps:
                    timestamps_P2 = indata['data-block'].P2.timestamps
                self.timestamps = NP.union1d(timestamps_P1, timestamps_P2)
                self.n_timestamps = self.timestamps.size
                
                self.data = NP.empty((self.n_timestamps, self.n_antennas, self.nchan, 2), dtype=NP.complex64)
                self.data.fill(NP.nan)

                for it in xrange(self.n_timestamps):
                    for ia in xrange(self.n_antennas):
                        if self.timestamps[it] in indata['data-block'].P1.data['tod']:
                            if self.antid[ia] in indata['data-block'].P1.data['tod'][self.timestamps[it]]:
                                self.data[it,ia,:,0] = indata['data-block'].P1.data['tod'][self.timestamps[it]][self.antid[ia]]

                        if self.timestamps[it] in indata['data-block'].P2.data['tod']:
                            if self.antid[ia] in indata['data-block'].P2.data['tod'][self.timestamps[it]]:
                                self.data[it,ia,:,1] = indata['data-block'].P2.data['tod'][self.timestamps[it]][self.antid[ia]]
                
            elif indata['intype'] == 'sim':
                pass
        else:
            raise TypeError('Input parameter must be of type string, dictionary or an instance of type DataHandler')

    #############################################################################

    def save(self, outfile, tabtype='BinTableHDU', overwrite=True, verbose=True):

        """
        ------------------------------------------------------------------------
        outfile     [string] filename with full path to save the antenna voltage 
                    data to.

        Keyword Input(s):

        tabtype     [string] indicates table type for one of the extensions in 
                    the FITS file. Allowed values are 'BinTableHDU' and 
                    'TableHDU' for binary ascii tables respectively. Default is
                    'BinTableHDU'.

        overwrite   [boolean] True indicates overwrite even if a file already 
                    exists. Default = True (allow overwrite)

        verbose     [boolean] If True (default), prints diagnostic and progress
                    messages. If False, suppress printing such messages.
        ------------------------------------------------------------------------
        """

        try:
            outfile
        except NameError:
            raise NameError('No output file specified.')

        use_ascii = False
        if tabtype == 'TableHDU':
            use_ascii = True

        hdulist = []

        hdulist += [fits.PrimaryHDU()]
        hdulist[0].header['format'] = ('CDF', 'Data Format Type')
        hdulist[0].header['latitude'] = (self.latitude, 'Latitude of Antennas [degrees]')
        hdulist[0].header['sample_rate'] = (self.sample_rate, 'Sampling rate [Hz]')
        hdulist[0].header['center_freq'] = (self.center_freq, 'Center Frequency [Hz]')
        hdulist[0].header['freq_resolution'] = (self.freq_resolution, 'Frequency Resolution [Hz]')
        hdulist[0].header['n_timestamps'] = (self.n_timestamps, 'Number of timestamps')
        hdulist[0].header['n_antennas'] = (self.n_antennas, 'Number of antennas')
        hdulist[0].header['nchan'] = (self.nchan, 'Number of frequency channels')
        hdulist[0].header['npol'] = (2, 'Number of polarizations')

        if verbose:
            print '\tCreated a primary HDU.'

        cols = []
        cols += [fits.Column(name='timestamp', format='16A', array=self.timestamps)]
        if astropy.__version__ == '0.4':
            columns = fits.ColDefs(cols, tbtype=tabtype)
        else:
            columns = fits.ColDefs(cols, ascii=use_ascii)

        tbhdu = fits.new_table(columns)
        tbhdu.header.set('EXTNAME', 'TIMESTAMPS')
        hdulist += [tbhdu]

        # hdulist += [fits.ImageHDU(self.timestamps, name='Timestamps')]
        if verbose:
            print '\tCreated an extension for timestamps.'
        
        cols = []
        cols += [fits.Column(name='labels', format='7A', array=self.antid)]
        cols += [fits.Column(name='Positions', format='3D', array=self.antpos)]
        if astropy.__version__ == '0.4':
            columns = fits.ColDefs(cols, tbtype=tabtype)
        else:
            columns = fits.ColDefs(cols, ascii=use_ascii)

        tbhdu = fits.new_table(columns)
        tbhdu.header.set('EXTNAME', 'ANTENNA INFO')
        hdulist += [tbhdu]
        if verbose:
            print '\tCreated an extension for antenna information.'
        
        hdulist += [fits.ImageHDU(self.freq, name='Frequencies')]
        if verbose:
            print '\tCreated an extension for frequencies.'
        
        hdulist += [fits.ImageHDU(self.data.real, name='real_antenna_voltages')]
        hdulist += [fits.ImageHDU(self.data.imag, name='imag_antenna_voltages')]
        if verbose:
            print '\tCreated extensions for real and imaginary parts of observed antenna voltages of size {0[0]} x {0[1]} x {0[2]} x {0[3]}'.format(self.data.shape)

        if verbose:
            print '\tNow writing FITS file to disk...'

        hdu = fits.HDUList(hdulist)
        hdu.writeto(outfile, clobber=overwrite)

        if verbose:
            print '\tAntenna data written successfully to FITS file on disk:\n\t\t{0}\n'.format(outfile)

#################################################################################            
            
