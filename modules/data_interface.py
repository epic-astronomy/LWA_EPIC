import numpy as NP
import astropy
from astropy.io import fits
import lwa_operations as LWAO
import progressbar as PGB

class DataHandler(object):

    def __init__(self, indata=None):
        self.data = None
        self.cable_delays = None
        self.antid = None
        self.antpos = None
        self.timestamps = None
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
                extnames1 = [hdu.header['EXTNAME'] for hdu in hdulist1]
                self.freq = hdulist1['FREQS']
                antid_P1 = hdulist1['Antenna Positions'].data['Antenna']
                antpos_P1 = hdulist1['Antenna Positions'].data['Position']
                timestamps = extnames1[5:]
                antid_P1 = map(str, antid_P1)
                antid_P1 = NP.asarray(antid_P1)
                self.antid = NP.copy(antid_P1)
            else:
                try:
                    hdulist1 = fits.open(indata[0])
                except IOError:
                    raise IOError('File {0} could not be read'.format(indata[0]))
                extnames1 = [hdu.header['EXTNAME'] for hdu in hdulist1]
                self.freq = hdulist1['FREQS'].data
                antid_P1 = hdulist1['Antenna Positions'].data['Antenna']
                antpos_P1 = hdulist1['Antenna Positions'].data['Position']
                self.timestamps = extnames1[5:]
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

    def load(self, indata=None, intype=None):
        pass
