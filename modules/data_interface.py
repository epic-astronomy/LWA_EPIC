import numpy as NP
import astropy
from astropy.io import fits
import lwa_operations as LWAO

class DataHandler(object):

    def __init__(self, indata=None):
        self.data = None
        if isinstance(indata, DataHandler):
            self.data = indata
        elif isinstance(indata, str):
            pass
        elif isinstance(indata, dict):
            if 'intype' not in indata:
                raise KeyError('Key "intype" not found in input parameter indata')
            elif indata['intype'] not in ['sim', 'LWA']:
                raise ValueError('Value in key "intype" is not currently accepted.')
            elif indata['intype'] == 'LWA':
                if not isinstance(indata['data'], LWAO.LWAObs):
                    raise TypeError('Data type in input should be an instance of class LWAO.LWAObs')
            elif indata['intype'] == 'sim':
                pass
        else:
            raise TypeError('Input parameter must be of type string, dictionary or an instance of type DataHandler')

    def load(self, indata=None, intype=None):
        pass
