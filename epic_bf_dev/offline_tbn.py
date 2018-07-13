#!/usr/bin/env python

import os
import sys
import json
import time
import numpy
import ctypes
import signal
import logging
import threading

from lsl.common.constants import c as speedOfLight
from lsl.common.stations import lwasv, parseSSMIF
from lsl.reader.ldp import TBNFile
from lsl.writer import fitsidi
from lsl import astro
from lsl.correlator import uvUtils
from lsl.imaging import utils

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.fft import Fft
from bifrost.quantize import quantize as Quantize
from bifrost.unpack import unpack as Unpack
from bifrost.linalg import LinAlg as Correlator
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost import map as BFMap, asarray as BFAsArray
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, stream_synchronize as BFSync, set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()


CHAN_BW = 25000
ANTENNAS = lwasv.getAntennas()
DUMP_TIME = 5   # s


class FillerOp(object):
    def __init__(self, log, oring, filename, ntime_gulp=25000, core=-1):
        self.log = log
        self.oring = oring
        self.filename = filename
        self.ntime_gulp = ntime_gulp
        self.core = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.out_proclog.update({'nring':1, 'ring0':self.oring.name})
        
        self.shutdown_event = threading.Event()
        
    def shutdown(self):
        self.shutdown_event.set()
        
    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        # Load in the metadata and sample the data
        idf = TBNFile(self.filename)
        cfreq = idf.getInfo('freq1')
        srate = idf.getInfo('sampleRate')
        tInt, tStart, data = idf.read(1.0*self.ntime_gulp/CHAN_BW, timeInSamples=True)
        idf.reset()
        
        # Setup the ring metadata and gulp sizes
        ntime = data.shape[1]
        nstand, npol = data.shape[0]/2, 2
        oshape = (ntime,nstand,npol)
        ogulp_size = ntime*nstand*npol*8        # complex64
        self.oring.resize(ogulp_size, buffer_factor=10)
        
        self.size_proclog.update({'nseq_per_gulp': ntime})
        
        # Build the initial ring header
        ohdr = {}
        ohdr['time_tag'] = tStart
        ohdr['seq0']     = 0
        ohdr['chan0']    = int((cfreq - srate/2)/CHAN_BW)
        ohdr['nchan']    = 1
        ohdr['cfreq']    = cfreq
        ohdr['bw']       = srate
        ohdr['nstand']   = nstand
        ohdr['npol']     = npol
        ohdr['nbit']     = 8
        ohdr['complex']  = True
        ohdr_str = json.dumps(ohdr)
        
        ## Fill the ring using the same data over and over again
        with self.oring.begin_writing() as oring:
            with oring.begin_sequence(time_tag=tStart, header=ohdr_str) as oseq:
                prev_time = time.time()
                while not self.shutdown_event.is_set():
                    ## Get the current section to use
                    try:
                        tInt, tStart, data = idf.read(1.0*self.ntime_gulp/CHAN_BW, timeInSamples=True)
                        self.log.debug("Read %.3f at %i", tInt, tStart)
                    except Exception as e:
                        self.log.error("FillerOp: Error - '%s'", str(e))
                        idf.close()
                        self.shutdown()
                        break
                        
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    
                    with oseq.reserve(ogulp_size) as ospan:
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                        
                        ## Setup and load
                        idata = data
                        odata = ospan.data_view(numpy.complex64).reshape(oshape)
                        
                        ## Transpose and reshape to time by stand by pol
                        idata = idata.transpose((1,0))
                        idata = idata.reshape((ntime,nstand,npol))
                        
                        ## Save
                        odata[...] = idata
                        
                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': reserve_time, 
                                              'process_time': process_time,})
        self.log.info("FillerOp - Done")


class FDomainOp(object):
    def __init__(self, log, iring, oring, ntime_gulp=2500, nchan_out=4, core=-1, gpu=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.nchan_out = nchan_out
        self.core = core
        self.gpu = gpu
        
        self.nchan_out = 4
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update( {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=True):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                self.log.info('FDomainOp: Config - %s', ihdr)
                
                # Setup the ring metadata and gulp sizes
                nchan  = self.nchan_out
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                
                igulp_size = self.ntime_gulp*1*nstand*npol * 8		# complex64
                ishape = (self.ntime_gulp/nchan,nchan,nstand,npol)
                ogulp_size = self.ntime_gulp*1*nstand*npol * 1		# ci4
                oshape = (self.ntime_gulp/nchan,nchan,nstand,npol)
                self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size)
                
                # Set the output header
                ohdr = ihdr.copy()
                ohdr['nchan'] = nchan
                ohdr['nbit']  = 4
                ohdr_str = json.dumps(ohdr)
                
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    iseq_spans = iseq.read(igulp_size)
                    while not self.iring.writing_ended():
                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            with oseq.reserve(ogulp_size) as ospan:
                                curr_time = time.time()
                                reserve_time = curr_time - prev_time
                                prev_time = curr_time
                                
                                ## Setup and load
                                idata = ispan.data_view(numpy.complex64).reshape(ishape)
                                odata = ospan.data_view(numpy.int8).reshape(oshape)
                                
                                ## Copy to the GPU
                                gdata = idata.copy(space='cuda')
                                
                                ## FFT
                                fdata = BFArray(shape=gdata.shape, dtype=numpy.complex64, space='cuda')
                                try:
                                    bfft.execute(gdata, fdata, inverse=False)
                                except NameError:
                                    bfft = Fft()
                                    bfft.init(gdata, fdata, axes=1, apply_fftshift=True)
                                    bfft.execute(gdata, fdata, inverse=False)
                                    
                                ## Quantization
                                try:
                                    Quantize(fdata, qdata, scale=8./numpy.sqrt(nchan))
                                except NameError:
                                    qdata = BFArray(shape=fdata.shape, native=False, dtype='ci4', space='cuda')
                                    Quantize(fdata, qdata, scale=8./numpy.sqrt(nchan))
                                    
                                ## Save
                                odata[...] = qdata.copy(space='cuda_host').view(numpy.int8).reshape(oshape)
                                
                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                      'reserve_time': reserve_time, 
                                                      'process_time': process_time,})
                break   # Only do one pass through the loop
        self.log.info("FDomainOp - Done")


class CorrelatorOp(object):
    # Note: Input data are: [time,chan,ant,pol,cpx,8bit]
    def __init__(self, log, iring, oring, ntime_gulp=2500, nchan_max=4, core=-1, gpu=-1):
        self.log   = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.core = core
        self.gpu = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
        self.nchan_max = nchan_max
        self.navg = max([int(self.ntime_gulp*100/25000), int(DUMP_TIME*100)])
        self.gain = 0
        
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        self.bfcc = Correlator()
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=True):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                self.log.info('CorrelatorOp: Config - %s', ihdr)
                
                # Setup the ring metadata and gulp sizes
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                igulp_size = self.ntime_gulp*nchan*nstand*npol*1		# ci4
                ogulp_size = nchan*nstand*npol*nstand*npol*8			# complex64
                ishape = (self.ntime_gulp,nchan,nstand*npol)
                oshape = (nchan,nstand*npol,nstand*npol)
                self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size)
                
                # Setup the correlator dump interval
                navg_seq = self.navg * int(CHAN_BW/100.0)
                navg_seq = int(navg_seq / self.ntime_gulp) * self.ntime_gulp
                gain_act = 1.0 / 2**self.gain / navg_seq
                navg = navg_seq / int(CHAN_BW/100.0)
                
                # Set the output header
                ohdr = ihdr.copy()
                ohdr['nbit'] = 32
                ohdr['navg'] = navg
                ohdr_str = json.dumps(ohdr)
                
                # Setup the intermediate arrays
                nAccumulate = 0
                cdata = numpy.zeros((nchan,nstand*npol,nstand*npol), dtype=numpy.complex64)
                cdata = BFArray(cdata, space='cuda')
                
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    iseq_spans = iseq.read(igulp_size)
                    while not self.iring.writing_ended():
                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            ## Setup and load
                            idata = ispan.data_view(numpy.uint8).reshape(ishape)
                            
                            ## Fix the type
                            tdata = BFArray(shape=ishape, dtype='ci4', native=False, buffer=idata.ctypes.data)
                            
                            ## Copy
                            tdata = tdata.copy(space='cuda')
                            
                            ## Unpack
                            try:
                                Unpack(tdata, udata)
                            except NameError:
                                udata = BFArray(shape=tdata.shape, dtype='ci8', space='cuda')
                                Unpack(tdata, udata)
                                
                            ## Correlate
                            cscale = gain_act if nAccumulate else 0.0
                            cdata = self.bfcc.matmul(gain_act, None, udata.transpose((1,0,2)), cscale, cdata)
                            nAccumulate += self.ntime_gulp
                             
                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            if nAccumulate == navg_seq:
                                self.log.debug("Dumbing correlator after accumlating %i sequences", nAccumulate)
                                with oseq.reserve(ogulp_size) as ospan:
                                    odata = ospan.data_view(numpy.complex64).reshape(oshape)
                                    
                                    curr_time = time.time()
                                    reserve_time = curr_time - prev_time
                                    prev_time = curr_time
                                        
                                    odata[...] = cdata
                                nAccumulate = 0
                            else:
                                reserve_time = 0.0
                                    
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                      'reserve_time': reserve_time, 
                                                      'process_time': process_time,})
                break   # Only do one pass through the loop
        self.log.info("CorrelatorOp - Done")


class FITSIDIOp(object):
    def __init__(self, log, iring, basename='TEST', core=-1, gpu=-1):
        self.log = log
        self.iring = iring
        self.basename = basename
        self.core = core
        self.gpu = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update({'nring':1, 'ring0':self.iring.name})
        
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        for iseq in self.iring.read(guarantee=True):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            self.log.info('FITSIDIOp: Config - %s', ihdr)
            
            # Setup the ring metadata and gulp sizes
            chan0  = ihdr['chan0']
            nchan  = ihdr['nchan']
            nstand = ihdr['nstand']
            npol   = ihdr['npol']
            navg   = ihdr['navg']
            time_tag0 = iseq.time_tag
            time_tag  = time_tag0
            igulp_size = nchan*nstand*npol*nstand*npol*8
            ishape = (nchan,nstand*npol,nstand*npol)
            
            # Setup the phasing terms for zenith
            phases = numpy.zeros((nchan,nstand,npol), dtype=numpy.complex64)
            freq = numpy.fft.fftfreq(nchan, d=1.0/ihdr['bw']) + ihdr['cfreq']
            for i in xrange(nstand):
                ## X
                a = ANTENNAS[2*i + 0]
                delay = a.cable.delay(freq) - a.stand.z / speedOfLight
                phases[:,i,0] = numpy.exp(2j*numpy.pi*freq*delay)
                ## Y
                a = ANTENNAS[2*i + 1]
                delay = a.cable.delay(freq) - a.stand.z / speedOfLight
                phases[:,i,1] = numpy.exp(2j*numpy.pi*freq*delay)
                
            # Get the correlator dump time and setup the intermediate arrays
            readT = navg / 100.0
            pols = ['xx', 'xy', 'yx', 'yy']
            ants = [a for a in ANTENNAS if a.pol == 0]
            freq = numpy.fft.fftfreq(ihdr['nchan'], d=1.0/ihdr['bw']) + ihdr['cfreq']
            blList = uvUtils.getBaselines(ANTENNAS[0::2], IncludeAuto=True)
            vis = numpy.zeros((len(blList),nchan), dtype=numpy.complex64)
            
            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            while not self.iring.writing_ended():
                intCount = 0
                
                for ispan in iseq_spans:
                    if ispan.size < igulp_size:
                        continue # Ignore final gulp
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    
                    ## Setup and load
                    idata = ispan.data_view(numpy.complex64).reshape(ishape)
                    ldata = idata.copy(space='cuda')
                    BFMap('a(i,j,j-1) = Complex<float>(a(i,j-1,j).real, -a(i,j-1,j).imag)', {'a':ldata}, axis_names=('i','j','k'), shape=ldata.shape)
                    odata = ldata.copy(space='system')
                    
                    ## Move the axes around to get them in an order that we know
                    odata = odata.reshape(nchan,nstand,2,nstand,2)
                    odata = numpy.swapaxes(odata, 2, 3)
                    
                    ## Start the FITS-IDI file
                    outname = '%s.FITS_%i' % (self.basename.upper(), intCount)
                    if os.path.exists(outname):
                        os.unlink(outname)
                    fits = fitsidi.ExtendedIDI(outname, refTime=time_tag0/196e6)
                    fits.setStokes(pols)
                    fits.setFrequency(freq)
                    fits.setGeometry(lwasv, ants)
                    
                    ## Set the observation time and start adding polarization data
                    obsTime = astro.unix_to_taimjd(time_tag0/196e6 + readT*intCount)
                    ### XX
                    if 'xx' in pols:
                        k = 0
                        for i in xrange(nstand):
                            for j in xrange(i, nstand):
                                vis[k,:] = odata[:,j,i,0,0] * phases[:,i,0]/phases[:,j,0]
                                k += 1
                        fits.addDataSet(obsTime, readT, blList, vis*2.0, pol='xx')
                    ### XY
                    if 'xy' in pols:
                        k = 0
                        for i in xrange(nstand):
                            for j in xrange(i, nstand):
                                vis[k,:] = odata[:,j,i,1,0] * phases[:,i,0]/phases[:,j,1]
                                k += 1
                        fits.addDataSet(obsTime, readT, blList, vis*1.0, pol='xy')
                    ### YX
                    if 'yx' in pols:
                        k = 0
                        for i in xrange(nstand):
                            for j in xrange(i, nstand):
                                vis[k,:] = odata[:,j,i,0,1] * phases[:,i,1]/phases[:,j,0]
                                k += 1
                        fits.addDataSet(obsTime, readT, blList, vis*1.0, pol='yx')
                    ### YY
                    if 'yy' in pols:
                        k = 0
                        for i in xrange(nstand):
                            for j in xrange(i, nstand):
                                vis[k,:] = odata[:,j,i,1,1] * phases[:,i,1]/phases[:,j,1]
                                k += 1
                        fits.addDataSet(obsTime, readT, blList, vis*1.0, pol='yy')
                        
                    ## Save and proceed
                    fits.write()
                    fits.close()
                    self.log.debug('Wrote %s to disk', fits.filename)
                    intCount += 1
                    
                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': 0.0, 
                                              'process_time': process_time,})
            break   # Only do one pass through the loop
        self.log.info("FITSIDIOp - Done")


class ImagingOp(object):
    def __init__(self, log, iring, basename='TEST', core=-1, gpu=-1):
        self.log = log
        self.iring = iring
        self.basename = basename
        self.core = core
        self.gpu = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update({'nring':1, 'ring0':self.iring.name})
        
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        for iseq in self.iring.read(guarantee=True):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            self.log.info('ImagingOp: Config - %s', ihdr)
            
            # Setup the ring metadata and gulp sizes
            chan0  = ihdr['chan0']
            nchan  = ihdr['nchan']
            nstand = ihdr['nstand']
            npol   = ihdr['npol']
            navg   = ihdr['navg']
            time_tag0 = iseq.time_tag
            time_tag  = time_tag0
            igulp_size = nchan*nstand*npol*nstand*npol*8
            ishape = (nchan,nstand*npol,nstand*npol)
            
            # Setup the phasing terms for zenith
            phases = numpy.zeros((nchan,nstand,npol), dtype=numpy.complex64)
            freq = numpy.fft.fftfreq(nchan, d=1.0/ihdr['bw']) + ihdr['cfreq']
            for i in xrange(nstand):
                ## X
                a = ANTENNAS[2*i + 0]
                delay = a.cable.delay(freq) - a.stand.z / speedOfLight
                phases[:,i,0] = numpy.exp(2j*numpy.pi*freq*delay)
                ## Y
                a = ANTENNAS[2*i + 1]
                delay = a.cable.delay(freq) - a.stand.z / speedOfLight
                phases[:,i,1] = numpy.exp(2j*numpy.pi*freq*delay)
                
            # Get the correlator dump time and setup the intermediate arrays
            readT = navg / 100.0
            pols = ['xx', 'yy', 'xy', 'yx']
            ants = [a for a in ANTENNAS if a.pol == 0]
            freq = numpy.fft.fftfreq(ihdr['nchan'], d=1.0/ihdr['bw']) + ihdr['cfreq']
            uvw = uvUtils.computeUVW(ants, HA=0.0, dec=lwasv.lat*180/numpy.pi, freq=freq, 
                                site=lwasv.getObserver(), IncludeAuto=True)
            wgt = numpy.zeros(freq.size, dtype=numpy.float32)
            msk = numpy.zeros(freq.size, dtype=numpy.bool)
            
            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            while not self.iring.writing_ended():
                intCount = 0
                
                for ispan in iseq_spans:
                    if ispan.size < igulp_size:
                        continue # Ignore final gulp
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    
                    ## Setup and load
                    idata = ispan.data_view(numpy.complex64).reshape(ishape)
                    ldata = idata.copy(space='cuda')
                    BFMap('a(i,j,j-1) = Complex<float>(a(i,j-1,j).real, -a(i,j-1,j).imag)', {'a':ldata}, axis_names=('i','j','k'), shape=ldata.shape)
                    odata = ldata.copy(space='system')
                    
                    ## Move the axes around to get them in an order that we know
                    odata = odata.reshape(nchan,nstand,2,nstand,2)
                    odata = numpy.swapaxes(odata, 2, 3)
                    
                    ## Build the data dictionary
                    dataDict = {'freq':freq, 'bls':{}, 'uvw':{}, 'vis':{}, 'wgt':{}, 'msk':{}}
                    for pol in pols:
                        for key in ('bls', 'uvw', 'vis', 'wgt', 'msk'):
                            dataDict[key][pol] = []
                            
                    ## Fill the data dictionary
                    k = 0
                    for i in xrange(nstand):
                        for j in xrange(i, nstand):
                            ### Drop auto-correlations
                            if i == j:
                                k += 1
                                continue
                            ### Drop baselines with antenna 256 - they are long
                            if i == 255 or j == 255:
                                k += 1
                                continue
                            ### Fill the polarizations
                            for pol in pols:
                                p0 = 0 if pol[1] == 'x' else 1
                                p1 = 0 if pol[0] == 'x' else 1
                                dataDict['bls'][pol].append( (i,j) )
                                dataDict['uvw'][pol].append( uvw[k,:,:] )
                                dataDict['vis'][pol].append( odata[:,j,i,p1,p0] *  phases[:,i,p0]/phases[:,j,p1] )
                                dataDict['wgt'][pol].append( wgt )
                                dataDict['msk'][pol].append( msk )
                            k += 1
                            
                    ## Image
                    fig = plt.figure()
                    for i,pol in enumerate(pols):
                        ax = fig.add_subplot(2, 2, i+1)
                        img = utils.buildGriddedImage(dataDict, MapSize=80, MapRes=0.5, MapWRes=1.0, 
                                                pol=pol, verbose=False)
                        cb = ax.imshow(img.image(center=(80,80)), origin='lower', interpolation='nearest')
                        fig.colorbar(cb, ax=ax)
                        ax.set_title(pol)
                        
                    ## Save and proceed
                    outname = '%s-%04i.png' % (self.basename, intCount)
                    if os.path.exists(outname):
                        os.unlink(outname)
                    fig.savefig(outname)
                    self.log.debug('Wrote %s to disk', outname)
                    intCount += 1
                    
                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': 0.0, 
                                              'process_time': process_time,})
            break   # Only do one pass through the loop
        self.log.info("ImagingOp - Done")


def main(args):
    filename = args[0]
    
    # Logging setup
    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    logHandler = logging.StreamHandler(sys.stdout)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)
    log.setLevel(logging.DEBUG)
    
    # Setup the cores and GPUs to use
    cores = [0, 1, 2, 3, 4]
    gpus  = [0, 0, 0, 0, 0]
    
    # Setup the signal handling
    ops = []
    shutdown_event = threading.Event()
    def handle_signal_terminate(signum, frame):
        SIGNAL_NAMES = dict((k, v) for v, k in \
                            reversed(sorted(signal.__dict__.items()))
                            if v.startswith('SIG') and \
                            not v.startswith('SIG_'))
        log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
        try:
            ops[0].shutdown()
        except IndexError:
            pass
        shutdown_event.set()
    for sig in [signal.SIGHUP,
                signal.SIGINT,
                signal.SIGQUIT,
                signal.SIGTERM,
                signal.SIGTSTP]:
        signal.signal(sig, handle_signal_terminate)
        
    # Create the rings that we need
    capture_ring = Ring(name="capture")
    fdomain_ring = Ring(name="fdomain", space='cuda_host')
    vis_ring     = Ring(name="vis", space='cuda')
    
    # Setup the processing blocks
    ops.append(FillerOp(log, capture_ring,
                        filename, ntime_gulp=500, core=cores.pop(0)))
    ops.append(FDomainOp(log, capture_ring, fdomain_ring, 
                         ntime_gulp=500, core=cores.pop(0), gpu=gpus.pop(0)))
    ops.append(CorrelatorOp(log, fdomain_ring, vis_ring, 
                            ntime_gulp=500, core=cores.pop(0), gpu=gpus.pop(0)))
    #ops.append(FITSIDIOp(log, vis_ring, basename=os.path.basename(filename), 
    #                     core=cores.pop(0), gpu=gpus.pop(0)))
    ops.append(ImagingOp(log, vis_ring, basename=os.path.basename(filename), 
                         core=cores.pop(0), gpu=gpus.pop(0)))
    
    # Launch everything and wait
    threads = [threading.Thread(target=op.main) for op in ops]
    
    log.info("Launching %i thread(s)", len(threads))
    for thread in threads:
        thread.daemon = True
        thread.start()
    log.info("Waiting for reader thread to finish")
    while threads[0].is_alive() and not shutdown_event.is_set():
        time.sleep(0.5)
    log.info("Waiting for remaining threads to finish")
    for thread in threads:
        thread.join()
    log.info("All done")


if __name__ == '__main__':
    main(sys.argv[1:])
    