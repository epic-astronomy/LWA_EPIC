#!/usr/bin/env python

## Core Python Includes
from __future__ import print_function
import signal
import logging
import time
import json
import os
import sys
import threading
import argparse
import numpy
import time
from collections import deque
from scipy.fftpack import fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import ctypes

## Profiling Includes
import cProfile
import pstats

## Bifrost Includes
from bifrost.address import Address as BF_Address
from bifrost.udp_socket import UDPSocket as BF_UDPSocket
from bifrost.udp_capture import UDPCapture as BF_UDPCapture
from bifrost.ring import Ring
from bifrost.unpack import unpack as Unpack
from bifrost.quantize import quantize as Quantize
from bifrost.reduce import reduce as Reduce
from bifrost.proclog import ProcLog
from bifrost.libbifrost import bf
from bifrost.fft import Fft
from bifrost.fft_shift import fft_shift_2d
from bifrost.romein import Romein
import bifrost
import bifrost.affinity
from bifrost.libbifrost import bf
from bifrost.ndarray import memset_array, copy_array
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()

## LWA Software Library Includes
from lsl.common.constants import c as speedOfLight
from lsl.writer import fitsidi
from lsl.reader.ldp import TBNFile, TBFFile
from lsl.common.stations import lwasv, parseSSMIF

#################### Trigger Processing #######################

TRIGGER_ACTIVE = threading.Event()

######################## Profiling ############################

def enable_thread_profiling():
    '''Monkey-patch Thread.run to enable global profiling.

    Each thread creates a local profiler; statistics are pooled
    to the global stats object on run completion.'''
    threading.Thread.stats = None
    thread_run = threading.Thread.run

    def profile_run(self):
        self._prof = cProfile.Profile()
        self._prof.enable()
        thread_run(self)
        self._prof.disable()

        if threading.Thread.stats is None:
            threading.Thread.stats = pstats.Stats(self._prof)
        else:
            threading.Thread.stats.add(self._prof)

    threading.Thread.run = profile_run


def get_thread_stats():
      stats = getattr(threading.Thread, 'stats', None)
      if stats is None:
          raise ValueError, 'Thread profiling was not enabled,'
      'or no threads finished running.'
      return stats

###############################################################


################ Frequency-Dependent Locations ################

def GenerateLocations(lsl_locs, frequencies, ntime, nchan, npol, grid_size=64, grid_resolution=20/60.):
    
    delta = (2*grid_size*numpy.sin(numpy.pi*grid_resolution/360))**-1
    chan_wavelengths = speedOfLight/frequencies
    sample_grid = chan_wavelengths*delta    
    sll = sample_grid[0] / chan_wavelengths[0]
    lsl_locs = lsl_locs.T
    lsl_locs = lsl_locs.copy()

    lsl_locsf = numpy.zeros(shape=(3,npol,nchan,lsl_locs.shape[1]))
    for l in numpy.arange(3):
        for i in numpy.arange(nchan):
            lsl_locsf[l,:,i,:] = lsl_locs[l,:]/sample_grid[i]

            # I'm sure there's a more numpy way of doing this.
            for p in numpy.arange(npol):
                lsl_locsf[l,p,i,:] -= numpy.min(lsl_locsf[l,p,i,:])


    
    #Calculate grid size needed
    range_u = numpy.max(lsl_locsf[0,...]) - numpy.min(lsl_locsf[0,...])
    range_v = numpy.max(lsl_locsf[0,...]) - numpy.min(lsl_locsf[0,...])
    
    # Centre locations slightly
    for l in numpy.arange(3):
        for i in numpy.arange(nchan):
            for p in numpy.arange(npol):
                lsl_locsf[l,p,i,:] += (grid_size - numpy.max(lsl_locsf[l,p,i,:]))/2
                


    # Tile them for ntime...
    locx = numpy.tile(lsl_locsf[0,...],(ntime,1,1,1))
    locy = numpy.tile(lsl_locsf[1,...],(ntime,1,1,1))
    locz = numpy.tile(lsl_locsf[2,...],(ntime,1,1,1))
    # .. and then stick them all into one large array
    locc = numpy.concatenate([[locx,], [locy,], [locz,]]).transpose(0,1,3,4,2).copy()

    return delta, locc, sll
    
    

    
    


###############################################################

######################### EPIC ################################


class TBNOfflineCaptureOp(object):
    def __init__(self, log, oring, filename, chan_bw=25000, profile=False, core=-1):
        self.log = log
        self.oring = oring
        self.filename = filename
        self.core = core
        self.profile = profile
        self.chan_bw = 25000

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
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1,
                                  'core0': bifrost.affinity.get_core(),})

        idf = TBNFile(self.filename)
        cfreq = idf.getInfo('freq1')
        srate = idf.getInfo('sampleRate')
        tInt, tStart, data = idf.read(0.1, timeInSamples=True)

        # Setup the ring metadata and gulp sizes
        ntime = data.shape[1]
        nstand, npol = data.shape[0]/2, 2
        oshape = (ntime,nstand,npol)
        ogulp_size = ntime*nstand*npol*8		# complex64
        self.oring.resize(ogulp_size, buffer_factor=10)

        self.size_proclog.update({'nseq_per_gulp': ntime})

        # Build the initial ring header
        ohdr = {}
        ohdr['time_tag'] = tStart
        ohdr['seq0']     = 0
        ohdr['chan0']    = int((cfreq - srate/2)/self.chan_bw)
        ohdr['nchan']    = 1
        ohdr['cfreq']    = cfreq
        ohdr['bw']       = srate
        ohdr['nstand']   = nstand
        ohdr['npol']     = npol
        ohdr['nbit']     = 8
        ohdr['complex']  = True
        ohdr['axes']     = 'time,stand,pol'
        ohdr_str = json.dumps(ohdr)

        ## Fill the ring using the same data over and over again
        with self.oring.begin_writing() as oring:
            with oring.begin_sequence(time_tag=tStart, header=ohdr_str) as oseq:
                prev_time = time.time()
                if self.profile:
                    spani = 0
                while not self.shutdown_event.is_set():
                    ## Get the current section to use
                    try:
                        _, _, next_data = idf.read(0.1, timeInSamples=True)
                    except Exception as e:
                        print("TBNFillerOp: Error - '%s'" % str(e))
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

                    data = next_data

                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time,
                                              'reserve_time': reserve_time,
                                              'process_time': process_time,})
                    if self.profile:
                        spani += 1
                        if spani >= 10:
                            sys.exit()
                            break
        print("TBNFillerOp - Done")


class FDomainOp(object):
    def __init__(self, log, iring, oring, ntime_gulp=2500, nchan_out=1, profile=False, core=-1, gpu=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.nchan_out = nchan_out
        self.core = core
        self.gpu = gpu
        self.profile = profile

        self.nchan_out = nchan_out

        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")

        self.in_proclog.update( {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update({'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def main(self):
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1,
                                  'core0': bifrost.affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=True):
                ihdr = json.loads(iseq.header.tostring())

                self.sequence_proclog.update(ihdr)
                print('FDomainOp: Config - %s' % ihdr)

                # Setup the ring metadata and gulp sizes
                nchan  = self.nchan_out
                nstand = ihdr['nstand']
                npol   = ihdr['npol']

                igulp_size = self.ntime_gulp*1*nstand*npol * 8		# complex64
                ishape = (self.ntime_gulp/nchan,nchan,nstand,npol)
                ogulp_size = self.ntime_gulp*1*nstand*npol * 1		# ci4
                oshape = (self.ntime_gulp/nchan,nchan,nstand,npol)
                #self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size,buffer_factor=5)

                # Set the output header
                ohdr = ihdr.copy()
                ohdr['nchan'] = nchan
                ohdr['nbit']  = 4
                ohdr['axes']  = 'time,chan,stand,pol'
                ohdr_str = json.dumps(ohdr)

                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    print("FDomain Out Sequence!")
                    iseq_spans = iseq.read(igulp_size)
                    while not self.iring.writing_ended():
                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time

                            if self.profile:
                                spani = 0

                            with oseq.reserve(ogulp_size) as ospan:
                                curr_time = time.time()
                                reserve_time = curr_time - prev_time
                                prev_time = curr_time

                                ## Setup and load
                                idata = ispan.data_view(numpy.complex64).reshape(ishape)

                                odata = ospan.data_view(numpy.int8).reshape(oshape)

                                ## FFT, shift, and phase
                                fdata = fft(idata, axis=1)
                                fdata = numpy.fft.fftshift(fdata, axes=1)
                                fdata = bifrost.ndarray(fdata, space='system')

                                ## Quantization
                                try:
                                    Quantize(fdata, qdata, scale=1./numpy.sqrt(nchan))
                                except NameError:
                                    qdata = bifrost.ndarray(shape=fdata.shape, native=False, dtype='ci4')
                                    Quantize(fdata, qdata, scale=1./numpy.sqrt(nchan))

                                ## Save
                                odata[...] = qdata.copy(space='cuda_host').view(numpy.int8).reshape(oshape)

                                if self.profile:
                                    spani += 1
                                    if spani >= 10:
                                        sys.exit()
                                        break

                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time,
                                                      'reserve_time': reserve_time,
                                                      'process_time': process_time,})
                    # Only do one pass through the loop
        print("FDomainOp - Done")


class TBFOfflineCaptureOp(object):
    def __init__(self, log, oring, filename, chan_bw=25000, profile=False, core=-1):
        self.log = log
        self.oring = oring
        self.filename = filename
        self.core = core
        self.profile = profile
        self.chan_bw = 25000

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
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1,
                                  'core0': bifrost.affinity.get_core(),})

        idf = TBFFile(self.filename)
        srate = idf.getInfo('sampleRate')
        chans = numpy.round(idf.getInfo('freq1') / srate).astype(numpy.int32)
        chan0 = int(chans[0])
        nchan = len(chans)
        tInt, tStart, data = idf.read(0.1, timeInSamples=True)
        
        # Setup the ring metadata and gulp sizes
        ntime = data.shape[2]
        nstand, npol = data.shape[0]/2, 2
        oshape = (ntime,nchan,nstand,npol)
        ogulp_size = ntime*nchan*nstand*npol*1        # ci4
        self.oring.resize(ogulp_size)

        self.size_proclog.update({'nseq_per_gulp': ntime})

        # Build the initial ring header
        ohdr = {}
        ohdr['time_tag'] = tStart
        ohdr['seq0']     = 0
        ohdr['chan0']    = chan0
        ohdr['nchan']    = nchan
        ohdr['cfreq']    = (chan0 + 0.5*(nchan-1))*srate
        ohdr['bw']       = srate*nchan
        ohdr['nstand']   = nstand
        ohdr['npol']     = npol
        ohdr['nbit']     = 4
        ohdr['complex']  = True
        ohdr['axes']     = 'time,chan,stand,pol'
        ohdr_str = json.dumps(ohdr)

        ## Fill the ring using the same data over and over again
        with self.oring.begin_writing() as oring:
            with oring.begin_sequence(time_tag=tStart, header=ohdr_str) as oseq:
                prev_time = time.time()
                if self.profile:
                    spani = 0
                while not self.shutdown_event.is_set():
                    ## Get the current section to use
                    try:
                        _, _, next_data = idf.read(0.1, timeInSamples=True)
                    except Exception as e:
                        print("TBFFillerOp: Error - '%s'" % str(e))
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

                        odata = ospan.data_view(numpy.int8).reshape(oshape)

                        ## Transpose and reshape to time by channel by stand by pol
                        idata = idata.transpose((2,1,0))
                        idata = idata.reshape((ntime,nchan,nstand,npol))
                        idata = idata.copy()
                        
                        ## Quantization
                        try:
                            Quantize(idata, qdata, scale=1./numpy.sqrt(nchan))
                        except NameError:
                            qdata = bifrost.ndarray(shape=idata.shape, native=False, dtype='ci4')
                            Quantize(idata, qdata, scale=1.0)
                            
                        ## Save
                        odata[...] = qdata.copy(space='cuda_host').view(numpy.int8).reshape(oshape)
                        
                    data = next_data

                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time,
                                              'reserve_time': reserve_time,
                                              'process_time': process_time,})
                    if self.profile:
                        spani += 1
                        if spani >= 10:
                            sys.exit()
                            break
        print("TBFFillerOp - Done")


## For when we don't need to care about doing the F-Engine ourself.
## TODO: Implement this come implementation time...
FS = 196.0e6
CHAN_BW = 25.0e3
ADP_EPOCH = datetime.datetime(1970, 1, 1)

class FEngineCaptureOp(object):
    '''
    Receives Fourier Spectra from LWA FPGA
    '''
    def __init__(self, log, *args, **kwargs):
        self.log = log
        self.args = args
        self.kwargs = kwargs
        self.utc_start = self.kwargs['utc_start']
        del self.kwargs['utc_start']
        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def seq_callback(self, seq0, chan0, nchan, nsrc,
                    time_tag_ptr, hdr_ptr, hdr_size_ptr):
        timestamp0 = int((self.utc_start - ADP_EPOCH).total_seconds())
        time_tag0  = timestamp0 * int(FS)
        time_tag   = time_tag0 + seq0*(int(FS)//int(CHAN_BW))
        print("++++++++++++++++ seq0     =", seq0)
        print("                 time_tag =", time_tag)
        time_tag_ptr[0] = time_tag
        hdr = {
            'time_tag': time_tag,
            'seq0':     seq0,
            'chan0':    chan0,
            'nchan':    nchan,
            'cfreq':    (chan0 + 0.5*(nchan-1))*CHAN_BW,
            'bw':       nchan*CHAN_BW,
            'nstand':   nsrc*16,
            #'stand0':   src0*16, # TODO: Pass src0 to the callback too(?)
            'npol':     2,
            'complex':  True,
            'nbit':     4,
            'axes':     'time,chan,stand,pol'
        }
        print("******** CFREQ:", hdr['cfreq'])
        hdr_str = json.dumps(hdr)
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        self.header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(self.header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0

    def main(self):
        seq_callback = bf.BFudpcapture_sequence_callback(self.seq_callback)
        with BF_UDPCapture(*self.args,
                        sequence_callback=seq_callback,
                        **self.kwargs) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
        del capture

class DecimationOp(object):
    def __init__(self, log, iring, oring, ntime_gulp=2500, nchan_out=1, npol_out=2, guarantee=True, core=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.nchan_out = nchan_out
        self.npol_out = npol_out
        self.guarantee = guarantee
        self.core = core

        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")

        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})

    def main(self):
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1,
                                  'core0': bifrost.affinity.get_core(),})

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())

                self.sequence_proclog.update(ihdr)

                self.log.info("Decimation: Start of new sequence: %s", str(ihdr))

                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                chan0  = ihdr['chan0']

                igulp_size = self.ntime_gulp*nchan*nstand*npol*1                     # ci4
                ishape = (self.ntime_gulp,nchan,nstand,npol)
                ogulp_size = self.ntime_gulp*self.nchan_out*nstand*self.npol_out*1   # ci4
                oshape = (self.ntime_gulp,self.nchan_out,nstand,self.npol_out)
                self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size)#, obuf_size)

                ohdr = ihdr.copy()
                ohdr['nchan'] = self.nchan_out
                ohdr['npol']  = self.npol_out
                ohdr['cfreq'] = (chan0 + 0.5*(self.nchan_out-1))*CHAN_BW
                ohdr['bw']    = self.nchan_out*CHAN_BW
                ohdr_str = json.dumps(ohdr)

                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    for ispan in iseq.read(igulp_size):
                        if ispan.size < igulp_size:
                            continue # Ignore final gulp
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time

                        with oseq.reserve(ogulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time

                            idata = ispan.data_view(numpy.uint8).reshape(ishape)
                            odata = ospan.data_view(numpy.uint8).reshape(oshape)

                            sdata = idata[:,:self.nchan_out,:,:]
                            if self.npol_out != npol:
                                sdata = sdata[:,:,:,:self.npol_out]
                            odata[...] = sdata

                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time,
                                                      'reserve_time': reserve_time,
                                                      'process_time': process_time,})

class TransposeOp(object):
    def __init__(self, log, iring, oring, ntime_gulp=2500, guarantee=True, core=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.guarantee = guarantee
        self.core = core

        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")

        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})

    def main(self):
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1,
                                  'core0': bifrost.affinity.get_core(),})

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())

                self.sequence_proclog.update(ihdr)

                self.log.info("Transpose: Start of new sequence: %s", str(ihdr))

                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                chan0  = ihdr['chan0']

                igulp_size = self.ntime_gulp*nchan*nstand*npol*1        # ci4
                ishape = (self.ntime_gulp,nchan,nstand,npol)
                ogulp_size = self.ntime_gulp*nchan*npol*nstand*1        # ci4
                oshape = (self.ntime_gulp,nchan,npol,nstand)
                self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size)#, obuf_size)

                ohdr = ihdr.copy()
                ohdr['axes'] = 'time,chan,pol,stand'
                ohdr_str = json.dumps(ohdr)

                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    for ispan in iseq.read(igulp_size):
                        if ispan.size < igulp_size:
                            continue # Ignore final gulp
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time

                        with oseq.reserve(ogulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time

                            idata = ispan.data_view(numpy.uint8).reshape(ishape)
                            odata = ospan.data_view(numpy.uint8).reshape(oshape)

                            idata = idata.transpose(0,1,3,2)
                            odata[...] = idata.copy()

                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time,
                                                      'reserve_time': reserve_time,
                                                      'process_time': process_time,})

class CalibrationOp(object):
    def __init__(self, log, iring, oring, *args, **kwargs):
        pass

class MOFFCorrelatorOp(object):
    def __init__(self, log, iring, oring, antennas, grid_size, grid_resolution, 
                 ntime_gulp=2500, accumulation_time=10000, core=-1, gpu=-1, 
                 remove_autocorrs = False, benchmark=False, profile=False, 
                 *args, **kwargs):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.accumulation_time=accumulation_time

        self.antennas = antennas
        locations = numpy.empty(shape=(0,3))
        for ant in self.antennas:
            locations = numpy.vstack((locations,[ant.stand[0],ant.stand[1],ant.stand[2]]))
        locations = numpy.delete(locations, list(range(0,locations.shape[0],2)),axis=0)
        locations[255,:] = 0.0
        self.locations = locations
        
        self.grid_size = grid_size
        self.grid_resolution  = grid_resolution
        
        self.core = core
        self.gpu = gpu
        self.remove_autocorrs = remove_autocorrs
        self.benchmark = benchmark
        self.newflag = True
        self.profile = profile
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")

        self.in_proclog.update( {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update({'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
        self.ant_extent = 1
        
        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()


    def main(self):
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1,
                                  'core0': bifrost.affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})

        runtime_history = deque([], 50)
        accum = 0
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=True):
                ihdr = json.loads(iseq.header.tostring())
                self.sequence_proclog.update(ihdr)
                self.log.info('MOFFCorrelatorOp: Config - %s' % ihdr)
                chan0 = ihdr['chan0']
                nchan = ihdr['nchan']
                nstand = ihdr['nstand']
                npol = ihdr['npol']
                self.newflag = True
                accum = 0
                
                igulp_size = self.ntime_gulp * nchan * nstand * npol * 1 # ci4
                itshape = (self.ntime_gulp,nchan,nstand,npol)
                
                
                freq = (chan0 + numpy.arange(nchan))*CHAN_BW
                sampling_length, locs, sll = GenerateLocations(self.locations, freq, 
                                                               self.ntime_gulp, nchan, npol, 
                                                               grid_size=self.grid_size,
                                                               grid_resolution=self.grid_resolution)
                try:
                    copy_array(self.locs, bifrost.ndarray(locs.astype(numpy.int32)))
                except AttributeError:
                    self.locs = bifrost.ndarray(locs.astype(numpy.int32), space='cuda')
                    
                ohdr = ihdr.copy()
                ohdr['nbit'] = 64


                ohdr['npol'] = npol**2 # Because of cross multiplying shenanigans
                ohdr['grid_size_x'] = self.grid_size
                ohdr['grid_size_y'] = self.grid_size
                ohdr['axes'] = 'time,chan,pol,gridy,gridx'
                ohdr['sampling_length_x'] = sampling_length
                ohdr['sampling_length_y'] = sampling_length
                ohdr['accumulation_time'] = self.accumulation_time
                ohdr['FS'] = FS
                ohdr['latitude'] = lwasv.lat * 180. / numpy.pi
                ohdr['longitude'] = lwasv.lon * 180. / numpy.pi
                ohdr['telescope'] = 'LWA-SV'
                ohdr['data_units'] = 'UNCALIB'
                if ohdr['npol'] == 1:
                    ohdr['pols'] = ['xx']
                elif ohdr['npol'] == 2:
                    ohdr['pols'] = ['xx', 'yy']
                elif ohdr['npol'] == 4:
                    ohdr['pols'] = ['xx', 'xy', 'yx', 'yy']
                else:
                    raise ValueError('Cannot write fits file without knowing polarization list')
                ohdr_str = json.dumps(ohdr)

                # Setup the kernels to include phasing terms for zenith
                # Phases are Ntime x Nchan x Nstand x Npol x extent x extent
                freq.shape += (1,1)
                phases = numpy.zeros((self.ntime_gulp,nchan,nstand,npol,self.ant_extent,self.ant_extent), dtype=numpy.complex64)
                for i in xrange(nstand):
                    ## X
                    a = self.antennas[2*i + 0]
                    delay = a.cable.delay(freq) - a.stand.z / speedOfLight
                    phases[:,:,i,0,:,:] = numpy.exp(2j*numpy.pi*freq*delay)
                    phases[:,:,i,0,:,:] /= numpy.sqrt(a.cable.gain(freq))
                    if a.getStatus() < 33:
                        ### Mask out a known bad or suspect antenna
                        phases[:,:,i,0,:,:] = 0.0
                    if npol == 2:
                        ## Y
                        a = self.antennas[2*i + 1]
                        delay = a.cable.delay(freq) - a.stand.z / speedOfLight
                        phases[:,:,i,1,:,:] = numpy.exp(2j*numpy.pi*freq*delay)
                        phases[:,:,i,1,:,:] /= numpy.sqrt(a.cable.gain(freq))
                        if a.getStatus() < 33:
                            ### Mask out a known bad or suspect antenna
                            phases[:,:,i,1,:,:] = 0.0
                    ## Explicit outrigger masking - we probably want to do
                    ## away with this at some point
                    if a.stand.id == 256:
                        phases[:,:,i,:,:,:] = 0.0
                phases = phases.conj()
                phases = bifrost.ndarray(phases)
                try:
                    copy_array(gphases, phases)
                except NameError:
                    gphases = phases.copy(space='cuda')
                    
                oshape = (1,nchan,npol**2,self.grid_size,self.grid_size)
                ogulp_size = nchan * npol**2 * self.grid_size * self.grid_size * 8
                self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size,buffer_factor=5)
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag,header=ohdr_str) as oseq:
                    iseq_spans = iseq.read(igulp_size)
                    while not self.iring.writing_ended():
                        reset_sequence = False

                        if self.profile:
                            spani = 0

                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time


                            if self.benchmark == True:
                                print(" ------------------ ")

                            ###### Correlator #######
                            ## Setup and load
                            idata = ispan.data_view(numpy.uint8).reshape(itshape)
                            ## Fix the type
                            tdata = bifrost.ndarray(shape=itshape, dtype='ci4', native=False, buffer=idata.ctypes.data)

                            if self.benchmark == True:
                                time1=time.time()
                            #tdata = tdata.transpose((0,1,3,2))

                            tdata = tdata.copy(space='cuda')
                            if self.benchmark == True:
                                time1a = time.time()
                                print("  Input copy time: %f" % (time1a-time1))

                            ## Unpack
                            #try:
                            #    udata = udata.reshape(*tdata.shape)
                            #    Unpack(tdata, udata)
                            #except NameError:
                            #    udata = bifrost.ndarray(shape=tdata.shape, dtype=numpy.complex64, space='cuda')
                            #    Unpack(tdata, udata)
                            udata = tdata
                            if self.benchmark == True:
                                time1b = time.time()
                            ### Phase
                            #bifrost.map('a(i,j,k,l) *= b(j,k,l)',
                            #            {'a':udata, 'b':gphases}, axis_names=('i','j','k','l'), shape=udata.shape)
                            if self.benchmark == True:
                                time1c = time.time()
                                print("  Unpack and phase-up time: %f" % (time1c-time1a))

                            ## Make sure we have a place to put the gridded data
                            # Gridded Antennas
                            try:
                                gdata = gdata.reshape(self.ntime_gulp,nchan,npol,self.grid_size,self.grid_size)
                                memset_array(gdata,0)
                            except NameError:
                                gdata = bifrost.zeros(shape=(self.ntime_gulp,nchan,npol,self.grid_size,self.grid_size),dtype=numpy.complex64, space='cuda')

                            ## Grid the Antennas
                            if self.benchmark == True:
                                timeg1 = time.time()


                            try:
                                bf_romein.execute(udata, gdata)
                            except NameError:
                                bf_romein = Romein()
                                bf_romein.init(self.locs, gphases, self.grid_size, polmajor=False)
                                bf_romein.execute(udata, gdata)
                            gdata = gdata.reshape(self.ntime_gulp*nchan*npol,self.grid_size,self.grid_size)
                            #gdata = self.LinAlgObj.matmul(1.0, udata, bfantgridmap, 0.0, gdata)
                            if self.benchmark == True:
                                timeg2 = time.time()
                                print("  Romein time: %f"%(timeg2 - timeg1))

                            #Inverse transform

                            if self.benchmark == True:
                                timefft1 = time.time()
                            try:

                                bf_fft.execute(gdata,gdata,inverse=True)
                            except NameError:

                                bf_fft = Fft()
                                bf_fft.init(gdata,gdata,axes=(1,2))
                                bf_fft.execute(gdata,gdata,inverse=True)
                            gdata = gdata.reshape(1,self.ntime_gulp,nchan,npol,self.grid_size,self.grid_size)
                            if self.benchmark == True:
                                timefft2 = time.time()
                                print("  FFT time: %f"%(timefft2 - timefft1))

                            #print ("Accum: %d"%accum,end='\n')
                            if self.newflag is True:
                                try:
                                    crosspol = crosspol.reshape(self.ntime_gulp,nchan,npol**2,self.grid_size,self.grid_size)
                                    accumulated_image = accumulated_image.reshape(1,nchan,npol**2,self.grid_size,self.grid_size)
                                    memset_array(crosspol, 0)
                                    memset_array(accumulated_image, 0)

                                except NameError:
                                    crosspol = bifrost.zeros(shape=(self.ntime_gulp,nchan,npol**2,self.grid_size,self.grid_size),
                                                             dtype=numpy.complex64, space='cuda')
                                    accumulated_image = bifrost.zeros(shape=(1,nchan,npol**2,self.grid_size,self.grid_size),
                                                                      dtype=numpy.complex64, space='cuda')
                                self.newflag=False



                            if self.remove_autocorrs == True:

                                ##Setup everything for the autocorrelation calculation.
                                try:
                                    # If one isn't allocated, then none of them are.
                                    autocorrs = autocorrs.reshape(self.ntime_gulp,nchan,nstand,npol**2)
                                    autocorr_g = autocorr_g.reshape(nchan*npol**2,self.grid_size,self.grid_size)
                                except NameError:
                                    autocorrs = bifrost.ndarray(shape=(self.ntime_gulp,nchan,nstand,npol**2),dtype=numpy.complex64, space='cuda')
                                    autocorrs_av = bifrost.zeros(shape=(1,nchan,nstand,npol**2), dtype=numpy.complex64, space='cuda')
                                    autocorr_g = bifrost.zeros(shape=(1,nchan,npol**2,self.grid_size,self.grid_size), dtype=numpy.complex64, space='cuda')
                                    autocorr_lo = bifrost.ndarray(numpy.ones(shape=(3,1,nchan,nstand,npol**2),dtype=numpy.int32)*self.grid_size/2,space='cuda')
                                    autocorr_il = bifrost.ndarray(numpy.ones(shape=(1,nchan,nstand,npol**2,self.ant_extent,self.ant_extent),dtype=numpy.complex64),space='cuda')


                                # Cross multiply to calculate autocorrs
                                bifrost.map('a(i,j,k,l) += (b(i,j,k,l/2) * b(i,j,k,l%2).conj())',
                                            {'a':autocorrs, 'b':udata,'t':self.ntime_gulp},
                                            axis_names=('i','j','k','l'),
                                            shape=(self.ntime_gulp,nchan,nstand,npol**2))

                            bifrost.map('a(i,j,p,k,l) += b(0,i,j,p/2,k,l)*b(0,i,j,p%2,k,l).conj()',
                                        {'a':crosspol, 'b':gdata},
                                        axis_names=('i','j', 'p', 'k', 'l'),
                                        shape=(self.ntime_gulp, nchan, npol**2, self.grid_size, self.grid_size))


                            # Increment
                            accum += 1e3 * self.ntime_gulp / CHAN_BW
                            
                            if accum >= self.accumulation_time:

                                bifrost.reduce(crosspol, accumulated_image, op='sum')
                                if self.remove_autocorrs == True:
                                    # Reduce along time axis.
                                    bifrost.reduce(autocorrs, autocorrs_av, op='sum')
                                    # Grid the autocorrelations.
                                    autocorr_g = autocorr_g.reshape(1,nchan,npol**2,self.grid_size, self.grid_size)
                                    try:
                                        bf_romein_autocorr.execute(autocorrs_av, autocorr_g)
                                    except NameError:
                                        bf_romein_autocorr = Romein()
                                        bf_romein_autocorr.init(autocorr_lo, autocorr_il, self.grid_size, polmajor=False)
                                        bf_romein_autocorr.execute(autocorrs_av, autocorr_g)
                                    autocorr_g = autocorr_g.reshape(1*nchan*npol**2,self.grid_size,self.grid_size)
                                    #autocorr_g = romein_float(autocorrs_av,autocorr_g,autocorr_il,autocorr_lx,autocorr_ly,autocorr_lz,self.ant_extent,self.grid_size,nstand,nchan*npol**2)
                                    #Inverse FFT
                                    try:
                                        autocorr_g = fft_shift_2d(autocorr_g, self.grid_size, nchan*npol**2)
                                        ac_fft.execute(autocorr_g,autocorr_g,inverse=True)
                                    except NameError:
                                         ac_fft = Fft()
                                         ac_fft.init(autocorr_g,autocorr_g,axes=(1,2))
                                         autocorr_g = fft_shift_2d(autocorr_g, self.grid_size, nchan*npol**2)
                                         ac_fft.execute(autocorr_g,autocorr_g,inverse=True)

                                    accumulated_image = accumulated_image.reshape(nchan,npol**2,self.grid_size, self.grid_size)
                                    autocorr_g = autocorr_g.reshape(nchan,npol**2,self.grid_size, self.grid_size)
                                    bifrost.map('a(i,j,k,l) -= b(i,j,k,l)',
                                                {'a':accumulated_image, 'b':autocorr_g},
                                                axis_names=('i','j','k','l'),
                                                shape=(nchan,npol**2,self.grid_size, self.grid_size))





                                curr_time = time.time()
                                process_time = curr_time - prev_time
                                prev_time = curr_time
                                
                                with oseq.reserve(ogulp_size) as ospan:
                                    odata = ospan.data_view(numpy.complex64).reshape(oshape)
                                    accumulated_image = accumulated_image.reshape(oshape)
                                    odata[...] = accumulated_image
                                    
                                curr_time = time.time()
                                reserve_time = curr_time - prev_time
                                prev_time = curr_time
                                
                                self.newflag = True
                                accum = 0

                                if self.remove_autocorrs == True:
                                    autocorr_g = autocorr_g.reshape(oshape)
                                    memset_array(autocorr_g,0)
                                    memset_array(autocorrs,0)
                                    memset_array(autocorrs_av,0)
                                    
                            else:
                                process_time = 0.0
                                reserve_time = 0.0

                            curr_time = time.time()
                            process_time += curr_time - prev_time
                            prev_time = curr_time
                            
                            #TODO: Autocorrs using Romein??
                            ## Output for gridded electric fields.
                            if self.benchmark == True:
                                time1h = time.time()
                            #gdata = gdata.reshape(self.ntime_gulp,nchan,2,self.grid_size,self.grid_size)
                            # image/autos, time, chan, pol, gridx, grid.
                            #accumulated_image = accumulated_image.reshape(oshape)

                            if self.benchmark == True:
                                time2=time.time()
                                print("-> GPU Time Taken: %f"%(time2-time1))

                                runtime_history.append(time2-time1)
                                print("-> Average GPU Time Taken: %f (%i samples)" % (1.0*sum(runtime_history)/len(runtime_history), len(runtime_history)))
                            if self.profile:
                                spani += 1
                                if spani >= 10:
                                    sys.exit()
                                    break

                            self.perf_proclog.update({'acquire_time': acquire_time,
                                                      'reserve_time': reserve_time,
                                                      'process_time': process_time,})

                        # Reset to move on to the next input sequence?
                        if not reset_sequence:
                            break


class TriggerOp(object):
    def __init__(self, log, iring, ints_per_analysis=1, threshold=6.0, elevation_limit=20.0, 
                 core=-1, gpu=-1, *args, **kwargs):
        self.log = log
        self.iring = iring
        self.ints_per_file = ints_per_analysis
        self.threshold = threshold
        self.elevation_limit = elevation_limit*numpy.pi/180.0

        self.core = core
        self.gpu = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")

        self.in_proclog.update( {'nring':1, 'ring0':self.iring.name})
        self.size_proclog.update({'nseq_per_gulp': 1})

        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def main(self):
        global TRIGGER_ACTIVE
        
        MAX_HISTORY = 10
        
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1,
                                  'core0': bifrost.affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        for iseq in self.iring.read(guarantee=True):
            ihdr = json.loads(iseq.header.tostring())
            fileid = 0
            
            self.sequence_proclog.update(ihdr)
            self.log.info('TriggerOp: Config - %s' % ihdr)

            cfreq             = ihdr['cfreq']
            nchan             = ihdr['nchan']
            npol              = ihdr['npol']
            grid_size_x       = ihdr['grid_size_x']
            grid_size_y       = ihdr['grid_size_y']
            grid_size         = max([grid_size_x, grid_size_y])
            sampling_length_x = ihdr['sampling_length_x']
            sampling_length_y = ihdr['sampling_length_y']
            sampling_length   = max([sampling_length_x, sampling_length_y])
            print("Channel no: %d, Polarisation no: %d, Grid no: %d, Sampling: %.3f"%(nchan,npol,grid_size,sampling_length))
            
            x, y = numpy.arange(grid_size_x), numpy.arange(grid_size_y)
            x, y = numpy.meshgrid(x, y)
            rho = numpy.sqrt((x-grid_size_x/2)**2 + (y-grid_size_y/2)**2)
            mask = numpy.where(rho <= grid_size*sampling_length*numpy.cos(self.elevation_limit), 
                               False, True)
            
            igulp_size = nchan * npol * grid_size_x * grid_size_y * 8
            ishape = (nchan,npol,grid_size_x,grid_size_y)
            image = []
            image_history = deque([], MAX_HISTORY)

            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            nints = 0
            
            for ispan in iseq_spans:
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                idata = ispan.data_view(numpy.complex64).reshape(ishape)
                itemp = idata.copy(space='cuda_host')
                image.append(itemp)
                nints += 1
                if nints >= self.ints_per_file:
                    image = numpy.fft.fftshift(image, axes=(3, 4))
                    image = image[:, :, :, ::-1, :]
                    ## NOTE:  This just uses the first polarization (XX) for now.
                    ##        In the future we probably want to use Stokes I (if
                    ##        possible) to beat the noise down a little.
                    image = image[:,:,0,:,:].real.sum(axis=0).sum(axis=0)
                    unix_time = (ihdr['time_tag'] / FS + ihdr['accumulation_time']
                                * 1e-3 * fileid * self.ints_per_file)
                    
                    if len(image_history) == MAX_HISTORY:
                        ## The transient detection is based on a differencing the
                        ## current image (image) with a moving average of the last
                        ## N images (image_background).  This is roughly like what
                        ## is done at LWA1/LWA-SV to find events in the LASI images.
                        image_background = numpy.median(image_history, axis=0)
                        image_diff = numpy.ma.array(image - image_background, mask=mask)
                        peak, mid, rms = image_diff.max(), image_diff.mean(), image_diff.std()
                        print('-->', peak, mid, rms, '@', (peak-mid)/rms)
                        if (peak-mid) > self.threshold*rms:
                            print("Trigger Set at %.3f with S/N %f" % (unix_time, (peak-mid)/rms,))
                            TRIGGER_ACTIVE.set()
                            
                    image_history.append( image )
                    image = []
                    nints = 0
                    fileid += 1

class SaveOp(object):
    def __init__(self, log, iring, filename, core=-1, gpu=-1, cpu=False,
                 profile=False, ints_per_file=1, out_dir='', triggering=False, *args, **kwargs):
        self.log = log
        self.iring = iring
        self.filename = filename
        self.ints_per_file = ints_per_file
        self.out_dir = out_dir
        self.triggering = triggering

        # TODO: Validate ntime_gulp vs accumulation_time
        self.core = core
        self.gpu = gpu
        self.cpu = cpu
        self.profile = profile

        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")

        self.in_proclog.update( {'nring':1, 'ring0':self.iring.name})
        self.size_proclog.update({'nseq_per_gulp': 1})

        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def main(self):
        global TRIGGER_ACTIVE
        
        MAX_HISTORY = 5
        
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1,
                                  'core0': bifrost.affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})

        image_history = deque([], MAX_HISTORY)
        
        for iseq in self.iring.read(guarantee=True):
            ihdr = json.loads(iseq.header.tostring())
            fileid = 0
            
            self.sequence_proclog.update(ihdr)
            self.log.info('SaveOp: Config - %s' % ihdr)

            cfreq = ihdr['cfreq']
            nchan = ihdr['nchan']
            npol = ihdr['npol']
            grid_size_x       = ihdr['grid_size_x']
            grid_size_y       = ihdr['grid_size_y']
            grid_size         = max([grid_size_x, grid_size_y])
            print("Channel no: %d, Polarisation no: %d, Grid no: %d"%(nchan,npol,grid_size))

            igulp_size = nchan * npol * grid_size_x * grid_size_y * 8
            ishape = (nchan,npol,grid_size_x,grid_size_y)
            image = []
            
            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            nints = 0
            
            dump_counter = 0

            if self.profile:
                spani = 0

            for ispan in iseq_spans:
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                idata = ispan.data_view(numpy.complex64).reshape(ishape)
                itemp = idata.copy(space='cuda_host')
                image.append(itemp)
                nints += 1
                if nints >= self.ints_per_file:
                    image = numpy.fft.fftshift(image, axes=(3, 4))
                    image = image[:, :, :, ::-1, :]
                    unix_time = (ihdr['time_tag'] / FS + ihdr['accumulation_time']
                                * 1e-3 * fileid * self.ints_per_file)
                    image_nums = numpy.arange(fileid * self.ints_per_file, (fileid + 1) * self.ints_per_file)
                    filename = os.path.join(self.out_dir, 'EPIC_{0:3f}_{1:0.3f}MHz.npz'.format(unix_time, cfreq/1e6))
                    
                    image_history.append( (filename, image, ihdr, image_nums) )
                    
                    if TRIGGER_ACTIVE.is_set() or not self.triggering:
                        if dump_counter == 0:
                            dump_counter = 20 + MAX_HISTORY
                        elif dump_counter == 1:
                            TRIGGER_ACTIVE.clear()
                        cfilename, cimage, chdr, cimage_nums = image_history.popleft()
                        numpy.savez(cfilename, image=cimage, hdr=chdr, image_nums=cimage_nums)
                        print("SaveOp - Image Saved")
                        dump_counter -= 1
                        
                    image = []
                    nints = 0
                    fileid += 1
                    
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time,
                                          'reserve_time': -1,
                                          'process_time': process_time,})
                if self.profile:
                    spani += 1
                    if spani >= 10:
                        sys.exit()
                        break

class SaveFFTOp(object):
    def __init__(self, log, iring, filename, ntime_gulp=2500, core=-1,*args, **kwargs):
        self.log = log
        self.iring = iring
        self.filename = filename
        self.core = core
        self.ntime_gulp = ntime_gulp

    def main(self):
        #bifrost.affinity.set_core(self.core)

        for iseq in self.iring.read(guarantee=True):

            ihdr = json.loads(iseq.header.tostring())
            nchan = ihdr['nchan']
            nstand = ihdr['nstand']
            npol = ihdr['npol']


            igulp_size = self.ntime_gulp*1*nstand*npol * 2         # ci8
            ishape = (self.ntime_gulp/nchan,nchan,nstand,npol,2)

            iseq_spans = iseq.read(igulp_size)

            while not self.iring.writing_ended():

                for ispan in iseq_spans:
                    if ispan.size < igulp_size:
                        continue

                    idata = ispan.data_view(numpy.int8)

                    idata = idata.reshape(ishape)
                    idata = bifrost.ndarray(shape=ishape, dtype='ci4', native=False, buffer=idata.ctypes.data)
                    print(numpy.shape(idata))
                    numpy.savez(self.filename + "asdasd.npy",data=idata)
                    print("Wrote to disk")
            break
        print("Save F-Engine Spectra.. done")


def main():

    # Main Input: UDP Broadcast RX from F-Engine?

    parser = argparse.ArgumentParser(description='EPIC Correlator', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group1 = parser.add_argument_group('Online Data Processing')
    group1.add_argument('--addr', type=str, default = "p5p1", help= 'F-Engine UDP Stream Address')
    group1.add_argument('--port', type=int, default = 4015, help= 'F-Engine UDP Stream Port')
    group1.add_argument('--utcstart', type=str, default = '1970_1_1T0_0_0', help= 'F-Engine UDP Stream Start Time')
    group2 = parser.add_argument_group('Offline Data Processing')
    group2.add_argument('--offline', action='store_true', help = 'Load TBN data from Disk')
    group2.add_argument('--tbnfile', type=str, help = 'TBN Data Path')
    group2.add_argument('--tbffile', type=str, help = 'TBF Data Path')
    group3 = parser.add_argument_group('Processing Options')
    group3.add_argument('--imagesize', type=int, default = 64, help = '1-D Image Size')
    group3.add_argument('--imageres', type=float, default = 1.79057, help = 'Image pixel size in degrees')
    group3.add_argument('--nts',type=int, default = 1000, help= 'Number of timestamps per span')
    group3.add_argument('--accumulate',type=int, default = 1000, help='How many milliseconds to accumulate an image over')
    group3.add_argument('--channels',type=int, default=1, help='How many channels to produce')
    group3.add_argument('--singlepol', action='store_true', help = 'Process only X pol. in online mode')
    group3.add_argument('--removeautocorrs', action='store_true', help = 'Removes Autocorrelations')
    group4 = parser.add_argument_group('Output')
    group4.add_argument('--ints_per_file', type=int, default=1, help='Number of integrations per output FITS file.')
    group4.add_argument('--out_dir', type=str, default='.', help='Directory for output files. Default is current directory.')
    group5 = parser.add_argument_group('Self Triggering')
    group5.add_argument('--triggering', action='store_true', help='Enable self-triggering')
    group5.add_argument('--threshold', type=float, default=8.0, help='Self-triggering threshold')
    group5.add_argument('--elevation-limit', type=float, default=20.0, help='Self-trigger minimum elevation limit in degrees')
    group6 = parser.add_argument_group('Benchmarking')
    group6.add_argument('--benchmark', action='store_true',help = 'benchmark gridder')
    group6.add_argument('--profile', action='store_true', help = 'Run cProfile on ALL threads. Produces trace for each individual thread')
    
    args = parser.parse_args()
    # Logging Setup
    # TODO: Set this up properly
    if args.profile:
        enable_thread_profiling()

    if not os.path.isdir(args.out_dir):
        print('Output directory does not exist. Defaulting to current directory.')
        args.out_dir = '.'


    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    logHandler = logging.StreamHandler(sys.stdout)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)
    log.setLevel(logging.DEBUG)

    # Setup the cores and GPUs to use
    cores = [0, 2, 3, 4, 5, 6, 7]
    gpus  = [0, 0, 0, 0, 0, 0, 0]

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


    # Setup Rings

    fcapture_ring = Ring(name="capture",space="cuda_host")
    fdomain_ring = Ring(name="fengine", space="cuda_host")
    transpose_ring = Ring(name="transpose", space="cuda_host")
    gridandfft_ring = Ring(name="gridandfft", space="cuda")
    image_ring = Ring(name="image", space="system")


    # Setup Antennas
    ## TODO: Some sort of switch for other stations?

    lwasv_antennas = lwasv.getAntennas()
    lwasv_stands = lwasv.getStands()
    
    # Setup threads

    if args.offline:
        if args.tbnfile is not None:
            ops.append(TBNOfflineCaptureOp(log, fcapture_ring, args.tbnfile,
                                           core=cores.pop(0), profile=args.profile))
            ops.append(FDomainOp(log, fcapture_ring, fdomain_ring, ntime_gulp=args.nts,
                                 nchan_out=args.channels, core=cores.pop(0), gpu=gpus.pop(0),
                                 profile=args.profile))
        elif args.tbffile is not None:
            ops.append(TBFOfflineCaptureOp(log, fcapture_ring, args.tbffile,
                                           core=cores.pop(0), profile=args.profile))
            ops.append(DecimationOp(log, fcapture_ring, fdomain_ring, ntime_gulp=args.nts,
                                nchan_out=args.channels, npol_out=1 if args.singlepol else 2,
                                core=cores.pop(0)))
        else:
            raise parser.error("--offline set but no file provided via --tbnfile or --tbffile")
    else:
        # It would be great is we could pull this from ADP MCS...
        utc_start_dt = datetime.datetime.strptime(args.utcstart, "%Y_%m_%dT%H_%M_%S")

        # Note: Capture uses Bifrost address+socket objects, while output uses
        #         plain Python address+socket objects.
        iaddr = BF_Address(args.addr, args.port)
        isock = BF_UDPSocket()
        isock.bind(iaddr)
        isock.timeout = 0.5

        ops.append(FEngineCaptureOp(log, fmt="chips", sock=isock, ring=fcapture_ring,
                                    nsrc=16, src0=0, max_payload_size=9000,
                                    buffer_ntime=args.nts, slot_ntime=25000, core=cores.pop(0),
                                    utc_start=utc_start_dt))
        ops.append(DecimationOp(log, fcapture_ring, fdomain_ring, ntime_gulp=args.nts,
                                nchan_out=args.channels, npol_out=1 if args.singlepol else 2,
                                core=cores.pop(0)))

    ##ops.append(TransposeOp(log, fdomain_ring, transpose_ring, ntime_gulp=args.nts,
    ##                            core=cores.pop(0)))
    ops.append(MOFFCorrelatorOp(log, fdomain_ring, gridandfft_ring, lwasv_antennas,
                                args.imagesize, args.imageres, ntime_gulp=args.nts, 
                                accumulation_time=args.accumulate, 
                                remove_autocorrs=args.removeautocorrs,
                                core=cores.pop(0), gpu=gpus.pop(0),benchmark=args.benchmark,
                                profile=args.profile))
    if args.triggering:
        ops.append(TriggerOp(log, gridandfft_ring, core=cores.pop(0), gpu=gpus.pop(0), 
                             ints_per_analysis=args.ints_per_file, threshold=args.threshold, 
                             elevation_limit=max([0.0, args.elevation_limit])))
    ops.append(SaveOp(log, gridandfft_ring, "EPIC_", out_dir=args.out_dir,
                         core=cores.pop(0), gpu=gpus.pop(0), cpu=False,
                         ints_per_file=args.ints_per_file, triggering=args.triggering, 
                         profile=args.profile))

    threads= [threading.Thread(target=op.main) for op in ops]

    # Go!

    for thread in threads:
        thread.daemon = False
        thread.start()

    while not shutdown_event.is_set():
        # Keep threads alive -- if reader is still alive, prevent timeout signal from executing
        if threads[0].is_alive():
            signal.pause()
        else:
            break

    # Wait for threads to finish

    for thread in threads:
        thread.join()

    if args.profile:
        stats = get_thread_stats()
        stats.print_stats()
        stats.dump_stats("EPIC_stats.prof")

    log.info("Done")


if __name__ == "__main__":
    main()
