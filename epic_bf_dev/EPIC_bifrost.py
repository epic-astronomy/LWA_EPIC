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
import matplotlib.pyplot as plt

## Profiling Includes
import cProfile
import pstats


## Bifrost Includes
from bifrost.address import Address as BF_Address
from bifrost.udp_socket import UDPSocket as BF_UDPSocket
from bifrost.udp_capture import UDPCapture as BF_UDPCapture
from bifrost.udp_transmit import UDPTransmit as BF_UDPTransmit
from bifrost.ring import Ring
from bifrost.unpack import unpack as Unpack
from bifrost.quantize import quantize as Quantize
from bifrost.reduce import reduce as Reduce 
from bifrost.proclog import ProcLog
from bifrost.libbifrost import bf
from bifrost.fft import Fft
from bifrost.romein import romein_float
import bifrost
import bifrost.affinity
from bifrost.ndarray import memset_array
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()

## LWA Software Library Includes
from lsl.common.constants import c as speedOfLight
from lsl.writer import fitsidi
from lsl.reader.ldp import TBNFile
from lsl.common.stations import lwasv, parseSSMIF

## TODO: Move this to argparse.

CHAN_BW = 25000	# Hz
ANTENNAS = lwasv.getAntennas()
DUMP_TIME = 5	# s

# Build map of where the LWA-SV Stations are.
## TODO: Make this non-SV specific
LOCATIONS = lwasv.getStands()

locations = numpy.empty(shape=(0,3))

for stand in LOCATIONS:
    locations = numpy.vstack((locations,[stand[0],stand[1],stand[2]]))
locations = numpy.delete(locations, list(range(0,locations.shape[0],2)),axis=0)
# Mask out the outrigger
locations[255,:] = 0
locations[:,1] = locations[:,1]-10.0

#print(locations)
#locations = locations[0:255,:]
#locations = locations[0:255,:]
print ("Max X: %f, Max Y: %f" % (numpy.nanmax(locations[:,0]), numpy.nanmax(locations[:,1])))
print ("Min X: %f, Min Y: %f" % (numpy.nanmin(locations[:,0]), numpy.nanmin(locations[:,1])))
#print (numpy.shape(locations))

#for location in locations:
#    print location
RESOLUTION = 2.0
loc_range_x = numpy.nanmax(locations[:,0]) - numpy.nanmin(locations[:,0])
loc_range_y = numpy.nanmax(locations[:,1]) - numpy.nanmin(locations[:,1])
loc_range = max([loc_range_x, loc_range_y])
#print (loc_range_x, loc_range_y, loc_range)
GRID_SIZE = int(numpy.power(2,numpy.ceil(numpy.log(loc_range/RESOLUTION)/numpy.log(2))+0))
locations = locations + GRID_SIZE
locations = locations/RESOLUTION
#print(GRID_SIZE)

#plt.plot(locations[:,0],locations[:,1],'x')
#plt.show()


#### Profiling #####

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




class OfflineCaptureOp(object):
    def __init__(self, log, oring, filename, profile=False, core=-1):
        self.log = log
        self.oring = oring
        self.filename = filename
        self.core = core
        self.profile = profile

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
                if self.profile:
                    spani = 0
                while not self.shutdown_event.is_set():
                    ## Get the current section to use
                    try:
                        _, _, next_data = idf.read(0.1, timeInSamples=True)
                    except Exception as e:
                        print("FillerOp: Error - '%s'" % str(e))
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
        print("FillerOp - Done")


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

                

## For when we don't need to care about doing the F-Engine ourself.
## TODO: Implement this come implementation time...
class FEngineCaptureOp(object):
    '''
    Receives Fourier Spectra from LWA FPGA
    '''
    def __init__(self, log, *args, **kwargs):
        self.log = log
        self.args = args
        self.kwargs = kwargs

        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def seq_callback(self):
        #What do I need to do to the data?
        pass

    def main(self):
        seq_callback = bf.BFudpcapture_sequence_callback(self.seq_callback)
        with UDPCapture(*self.args,
                        sequence_callback=seq_callback,
                        **self.kwargs) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
        del capture

class CalibrationOp(object):
    def __init__(self, log, iring, oring, *args, **kwargs):
        pass

class MOFFCorrelatorOp(object):
    def __init__(self, log, iring, oring, ntime_gulp=2500, core=-1, gpu=-1, remove_autocorrs = False, benchmark=False, profile=False, *args, **kwargs):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp

        self.core = core
        self.gpu = gpu
        self.remove_autocorrs = remove_autocorrs
        self.benchmark = benchmark
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

        self.antgridmap = bifrost.ndarray(numpy.ones(shape=(1,1),dtype=numpy.complex64),space='cuda')
        self.antgridmap = self.antgridmap.copy(space='cuda',order='C')
        
        if self.remove_autocorrs == True:
            self.autocorrmap = self.make_autocorr_grid(self.antgridmap,GRID_SIZE)
            
        if self.gpu != -1:
            BFSetGPU(self.gpu)

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
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=True):
                ihdr = json.loads(iseq.header.tostring())
                self.sequence_proclog.update(ihdr)
                print('MOFFCorrelatorOp: Config - %s' % ihdr)
                nchan = ihdr['nchan']
                nstand = ihdr['nstand']
                npol = ihdr['npol']
                locations_x = bifrost.ndarray(numpy.tile(locations[:,0],self.ntime_gulp*nchan*npol).astype(numpy.int32),space='cuda')
                locations_x = locations_x.reshape(self.ntime_gulp*nchan*npol,nstand)
                locations_x = locations_x.copy(space='cuda',order='C')
                locations_y = bifrost.ndarray(numpy.tile(locations[:,1],self.ntime_gulp*nchan*npol).astype(numpy.int32),space='cuda')
                locations_y = locations_y.reshape(self.ntime_gulp*nchan*npol,nstand)
                locations_y = locations_y.copy(space='cuda',order='C')
                locations_z = bifrost.ndarray(numpy.zeros(shape=(self.ntime_gulp*nchan*npol*nstand)).astype(numpy.int32),space='cuda')
                locations_z.reshape(self.ntime_gulp*nchan*npol,nstand)
                print(locations_x)
                print(locations_y)
                print(locations_z.shape)
                
                igulp_size = self.ntime_gulp * nchan * nstand * npol * 1 # ci4
                ishape = (self.ntime_gulp,nchan,nstand,npol)
                itshape = (self.ntime_gulp,nchan,npol,nstand)

                ohdr = ihdr.copy()
                ohdr['nbit'] = 64
                if self.remove_autocorrs == True:
                    ohdr['npol'] = 4
                else:
                    ohdr['npol'] = 2
                ohdr['nchan'] = nchan
                ohdr_str = json.dumps(ohdr)
                
                #Setup the phasing terms for zenith
                phases = numpy.zeros(itshape, dtype=numpy.complex64)
                freq = numpy.fft.fftfreq(nchan, d=1.0/ihdr['bw']) + ihdr['cfreq']
                for i in xrange(nstand):
                    ## X
                    a = ANTENNAS[2*i + 0]  
                    delay = a.cable.delay(freq) - a.stand.z / speedOfLight
                    phases[:,:,0,i] = numpy.exp(2j*numpy.pi*freq*delay)
                    phases[:,:,0,i] /= numpy.sqrt(a.cable.gain(freq))
                    ## Y
                    a = ANTENNAS[2*i + 1]
                    delay = a.cable.delay(freq) - a.stand.z / speedOfLight
                    phases[:,:,1,i] = numpy.exp(2j*numpy.pi*freq*delay)
                    phases[:,:,1,i] /= numpy.sqrt(a.cable.gain(freq))
                    
                # Four polarisations as I need autocorrelations for XX,XY,YX,YY
                # Axes are as follows (Image/Autocorrs, Time, Channel, Polarisation, Image, Image)
                if self.remove_autocorrs == True:
                    oshape = (2,self.ntime_gulp,nchan,4,GRID_SIZE,GRID_SIZE)
                    ogulp_size = 2 * self.ntime_gulp * nchan * 4 * GRID_SIZE * GRID_SIZE * 8 #Complex64
                else:
                    oshape = (1,self.ntime_gulp,nchan,npol,GRID_SIZE,GRID_SIZE)
                    ogulp_size = self.ntime_gulp * nchan * npol * GRID_SIZE * GRID_SIZE * 8
                self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size,buffer_factor=1)
                
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag,header=ohdr_str) as oseq:
                    iseq_spans = iseq.read(igulp_size)
                    while not self.iring.writing_ended():

                        if self.profile:
                            spani = 0
                        
                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            with oseq.reserve(ogulp_size) as ospan:
                                if self.benchmark == True:
                                    print(" ------------------ ")
                                curr_time = time.time()
                                reserve_time = curr_time - prev_time
                                prev_time = curr_time
                                
                                ###### Correlator #######
                                ## Setup and load
                                idata = ispan.data_view(numpy.uint8).reshape(ishape)
                                idata = idata.transpose((0,1,3,2))
                                idata = idata.copy(order='C')
                                ## Fix the type
                                tdata = bifrost.ndarray(shape=itshape, dtype='ci4', native=False, buffer=idata.ctypes.data)
                                odata = ospan.data_view(numpy.complex64).reshape(oshape)
                                if self.benchmark == True:
                                    time1=time.time()
                                #tdata = tdata.transpose((0,1,3,2))
                                
                                tdata = tdata.copy(space='cuda')
                                if self.benchmark == True:
                                    time1a = time.time()
                                    print("  Input copy time: %f" % (time1a-time1))
                                
                                ## Unpack
                                try:
                                    udata = udata.reshape(*tdata.shape)
                                    Unpack(tdata, udata)
                                except NameError:
                                    udata = bifrost.ndarray(shape=tdata.shape, dtype=numpy.complex64, space='cuda')
                                    Unpack(tdata, udata)
                                if self.benchmark == True:
                                    time1b = time.time()
                                ## Phase
                                try:
                                    bifrost.map('a(i,j,k,l) *= b(i,j,k,l)', {'a':udata, 'b':gphases}, axis_names=('i','j','k','l'), shape=udata.shape)                    
                                except NameError:
                                    phases = bifrost.ndarray(phases)
                                    gphases = phases.copy(space='cuda')
                                    bifrost.map('a(i,j,k,l) *= b(i,j,k,l)', {'a':udata, 'b':gphases}, axis_names=('i','j','k','l'), shape=udata.shape)
                                    #udata = udata.transpose((0,1,3,2))
                                    #Transpose
                                if self.benchmark == True:
                                    time1c = time.time()
                                    print("  Phase-up time: %f" % (time1c-time1b))
                                
                                #Combine stand and pols into standpols
                                
                                ## Make sure we have a place to put the data
                                # Gridded Antennas
                                try:
                                    gdata = gdata.reshape(self.ntime_gulp*nchan*npol,GRID_SIZE,GRID_SIZE)
                                    gdata = gdata.copy(space='cuda',order='C')
                                except NameError:
                                    gdata = bifrost.ndarray(shape=(self.ntime_gulp*nchan*npol,GRID_SIZE,GRID_SIZE),dtype=numpy.complex64, space='cuda')
                                    gdata = gdata.copy(space='cuda',order='C')
                                    
                                ## Grid the Antennas
                                if self.benchmark == True:
                                    timeg1 = time.time()
                                
                                gdata = romein_float(udata,gdata,self.antgridmap,locations_x,locations_y,locations_z,3,GRID_SIZE,256,self.ntime_gulp*npol*nchan)
                                #gdata = self.LinAlgObj.matmul(1.0, udata, bfantgridmap, 0.0, gdata)
                                if self.benchmark == True:
                                    timeg2 = time.time()
                                    print("  LinAlg time: %f"%(timeg2 - timeg1))
                                
                                #Inverse transform
                               
                                if self.benchmark == True:
                                    timefft1 = time.time()
                                try:
                                    fdata = fdata.reshape(self.ntime_gulp*nchan*2,GRID_SIZE,GRID_SIZE)
                                    bf_fft.execute(gdata,fdata,inverse=True)
                                except NameError:
                                    fdata = bifrost.ndarray(shape=gdata.shape, dtype=numpy.complex64, space='cuda')
                                    bf_fft = Fft()
                                    bf_fft.init(gdata,fdata,axes=(1,2))
                                    bf_fft.execute(gdata,fdata,inverse=True)
                                fdata = fdata.reshape(self.ntime_gulp,nchan,2,GRID_SIZE,GRID_SIZE)
                                if self.benchmark == True:
                                    timefft2 = time.time()
                                    print("  FFT time: %f"%(timefft2 - timefft1))

                                
                                #TODO: Autocorrs using Romein??
                                ## Output for gridded electric fields.
                                if self.benchmark == True:
                                    time1h = time.time()
                                #gdata = gdata.reshape(self.ntime_gulp,nchan,2,GRID_SIZE,GRID_SIZE)
                                odata[0,:,:,0:2,:,:] = fdata
                                if self.benchmark == True:
                                    time1i = time.time()
                                    print("  Shift-n-save time: %f" % (time1i-time1h))
                                
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
                                
                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                      'reserve_time': reserve_time, 
                                                      'process_time': process_time,})

                                
class ImagingOp(object):
    def __init__(self, log, iring, filename, ntime_gulp=100, accumulation_time=1, core=-1, gpu=-1, cpu=False, remove_autocorrs = False, profile=False, *args, **kwargs):
        self.log = log
        self.iring = iring
        self.filename = filename
        self.ntime_gulp= ntime_gulp
        self.accumulation_time = accumulation_time
        self.core = core
        self.gpu = gpu
        self.cpu = cpu
        self.remove_autocorrs = remove_autocorrs
        self.accumulated_image = None
        self.newflag = True
        self.profile = profile
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update( {'nring':1, 'ring0':self.iring.name})
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
        
        accum = 0
        fileid = 1
        for iseq in self.iring.read(guarantee=True):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            print('ImagingOp: Config - %s' % ihdr)
            
            nchan = ihdr['nchan']
            npol = ihdr['npol']
            print("Channel no: %d, Polarisation no: %d"%(nchan,npol))

            if self.remove_autocorrs == True:
                igulp_size = 2 * self.ntime_gulp * nchan * npol * GRID_SIZE * GRID_SIZE * 8
                ishape = (2,self.ntime_gulp,nchan,npol,GRID_SIZE,GRID_SIZE)
            else:
                igulp_size = self.ntime_gulp * nchan * npol * GRID_SIZE * GRID_SIZE * 8
                ishape = (1,self.ntime_gulp,nchan,npol,GRID_SIZE,GRID_SIZE)
                
            crosspol = bifrost.ndarray(shape=(self.ntime_gulp,nchan,npol**2,GRID_SIZE,GRID_SIZE), dtype=numpy.complex64, space='cuda')
            accumulated_image = bifrost.ndarray(shape=(1,nchan,npol**2,GRID_SIZE,GRID_SIZE), dtype=numpy.complex64, space='cuda')
            
            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            while not self.iring.writing_ended():
                if self.profile:
                    spani = 0
                for ispan in iseq_spans:
                    if ispan.size < igulp_size:
                        continue # Ignore final gulp
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    
                    idata = ispan.data_view(numpy.complex64).reshape(ishape)
                    if self.cpu:
                        
                        idata = idata.copy(space="system")
                        idata = idata.astype(numpy.complex128)
                        #Square
                        
                        xdata, ydata = idata[0,:,:,0,:,:], idata[0,:,:,1,:,:]
                        xxdata = numpy.abs(xdata*xdata.conj())
                        xydata = numpy.abs(xdata*ydata.conj()) \
                                * numpy.where( numpy.angle(xdata*ydata.conj()) > 0, 1, -1 )
                        yxdata = numpy.abs(ydata*xdata.conj()) \
                                * numpy.where( numpy.angle(ydata*xdata.conj()) > 0, 1, -1 )
                        yydata = numpy.abs(ydata*ydata.conj())
                        

                        for ti in numpy.arange(0,self.ntime_gulp):
                            #print ("Accum: %d"%accum,end='\n')
                            if self.newflag is True:
                            
                                self.accumulated_image = numpy.zeros(shape=(nchan,npol**2,GRID_SIZE,GRID_SIZE),dtype=numpy.complex128)
                                self.newflag=False
                                
                            #Accumulate
                            #Subtract auto-correlations.
                            if self.remove_autocorrs == True:
                                self.accumulated_image[:,0,:,:] += (xxdata[ti,:,:,:] - numpy.abs(idata[1,ti,:,0,:,:]))
                                self.accumulated_image[:,1,:,:] += (xydata[ti,:,:,:] - numpy.abs(idata[1,ti,:,1,:,:]))
                                self.accumulated_image[:,2,:,:] += (yxdata[ti,:,:,:] - numpy.abs(idata[1,ti,:,2,:,:]))
                                self.accumulated_image[:,3,:,:] += (yydata[ti,:,:,:] - numpy.abs(idata[1,ti,:,3,:,:]))
                            else:
                                self.accumulated_image[:,0,:,:] += (xxdata[ti,:,:,:])
                                self.accumulated_image[:,1,:,:] += (xydata[ti,:,:,:])
                                self.accumulated_image[:,2,:,:] += (yxdata[ti,:,:,:])
                                self.accumulated_image[:,3,:,:] += (yydata[ti,:,:,:])
                                
                            # Increment
                            accum += 1
                    else:
                        #print ("Accum: %d"%accum,end='\n')
                        if self.newflag is True:
                            memset_array(crosspol, 0)
                            self.newflag=False
                                
                        #Accumulate
                        #Subtract auto-correlations.
                        if self.remove_autocorrs == True:
                            bifrost.map('a(i,j,p,k,l) += b(0,i,j,p/2,k,l)*b(0,i,j,p%2,k,l).conj() - b(1,i,j,p,k,l)', 
                                        {'a':crosspol, 'b':idata}, 
                                        axis_names=('i','j', 'p', 'k', 'l'), 
                                        shape=(self.ntime_gulp, nchan, 4, GRID_SIZE, GRID_SIZE))
                        else:
                            bifrost.map('a(i,j,p,k,l) += b(0,i,j,p/2,k,l)*b(0,i,j,p%2,k,l).conj()', 
                                        {'a':crosspol, 'b':idata}, 
                                        axis_names=('i','j', 'p', 'k', 'l'), 
                                        shape=(self.ntime_gulp, nchan, 4, GRID_SIZE, GRID_SIZE))
                                        
                        # Increment
                        accum += self.ntime_gulp
                            
                    #Save and output
                    if accum >= self.accumulation_time:
                        if self.cpu == False:
                            bifrost.reduce(crosspol, accumulated_image, op='sum')
                            image = accumulated_image.copy(space='cuda_host')
                            image = image.reshape(nchan,npol**2,GRID_SIZE,GRID_SIZE)
                        else:
                            image = self.accumulated_image
                        image = numpy.fft.fftshift(numpy.abs(image), axes=(2,3))
                        image = numpy.transpose(image, (0,1,3,2))
                        
                        accum = 0
                        self.newflag = True
                        fig = plt.figure(fileid)
                        for i in xrange(4):
                            ax = fig.add_subplot(2, 2, i+1)
                            im = ax.imshow(image[0,i,:,:])
                            fig.colorbar(im,orientation='vertical')
                        #plt.imshow(numpy.real(self.accumulated_image[0,0,:,:].T))
                        plt.savefig("ImagingOP-%04i.png"%(fileid))
                        fileid += 1
                        print("ImagingOP - Image Saved")
                        
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

## TODO:                                 
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
            
            
            igulp_size = self.ntime_gulp*1*nstand*npol * 2		# ci8
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

    # Main Input: UDP Broadcast RX from F-Engine??

    parser = argparse.ArgumentParser(description='EPIC Correlator')
    parser.add_argument('-a', '--addr', type=str, help= 'F-Engine UDP Stream Address')
    parser.add_argument('-p', '--port', type=int, help= 'F-Engine UDP Stream Port')
    parser.add_argument('-o', '--offline', action='store_true', help = 'Load TBN data from Disk')
    parser.add_argument('-f', '--tbnfile', type=str, help = 'TBN Data Path')
    parser.add_argument('-t', '--nts',type=int, default = 1000, help= 'Number of timestamps per span.')
    parser.add_argument('-u', '--accumulate',type=int, default = 1000, help='How many milliseconds to accumulate an image over.')
    parser.add_argument('-n', '--channels',type=int, default=1, help='How many channels to produce.')
    parser.add_argument('-r', '--removeautocorrs',action='store_true', help = 'Removes Autocorrelations')
    parser.add_argument('-b', '--benchmark', action='store_true',help = 'benchmark gridder')
    parser.add_argument('-c', '--profile', action='store_true', help = 'Run cProfile on ALL threads. Produces trace for each individual thread.')

    args = parser.parse_args()
    # Logging Setup
    # TODO: Set this up properly
    if args.profile:
        enable_thread_profiling()
    
    
    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    logHandler = logging.StreamHandler(sys.stdout)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)
    log.setLevel(logging.DEBUG)
    
    # Setup the cores and GPUs to use
    cores = [0, 1, 2, 3, 4, 5, 6, 7]
    gpus  = [0, 0, 0, 0, 0, 0, 0, 0]
        
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


    if args.offline:
        fdomain_ring = Ring(name="fengine", space="cuda_host")
        fcapture_ring = Ring(name="capture",space="cuda_host")
    gridandfft_ring = Ring(name="gridandfft", space="cuda")
    image_ring = Ring(name="image", space="system")

    
    ops.append(OfflineCaptureOp(log, fcapture_ring,args.tbnfile,profile=args.profile))
    ops.append(FDomainOp(log, fcapture_ring, fdomain_ring, ntime_gulp=args.nts, nchan_out=args.channels, gpu=gpus.pop(0), profile=args.profile))
    ops.append(MOFFCorrelatorOp(log, fdomain_ring, gridandfft_ring, ntime_gulp=args.nts, remove_autocorrs=args.removeautocorrs, gpu=gpus.pop(0),benchmark=args.benchmark, profile=args.profile))
    ops.append(ImagingOp(log, gridandfft_ring, "EPIC_", ntime_gulp=args.nts, accumulation_time=args.accumulate, remove_autocorrs=args.removeautocorrs, gpu=gpus.pop(0), cpu=False, profile=args.profile))

    threads= [threading.Thread(target=op.main) for op in ops]

    for thread in threads:
        thread.daemon = False
        thread.start()

    while not shutdown_event.is_set():
        if threads[0].is_alive():
            signal.pause()
        else:
            break
    for thread in threads:
        thread.join()

    if args.profile:
        stats = get_thread_stats()
        stats.print_stats()
        stats.dump_stats("EPIC_stats.prof")
        
    log.info("Done")


    

if __name__ == "__main__":
    main()
    
    
