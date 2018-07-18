#!/usr/bin/env python
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

from scipy.fftpack import fft

import matplotlib.pyplot as plt

from bifrost.address import Address as BF_Address
from bifrost.udp_socket import UDPSocket as BF_UDPSocket
from bifrost.udp_capture import UDPCapture as BF_UDPCapture
from bifrost.udp_transmit import UDPTransmit as BF_UDPTransmit
from bifrost.ring import Ring
from bifrost.unpack import unpack as Unpack
from bifrost.quantize import quantize as Quantize
from bifrost.proclog import ProcLog
from bifrost.libbifrost import bf
from bifrost.linalg import LinAlg
from bifrost.fft import Fft

import bifrost
import bifrost.affinity



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
locations[255,:] = numpy.nan
#locations = locations[0:255,:]
#locations = locations[0:255,:]
print ("Max X: %f, Max Y: %f" % (numpy.nanmax(locations[:,0]), numpy.nanmax(locations[:,1])))
print ("Min X: %f, Min Y: %f" % (numpy.nanmin(locations[:,0]), numpy.nanmin(locations[:,1])))
print (numpy.shape(locations))

#for location in locations:
#    print location
RESOLUTION = 1.0
GRID_SIZE = int(numpy.power(2,numpy.ceil(numpy.log(numpy.max(numpy.abs(locations[:255,:]))/RESOLUTION)/numpy.log(2))+1))
print(GRID_SIZE)


#plt.plot(locations[:,0],locations[:,1],'x')
#plt.show()

## Antenna Illumination Pattern params:
# For now just use a square top-hat function..
## TODO: Do this properly...
Xmin= -1.5
Xmax= 1.5
Ymin= -1.5
Ymax= 1.5




class OfflineCaptureOp(object):
    def __init__(self, log, oring, filename, core=-1):
        self.log = log
        self.oring = oring
        self.filename = filename
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
        #bifrost.affinity.set_core(self.core)

        idf = TBNFile(self.filename)
	cfreq = idf.getInfo('freq1')
	srate = idf.getInfo('sampleRate')
	tInt, tStart, data = idf.read(0.1, timeInSamples=True)
	idf.reset()
	
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
		while not self.shutdown_event.is_set():
		    ## Get the current section to use
		    try:
			tInt, tStart, data = idf.read(0.1, timeInSamples=True)
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
			
			curr_time = time.time()
			process_time = curr_time - prev_time
			prev_time = curr_time
			self.perf_proclog.update({'acquire_time': acquire_time, 
					          'reserve_time': reserve_time, 
					          'process_time': process_time,})
	print("FillerOp - Done")


class FDomainOp(object):
	def __init__(self, log, iring, oring, ntime_gulp=2500, nchan_out=1, core=-1):
		self.log = log
		self.iring = iring
		self.oring = oring
		self.ntime_gulp = ntime_gulp
		self.nchan_out = nchan_out
		self.core = core
		
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
		
	def main(self):
		#bifrost.affinity.set_core(self.core)
		self.bind_proclog.update({'ncore': 1, 
		                          'core0': bifrost.affinity.get_core(),})
		
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
				ogulp_size = self.ntime_gulp*1*nstand*npol * 2		# ci8
				oshape = (self.ntime_gulp/nchan,nchan,nstand,npol,2)
				#self.iring.resize(igulp_size)
				self.oring.resize(ogulp_size,buffer_factor=10)
				
				# Set the output header
				ohdr = ihdr.copy()
				ohdr['nchan'] = nchan
				ohdr['nbit']  = 8
				ohdr_str = json.dumps(ohdr)
				
				# Setup the phasing terms for zenith
				phases = numpy.zeros(ishape, dtype=numpy.complex64)
				freq = numpy.fft.fftfreq(nchan, d=1.0/ihdr['bw']) + ihdr['cfreq']
				for i in xrange(nstand):
					## X
					a = ANTENNAS[2*i + 0]  
					delay = a.cable.delay(freq) - a.stand.z / speedOfLight
					phases[:,:,i,0] = numpy.exp(2j*numpy.pi*freq*delay)
					## Y
					a = ANTENNAS[2*i + 1]
					delay = a.cable.delay(freq) - a.stand.z / speedOfLight
					phases[:,:,i,1] = numpy.exp(2j*numpy.pi*freq*delay)
					
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
								fdata *= phases
								
								## Quantization
								try:
									Quantize(fdata, qdata, scale=1./numpy.sqrt(nchan))
								except NameError:
									qdata = bifrost.ndarray(shape=fdata.shape, native=False, dtype='ci8', space='system')
									Quantize(fdata, qdata, scale=1./numpy.sqrt(nchan))
									
								## Save
								rdata = qdata.view(numpy.int8).reshape(oshape)
								#rdata = rdata.copy(space='cuda')
								odata[...] = rdata
								
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
    def __init__(self, log, iring, oring, ntime_gulp=2500, core=-1, cpu=False, *args, **kwargs):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp

        self.core = core
        self.cpu = cpu        
        self.grid = None
        self.image = None


        self.antgridmap = self.make_grid(RESOLUTION,
                                    GRID_SIZE,
                                    numpy.ones((3,3)),
                                    locations).astype(numpy.float)

        self.LinAlgObj = LinAlg() # Jayce does this need to be here?


    def ant_conv(self, ngrid, kernel, pos):
        """ Function to convolve single antenna onto grid
            Args:
            ngrid: Number of grid size per side (512x512 would have ngrid=512)
            kernel: Convolution kernel
            pos: Antenna position x and y, in units of grid pixels
            Returns:
            mat: 1D mapping from antenna to grid (this is one row of the total
             gridding matrix)
            """
        mat = numpy.zeros((ngrid, ngrid))

        # Do convolution
        pos = numpy.round(pos).astype(numpy.int)  # Nearest pixel for now
        try:
            mat[pos[0]:pos[0] + kernel.shape[0], pos[1]:pos[1] + kernel.shape[1]] = kernel
        except ValueError:
            ## Ignore anything that doesn't fall on the grid
            pass
            
        return mat.reshape(-1)

    def make_grid(self, delta, ngrid, kernel, pos, wavelength=None):
        """ Function to convolve antennas onto grid. This creates the gridding matrix
        used to map antenna signals to grid.
    Args:
        delta: Grid resolution, in wavelength
        ngrid: Number of grid size per side (512x512 would have ngrid=512)
        kernel: Convolution kernel. Can be single kernel (same for all antennas),
                or list of kernels.
        pos: Antenna positions (Nant, 2), in wavelengths (if wavelength==None) or
             meters (if wavelength given)
        wavelength: wavelength of observations in meters. If None (default), pos
                    is assumed to be in wavelength already.
    Returns:
        mat: Matrix mapping from antennas to grid.
        """

        mat = numpy.zeros((int(ngrid**2), int(pos.shape[0])))
        if not isinstance(kernel, list):
            kernel = [kernel for i in range(int(pos.shape[0]))]
            # put positions in units of grid pixels
        pos[:, 0] -= numpy.nanmin(pos[:, 0])
        pos[:, 1] -= numpy.nanmin(pos[:, 1])
        if wavelength is not None:
            pos /= wavelength
        pos /= delta  # units of grid pixels

        for i, p in enumerate(pos):
            mat[:, i] = self.ant_conv(ngrid, kernel[i], p)
        return mat

    
        
    def main(self):
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=True):
                print("Sequence!")
                ihdr = json.loads(iseq.header.tostring())
                nchan = ihdr['nchan']
                nstand = ihdr['nstand']
                npol = ihdr['npol']
                
                igulp_size = self.ntime_gulp * nchan * nstand * npol * 2 # Complex 8
                ishape = (self.ntime_gulp,nchan,nstand,npol,2) 

                ohdr = ihdr.copy()
                ohdr['nbit'] = 64
                ohdr['npol'] = npol
                ohdr['nchan'] = nchan
                ohdr_str = json.dumps(ohdr)
                
                ##TODO: Setup output gulp and shape.
                oshape = (self.ntime_gulp,nchan,npol,GRID_SIZE,GRID_SIZE)
                ogulp_size = self.ntime_gulp * nchan * npol * GRID_SIZE * GRID_SIZE * 8 #Complex64
                self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size)
                self.grid = numpy.zeros(shape=(self.ntime_gulp,nchan,npol,GRID_SIZE,GRID_SIZE),dtype=numpy.complex64)
                self.grid = bifrost.ndarray(self.grid)
                self.image = numpy.zeros(shape=(self.ntime_gulp,nchan,npol,GRID_SIZE,GRID_SIZE),dtype=numpy.complex64)
                while not self.iring.writing_ended():
                    print("More spans")
                    iseq_spans = iseq.read(igulp_size)
                    for ispan in iseq_spans:
                        if ispan.size < igulp_size:
                            continue
                        with oring.begin_sequence(header=ohdr_str) as oseq:
                            print("New sequence")
                            with oseq.reserve(ogulp_size) as ospan:
                                    ###### Correlator #######
                                    ## Check the casting of ingulp to bfidata ci8
                                    idata = ispan.data_view(numpy.int8).reshape(ishape)
                                    #idata = idata[:,:,0:255,:,:] # Get rid of outrigger.
                                    odata = ospan.data_view(numpy.complex64).reshape(oshape)
                                    if(self.cpu == True): #CPU
                                        for time_index in numpy.arange(0,self.ntime_gulp):
                                            print("TI: %d"%time_index)

                                            for i in numpy.arange(nchan): 
                                                for j in numpy.arange(npol):
                                                    evectorr = idata[time_index,i,:,j,0]
                                                    evectorc = idata[time_index,i,:,j,1]
                                                    #t1 = time.time()
                                                    dotr = numpy.dot(self.antgridmap,evectorr).reshape(GRID_SIZE,GRID_SIZE)
                                                    doti = numpy.dot(self.antgridmap,evectorc).reshape(GRID_SIZE,GRID_SIZE)
                                                    #t2 = time.time()
                                                    #print("DOT TIME: %f"%(t2-t1))
                                                    self.grid[time_index,i,j,:,:] += dotr + 1J*doti
                                          
                                            #FFT
                                            for i in numpy.arange(nchan):
                                                for j in numpy.arange(npol):
                                                    self.image[time_index,i,j,:,:] = numpy.fft.fftshift(self.grid[time_index,i,j,:,:])
                                                    self.image[time_index,i,j,:,:] = numpy.fft.ifft2(self.image[time_index,i,j,:,:])
                                                    self.image[time_index,i,j,:,:] = numpy.fft.fftshift(self.image[time_index,i,j,:,:])
                                            odata[...] = self.image
                                    else: #GPU
                                        ## Combine stand and pols into standpols, promote the data to complex64, and 
                                        ## move it over to the GPU 
                                        idata = numpy.reshape(idata, (self.ntime_gulp,nchan,-1,2))
                                        idata = (idata[...,0] + 1j*idata[...,1]).astype(numpy.complex64)
                                        bfidata = bifrost.ndarray(shape=(self.ntime_gulp,nchan,nstand*npol),
                                                                  dtype=numpy.complex64, 
                                                                  buffer=idata.ctypes.data)
                                        #bfidata = bifrost.ndarray(shape=(self.ntime_gulp,nchan,nstand*npol),
                                        #                          dtype='ci8', 
                                        #                          buffer=idata.ctypes.data)
                                        bfidata = bfidata.copy(space='cuda')
                                        
                                        ## Make sure we have gridding kernel on the GPU
                                        try:
                                            bfantgridmap
                                        except NameError:
                                            antgrid = numpy.zeros((1,2,GRID_SIZE**2,nstand*npol), dtype=numpy.complex64)
                                            antgrid[:,0,:,0::2] = self.antgridmap
                                            antgrid[:,1,:,1::2] = self.antgridmap
                                            antgrid = antgrid.reshape(1,2*GRID_SIZE**2,nstand*npol)
                                            
                                            bfantgridmap = bifrost.ndarray(antgrid)
                                            #bfantgridmap = bifrost.ndarray(shape=(1,2*GRID_SIZE**2,nstand*npol), 
                                            #                                dtype='ci8',
                                            #                                buffer=antgrid.ctypes.data)
                                            bfantgridmap = bfantgridmap.copy(space='cuda')
                                            
                                        ## Make sure we have a place to put the data
                                        try:
                                            gdata = gdata.reshape(self.ntime_gulp,nchan,2*GRID_SIZE*GRID_SIZE)
                                        except NameError:
                                            gdata = bifrost.zeros(shape=(self.ntime_gulp,nchan,2*GRID_SIZE*GRID_SIZE),dtype=numpy.complex64, space='cuda')
                                            
                                        ## Grid the data
                                        gdata = self.LinAlgObj.matmul(1.0, bfidata, bfantgridmap.transpose(0,2,1), 0.0, gdata)
                                        
                                        ## Inverse transform
                                        gdata = gdata.reshape(self.ntime_gulp,nchan,2,GRID_SIZE,GRID_SIZE)
                                        try:
                                            bf_fft.execute(gdata,fdata,inverse=True)
                                        except NameError:
                                            fdata = bifrost.ndarray(shape=gdata.shape, dtype=numpy.complex64, space='cuda')
                                            bf_fft = Fft()
                                            bf_fft.init(gdata,fdata,axes=(3,4))
                                            bf_fft.execute(gdata,fdata,inverse=True)
                                            
                                        ## Combine the polarizations and output
                                        odata[...] = numpy.fft.fftshift(fdata.copy(space='system') , axes=(3,4))                     

                                        
                                        
                                
                                



                        

class ImagingOp(object):
    def __init__(self, log, iring, filename, ntime_gulp=100, accumulation_time=1, core=-1,*args, **kwargs):
        self.log = log
        self.iring = iring
        self.filename = filename
        self.ntime_gulp= ntime_gulp
        self.accumulation_time = accumulation_time
        self.core = core
        self.accumulated_image = None
        self.newflag = True

    def main(self):
        #bifrost.affinity.set_core(self.core)
        accum = 0
        fileid = 1
        for iseq in self.iring.read(guarantee=True):
            print("New Sequence")
            ihdr = json.loads(iseq.header.tostring())
            nchan = ihdr['nchan']
            npol = ihdr['npol']
            print("Channel no: %d, Polarisation no: %d"%(nchan,npol))
            igulp_size = self.ntime_gulp * nchan * npol * GRID_SIZE * GRID_SIZE * 8
            ishape = (self.ntime_gulp,nchan,npol,GRID_SIZE,GRID_SIZE)     


            while not self.iring.writing_ended():
                iseq_spans = iseq.read(igulp_size)
                print("Imaging ISEQ SPAN")
                for ispan in iseq_spans:
                    if ispan.size < igulp_size:
                        continue
                    idata = ispan.data_view(numpy.complex64).reshape(ishape)
                    #Square
                    xdata, ydata = idata[:,:,0,:,:], idata[:,:,1,:,:]
                    xxdata = numpy.abs(xdata*xdata.conj())
                    xydata = numpy.abs(xdata*ydata.conj()) \
                             * numpy.where( numpy.angle(xdata*ydata.conj()) > 0, 1, -1 )
                    yxdata = numpy.abs(ydata*xdata.conj())\
                             * numpy.where( numpy.angle(ydata*xdata.conj()) > 0, 1, -1 )
                    yydata = numpy.abs(ydata*ydata.conj())


                    for ti in numpy.arange(0,self.ntime_gulp):
                        #print ("Accum: %d"%accum,end='\n')
                        if self.newflag is True:
                        
                            self.accumulated_image = numpy.zeros(shape=(nchan,npol**2,GRID_SIZE,GRID_SIZE),dtype=numpy.complex64)
                            self.newflag=False
                    
                        #Accumulate
                        #self.accumulated_image = self.accumulated_image + idata[ti,:,:,:,:]
                        self.accumulated_image[:,0,:,:] += xxdata[ti,:,:,:]
                        self.accumulated_image[:,1,:,:] += xydata[ti,:,:,:]
                        self.accumulated_image[:,2,:,:] += yxdata[ti,:,:,:]
                        self.accumulated_image[:,3,:,:] += yydata[ti,:,:,:]
                        
                        #Save and output
                        accum += 1
                        if accum >= self.accumulation_time:
                            accum = 0
                            # Save and output
                            ## TODO: Do this in Matplotlib. For now NPZ files..
                            numpy.savez(self.filename+'%04i.png'%(fileid),image=self.accumulated_image)
                            self.newflag = True
                            fig = plt.figure(1)
                            #print(numpy.shape(self.image[2,0,:,:]))
                            #print(self.image[2,0,:,:])
                            
                            for i in xrange(npol**2):
                                ax = fig.add_subplot(2, 2, i+1)
                                ax.imshow(numpy.real(self.accumulated_image[0,i,:,:].T))
                            #plt.imshow(numpy.real(self.accumulated_image[0,0,:,:].T))
                            plt.savefig("ImagingOP-%04i.png"%(fileid))
                            fileid += 1
                            print("ImagingOP - Image Saved")
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
    parser.add_argument('-c', '--cpuonly', action='store_true', help = 'Runs EPIC Correlator on CPU Only.')
    parser.add_argument('-t', '--nts',type=int, default = 1000, help= 'Number of timestamps per span.')
    parser.add_argument('-u', '--accumulate',type=int, default = 1000, help='How many milliseconds to accumulate an image over.')
    parser.add_argument('-n', '--channels',type=int, default=4, help='How many channels to produce.')

    args = parser.parse_args()
    # Logging Setup
    # TODO: Set this up properly
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

    fcapture_ring = Ring(name="capture")
    if args.offline:
        fdomain_ring = Ring(name="fengine")
    #Think flagging/gains should be done on CPU?
    #calibration_ring = Ring(name="gains", space="cuda")
    if args.cpuonly:
        gridandfft_ring = Ring(name="gridandfft")
    else:
        gridandfft_ring = Ring(name="gridandfft", space="system")
    
    ops.append(OfflineCaptureOp(log, fcapture_ring,args.tbnfile))
    ops.append(FDomainOp(log, fcapture_ring, fdomain_ring, ntime_gulp=args.nts, nchan_out=1))
    ops.append(MOFFCorrelatorOp(log, fdomain_ring, gridandfft_ring, ntime_gulp=args.nts, cpu=args.cpuonly))
    ops.append(ImagingOp(log, gridandfft_ring, "EPIC_", ntime_gulp=args.nts, accumulation_time=args.accumulate))

    threads= [threading.Thread(target=op.main) for op in ops]

    for thread in threads:
        thread.daemon = False
        thread.start()

    while threads[0].is_alive() and not shutdown_event.is_set():
        time.sleep(0.5)
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
    
    
