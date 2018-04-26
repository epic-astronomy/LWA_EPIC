#!/usr/bin/env python
# -*- coding: utf-8 -*-

from adp import MCS2 as MCS
from adp import Adp
from adp.AdpCommon import *
from adp import ISC

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.udp_capture import UDPCapture
from bifrost.udp_transmit import UDPTransmit
from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.unpack import unpack as Unpack
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost.memory import memcpy as BFMemCopy, memset as BFMemSet
from bifrost.linalg import LinAlg
from bifrost import map as BFMap, asarray as BFAsArray
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, stream_synchronize as BFSync, set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()

#import numpy as np
import signal
import logging
import time
import os
import argparse
import ctypes
import threading
import json
import socket
import struct
#import time
import datetime
from collections import deque

ACTIVE_COR_CONFIG = threading.Event()

__version__    = "0.2"
__date__       = '$LastChangedDate: 2016-08-09 15:44:00 -0600 (Fri, 25 Jul 2014) $'
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2016, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"
#{"nbit": 4, "nchan": 136, "nsrc": 16, "chan0": 1456, "time_tag": 288274740432000000}
class CaptureOp(object):
	def __init__(self, log, *args, **kwargs):
		self.log    = log
		self.args   = args
		self.kwargs = kwargs
		self.utc_start = self.kwargs['utc_start']
		del self.kwargs['utc_start']
		self.shutdown_event = threading.Event()
		## HACK TESTING
		#self.seq_callback = None
	def shutdown(self):
		self.shutdown_event.set()
	def seq_callback(self, seq0, chan0, nchan, nsrc,
	                 time_tag_ptr, hdr_ptr, hdr_size_ptr):
		timestamp0 = int((self.utc_start - ADP_EPOCH).total_seconds())
		time_tag0  = timestamp0 * int(FS)
		time_tag   = time_tag0 + seq0*(int(FS)//int(CHAN_BW))
		print "++++++++++++++++ seq0     =", seq0
		print "                 time_tag =", time_tag
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
			'nbit':     4
		}
		print "******** CFREQ:", hdr['cfreq']
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
		with UDPCapture(*self.args,
		                sequence_callback=seq_callback,
		                **self.kwargs) as capture:
			while not self.shutdown_event.is_set():
				status = capture.recv()
				#print status
		del capture

class CopyOp(object):
	def __init__(self, log, iring, oring, ntime_gulp=2500,# ntime_buf=None,
	             guarantee=True, core=-1):
		self.log = log
		self.iring = iring
		self.oring = oring
		self.ntime_gulp = ntime_gulp
		#if ntime_buf is None:
		#	ntime_buf = self.ntime_gulp*3
		#self.ntime_buf = ntime_buf
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
		cpu_affinity.set_core(self.core)
		self.bind_proclog.update({'ncore': 1,
							 'core0': cpu_affinity.get_core(),})

		with self.oring.begin_writing() as oring:
			for iseq in self.iring.read(guarantee=self.guarantee):
				ihdr = json.loads(iseq.header.tostring())

				self.sequence_proclog.update(ihdr)

				self.log.info("Copy: Start of new sequence: %s", str(ihdr))

				nchan  = ihdr['nchan']
				nstand = ihdr['nstand']
				npol   = ihdr['npol']
				igulp_size = self.ntime_gulp*nchan*nstand*npol
				ogulp_size = igulp_size
				#obuf_size  = 5*25000*nchan*nstand*npol
				self.iring.resize(igulp_size, igulp_size*10)
				#self.oring.resize(ogulp_size)#, obuf_size)

				ticksPerTime = int(FS / CHAN_BW)
				base_time_tag = iseq.time_tag

				ohdr = ihdr.copy()

				prev_time = time.time()
				iseq_spans = iseq.read(igulp_size)
				while not self.iring.writing_ended():
					reset_sequence = False

					ohdr['timetag'] = base_time_tag
					ohdr_str = json.dumps(ohdr)

					with oring.begin_sequence(time_tag=base_time_tag, header=ohdr_str) as oseq:
						for ispan in iseq_spans:
							if ispan.size < igulp_size:
								continue # Ignore final gulp
							curr_time = time.time()
							acquire_time = curr_time - prev_time
							prev_time = curr_time

							try:
								with oseq.reserve(ogulp_size, nonblocking=True) as ospan:
									curr_time = time.time()
									reserve_time = curr_time - prev_time
									prev_time = curr_time

									idata = ispan.data_view(np.uint8)
									odata = ospan.data_view(np.uint8)
									BFMemCopy(odata, idata)
									#print "COPY"

							except IOError:
								curr_time = time.time()
								reserve_time = curr_time - prev_time
								prev_time = curr_time

								reset_sequence = True

							## Update the base time tag
							base_time_tag += self.ntime_gulp*ticksPerTime

							curr_time = time.time()
							process_time = curr_time - prev_time
							prev_time = curr_time
							self.perf_proclog.update({'acquire_time': acquire_time,
							                          'reserve_time': reserve_time,
							                          'process_time': process_time,})

							# Reset to move on to the next input sequence?
							if reset_sequence:
								break

					# Reset to move on to the next input sequence?
					if not reset_sequence:
						break

def get_time_tag(dt=datetime.datetime.utcnow(), seq_offset=0):
	timestamp = int((dt - ADP_EPOCH).total_seconds())
	time_tag  = timestamp*int(FS) + seq_offset*(int(FS)//int(CHAN_BW))
	return time_tag
def seq_to_time_tag(seq):
	return seq*(int(FS)//int(CHAN_BW))
def time_tag_to_seq_float(time_tag):
	return time_tag*CHAN_BW/FS

def gen_tbf_header(chan0, time_tag, time_tag0):
	sync_word    = 0xDEC0DE5C
	idval        = 0x01
	#frame_num    = (time_tag % int(FS)) // NFRAME_PER_SPECTRUM # Spectrum no.
	frame_num_wrap = 10*60 * int(CHAN_BW) # 10 mins = 15e6, just fits within a uint24
	frame_num    = ((time_tag - time_tag0) // NFRAME_PER_SPECTRUM) % frame_num_wrap + 1 # Spectrum no.
	id_frame_num = idval << 24 | frame_num
	secs_count   = time_tag // int(FS) - M5C_OFFSET
	freq_chan    = chan0
	unassigned   = 0
	return struct.pack('>IIIhhq',
	                   sync_word,
	                   id_frame_num,
	                   secs_count,
	                   freq_chan,
	                   unassigned,
	                   time_tag)

class TriggeredDumpOp(object):
	def __init__(self, log, osock, iring, ntime_gulp, ntime_buf, tuning=0, nchan_max=256, core=-1, max_bytes_per_sec=None):
		self.log = log
		self.sock = osock
		self.iring = iring
		self.tuning = tuning
		self.nchan_max = nchan_max
		self.core  = core
		self.ntime_gulp = ntime_gulp
		self.ntime_buf = ntime_buf

		self.bind_proclog = ProcLog(type(self).__name__+"/bind")
		self.in_proclog   = ProcLog(type(self).__name__+"/in")

		self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})

		self.configMessage = ISC.TriggerClient(addr=('adp',5832))
		self.tbfLock       = ISC.PipelineEventClient(addr=('adp',5834))

		if max_bytes_per_sec is None:
			max_bytes_per_sec = 104857600		# default to 100 MB/s
		self.max_bytes_per_sec = max_bytes_per_sec

	def main(self):
		cpu_affinity.set_core(self.core)
		self.bind_proclog.update({'ncore': 1,
							 'core0': cpu_affinity.get_core(),})

		ninput_max = 512
		frame_nbyte_max = self.nchan_max*ninput_max
		#self.iring.resize(self.ntime_gulp*frame_nbyte_max,
		#                  self.ntime_buf *frame_nbyte_max)

		while not self.iring.writing_ended():
			config = self.configMessage(block=False)
			if not config:
				time.sleep(0.1)
				continue

			print "Trigger: New trigger received: %s" % str(config)
			try:
				self.dump(samples=config[1], mask=config[2], local=config[3])
			except RuntimeError as e:
				print "Error on TBF dump: %s" % str(e)
		print "Writing ended, TBFOp exiting"

	def dump(self, samples, time_tag=None, mask=None, local=False):
		if mask is None:
			mask = 0b11
		if (mask >> self.tuning) & 1 == 0:
			self.log.info('Not for us: %i -> %i @ %i', mask, (mask >> self.tuning) & 1, self.tuning)
			return False
		speed_factor = 2 / sum([mask>>i&1 for i in xrange(2)])		# TODO: Slightly hacky

		ntime_pkt = 1 # TODO: Should be TBF_NTIME_PER_PKT?

		# HACK TESTING
		dump_time_tag = time_tag
		if dump_time_tag is None:
			time_offset    = -4.0
			time_offset_s  = int(time_offset)
			time_offset_us = int(round((time_offset-time_offset_s)*1e6))
			time_offset    = datetime.timedelta(seconds=time_offset_s, microseconds=time_offset_us)

			utc_now = datetime.datetime.utcnow()
			dump_time_tag = get_time_tag(utc_now + time_offset)
		#print "********* dump_time_tag =", dump_time_tag
		#time.sleep(3)
		#ntime_dump = 0.25*1*25000
		#ntime_dump = 0.1*1*25000
		ntime_dump = int(round(time_tag_to_seq_float(samples)))

		print "TBF DUMPING %f secs at time_tag = %i (%s)%s" % (samples/FS, dump_time_tag, datetime.datetime.utcfromtimestamp(dump_time_tag/FS), (' locallay' if local else ''))
		if not local:
			self.tbfLock.set()
		with self.iring.open_sequence_at(dump_time_tag, guarantee=True) as iseq:
			time_tag0 = iseq.time_tag
			ihdr = json.loads(iseq.header.tostring())
			nchan  = ihdr['nchan']
			chan0  = ihdr['chan0']
			nstand = ihdr['nstand']
			npol   = ihdr['npol']
			ninput = nstand*npol
			print "*******", nchan, ninput
			ishape = (-1,nchan,ninput)#,2)
			frame_nbyte = nchan*ninput#*2
			igulp_size = self.ntime_gulp*nchan*ninput#*2

			dump_seq_offset  = int(time_tag_to_seq_float(dump_time_tag - time_tag0))
			dump_byte_offset = dump_seq_offset * frame_nbyte

			# HACK TESTING write to file instead of socket
			if local:
				filename = '/data0/test_%s_%i_%020i.tbf' % (socket.gethostname(), self.tuning, dump_time_tag)#time_tag0
				ofile = open(filename, 'wb')
			else:
				udt = UDPTransmit(sock=self.sock, core=self.core)
			ntime_dumped = 0
			nchan_rounded = nchan // TBF_NCHAN_PER_PKT * TBF_NCHAN_PER_PKT
			bytesSent, bytesStart = 0.0, time.time()

			print "Opening read space of %i bytes at offset = %i" % (igulp_size, dump_byte_offset)
			for ispan in iseq.read(igulp_size, begin=dump_byte_offset):
				print "**** ispan.size, offset", ispan.size, ispan.offset
				print "**** Dumping at", ntime_dumped
				if ntime_dumped >= ntime_dump:
					break
				#print ispan.offset, seq_offset
				seq_offset = ispan.offset // frame_nbyte
				data = ispan.data.reshape(ishape)

				for t in xrange(0, self.ntime_gulp, ntime_pkt):
					if ntime_dumped >= ntime_dump:
						break
					ntime_dumped += 1

					pkts = []
					time_tag = time_tag0 + seq_to_time_tag(seq_offset + t)
					if t == 0:
						print "**** first timestamp is", time_tag
					for c in xrange(0, nchan_rounded, TBF_NCHAN_PER_PKT):
						pktdata = data[t:t+ntime_pkt,c:c+TBF_NCHAN_PER_PKT]
						hdr = gen_tbf_header(chan0+c, time_tag, time_tag0)
						try:
							pkt = hdr + pktdata.tostring()
							pkts.append( pkt )
						except Exception as e:
								print 'Packing Error', str(e)

					if local:
						for pkt in pkts:
							ofile.write(pkt)
							bytesSent += len(pkt)

					if not local:
						try:
							udt.sendmany(pkts)
							bytesSent += sum([len(p) for p in pkts])
						except Exception as e:
							print 'Sending Error', str(e)

						while bytesSent/(time.time()-bytesStart) >= self.max_bytes_per_sec*speed_factor:
							time.sleep(0.001)
		if local:
			ofile.close()
		if not local:
			del udt
			self.tbfLock.clear()
		print "TBF DUMP COMPLETE - average rate was %.3f MB/s" % (bytesSent/(time.time()-bytesStart)/1024**2,)

class BeamformerOp(object):
	# Note: Input data are: [time,chan,ant,pol,cpx,8bit]
	def __init__(self, log, iring, oring, tuning=0, nchan_max=256, nbeam_max=1, nroach=16, ntime_gulp=2500, guarantee=True, core=-1, gpu=-1):
		self.log   = log
		self.iring = iring
		self.oring = oring
		self.tuning = tuning
		ninput_max = nroach*32#*2
		self.ntime_gulp = ntime_gulp
		self.guarantee = guarantee
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
		self.nbeam_max = nbeam_max
		self.configMessage = ISC.BAMConfigurationClient(addr=('adp',5832))
		self._pending = deque()

		# Setup the beamformer
		if self.gpu != -1:
			BFSetGPU(self.gpu)
		## Metadata
		nchan = self.nchan_max
		nstand, npol = nroach*16, 2
		## Object
		self.bfbf = LinAlg()
		## Delays and gains
		self.delays = np.zeros((self.nbeam_max*2,nstand*npol), dtype=np.float64)
		self.gains = np.zeros((self.nbeam_max*2,nstand*npol), dtype=np.float64)
		self.cgains = BFArray(shape=(self.nbeam_max*2,nchan,nstand*npol), dtype=np.complex64, space='cuda')
		## Intermidiate arrays
		## NOTE:  This should be OK to do since the roaches only output one bandwidth per INI
		self.tdata = BFArray(shape=(self.ntime_gulp,nchan,nstand*npol), dtype='ci4', native=False, space='cuda')
		self.bdata = BFArray(shape=(nchan,self.nbeam_max*2,self.ntime_gulp), dtype=np.complex64, space='cuda')

	#@ISC.logException
	def updateConfig(self, config, hdr, time_tag, forceUpdate=False):
		# Get the current pipeline time to figure out if we need to shelve a command or not
		pipeline_time = time_tag / FS

		# Can we act on this configuration change now?
		if config:
			## Pull out the tuning (something unique to DRX/BAM/COR)
			beam, tuning = config[0], config[3]
			if beam > self.nbeam_max or tuning != self.tuning:
				return False

			## Set the configuration time - BAM commands are for the specified slot in the next second
			slot = config[4] / 100.0
			config_time = int(time.time()) + 1 + slot

			## Is this command from the future?
			if pipeline_time < config_time:
				### Looks like it, save it for later
				self._pending.append( (config_time, config) )
				config = None

				### Is there something pending?
				try:
					stored_time, stored_config = self._pending[0]
					if pipeline_time >= stored_time:
						config_time, config = self._pending.popleft()
				except IndexError:
					pass
			else:
				### Nope, this is something we can use now
				pass

		else:
			## Is there something pending?
			try:
				stored_time, stored_config = self._pending[0]
				if pipeline_time >= stored_time:
					config_time, config = self._pending.popleft()
			except IndexError:
				#print "No pending configuation at %.1f" % pipeline_time
				pass

		if config:
			self.log.info("Beamformer: New configuration received for beam %i (delta = %.1f subslots)", config[0], (pipeline_time-config_time)*100.0)
			beam, delays, gains, tuning, slot = config
			if tuning != self.tuning:
				self.log.info("Beamformer: Not for this tuning, skipping")
				return False

			# Byteswap to get into little endian
			delays = delays.byteswap().newbyteorder()
			gains = gains.byteswap().newbyteorder()

			# Unpack and re-shape the delays (to seconds) and gains (to floating-point)
			delays = (((delays>>4)&0xFFF) + (delays&0xF)/16.0) / FS
			gains = gains/32767.0
			gains.shape = (gains.size/2, 2)

			# Update the internal delay and gain cache so that we can use these later
			self.delays[2*(beam-1)+0,:] = delays
			self.delays[2*(beam-1)+1,:] = delays
			self.gains[2*(beam-1)+0,:] = gains[:,0]
			self.gains[2*(beam-1)+1,:] = gains[:,1]

			# Compute the complex gains needed for the beamformer
			freqs = CHAN_BW * (hdr['chan0'] + np.arange(hdr['nchan']))
			freqs.shape = (freqs.size, 1)
			self.cgains[2*(beam-1)+0,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*(beam-1)+0,:]) * \
			                                 self.gains[2*(beam-1)+0,:]).astype(np.complex64)
			self.cgains[2*(beam-1)+1,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*(beam-1)+1,:]) * \
			                                 self.gains[2*(beam-1)+1,:]).astype(np.complex64)
			self.log.info('  Complex gains set - beam %i' % beam)

			return True

		elif forceUpdate:
			self.log.info("Beamformer: New sequence configuration received")

			# Compute the complex gains needed for the beamformer
			freqs = CHAN_BW * (hdr['chan0'] + np.arange(hdr['nchan']))
			freqs.shape = (freqs.size, 1)
			for beam in xrange(1, self.nbeam_max+1):
				self.cgains[2*(beam-1)+0,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*(beam-1)+0,:]) * \
				                                 self.gains[2*(beam-1)+0,:]).astype(np.complex64)
				self.cgains[2*(beam-1)+1,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*(beam-1)+1,:]) * \
				                                 self.gains[2*(beam-1)+1,:]).astype(np.complex64)
				self.log.info('  Complex gains set - beam %i' % beam)

			return True

		else:
			return False

	#@ISC.logException
	def main(self):
		cpu_affinity.set_core(self.core)
		if self.gpu != -1:
			BFSetGPU(self.gpu)
		self.bind_proclog.update({'ncore': 1,
							 'core0': cpu_affinity.get_core(),
							 'ngpu': 1,
							 'gpu0': BFGetGPU(),})

		with self.oring.begin_writing() as oring:
			for iseq in self.iring.read(guarantee=self.guarantee):
				ihdr = json.loads(iseq.header.tostring())

				self.sequence_proclog.update(ihdr)

				nchan  = ihdr['nchan']
				nstand = ihdr['nstand']
				npol   = ihdr['npol']

				status = self.updateConfig( self.configMessage(), ihdr, iseq.time_tag, forceUpdate=True )

				igulp_size = self.ntime_gulp*nchan*nstand*npol		# 4+4 complex
				ogulp_size = self.ntime_gulp*nchan*self.nbeam_max*npol*8			# complex64
				ishape = (self.ntime_gulp,nchan,nstand*npol)
				oshape = (self.ntime_gulp,nchan,self.nbeam_max*2)

				ticksPerTime = int(FS) / int(CHAN_BW)
				base_time_tag = iseq.time_tag

				ohdr = ihdr.copy()
				ohdr['nstand'] = self.nbeam_max
				ohdr['nbit'] = 32
				ohdr['complex'] = True
				ohdr_str = json.dumps(ohdr)

				self.oring.resize(ogulp_size)

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

							## Setup and load
							idata = ispan.data_view(np.uint8).reshape(ishape)
							odata = ospan.data_view(np.complex64).reshape(oshape)

							## Fix the type
							bfidata = BFArray(shape=idata.shape, dtype='ci4', native=False, buffer=idata.ctypes.data)

							## Copy
							copy_array(self.tdata, bfidata)

							## Beamform
							self.bdata = self.bfbf.matmul(1.0, self.cgains.transpose(1,0,2), self.tdata.transpose(1,2,0), 0.0, self.bdata)

							## Save and cleanup
							odata[...] = self.bdata.copy(space='system').transpose(2,0,1)

						## Update the base time tag
						base_time_tag += self.ntime_gulp*ticksPerTime

						## Check for an update to the configuration
						self.updateConfig( self.configMessage(), ihdr, base_time_tag, forceUpdate=False )

						curr_time = time.time()
						process_time = curr_time - prev_time
						prev_time = curr_time
						self.perf_proclog.update({'acquire_time': acquire_time,
						                          'reserve_time': reserve_time,
						                          'process_time': process_time,})

class CorrelatorOp(object):
	# Note: Input data are: [time,chan,ant,pol,cpx,8bit]
	def __init__(self, log, iring, oring, tuning=0, nchan_max=256, nroach=16, ntime_gulp=2500, guarantee=True, core=-1, gpu=-1):
		self.log   = log
		self.iring = iring
		self.oring = oring
		self.tuning = tuning
		ninput_max = nroach*32#*2
		self.ntime_gulp = ntime_gulp
		self.guarantee = guarantee
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
		self.configMessage = ISC.CORConfigurationClient(addr=('adp',5832))
		self._pending = deque()
		self.navg = 5*100
		self.gain = 0

		# Setup the correlator
		if self.gpu != -1:
			BFSetGPU(self.gpu)
		## Metadata
		nchan = self.nchan_max
		nstand, npol = nroach*16, 2
		## Decimation
		self.chan_decim = 1
		ochan = nchan / self.chan_decim
		## Object
		self.bfcc = LinAlg()
		## Intermidiate arrays
		## NOTE:  This should be OK to do since the roaches only output one bandwidth per INI
		self.tdata = BFArray(shape=(self.ntime_gulp,ochan,nstand*npol), dtype='ci4', native=False, space='cuda')
		self.udata = BFArray(shape=(self.ntime_gulp,ochan,nstand*npol), dtype='ci8', space='cuda')
		self.cdata = BFArray(shape=(ochan,nstand*npol,nstand*npol), dtype=np.complex64, space='cuda')

	#@ISC.logException
	def updateConfig(self, config, hdr, time_tag, forceUpdate=False):
		global ACTIVE_COR_CONFIG

		# Get the current pipeline time to figure out if we need to shelve a command or not
		pipeline_time = time_tag / FS

		# Can we act on this configuration change now?
		if config:
			## Pull out the tuning (something unique to DRX/BAM/COR)
			tuning = config[1]
			if tuning != self.tuning:
				return False

			## Set the configuration time - COR commands are for the specified slot in the next second
			slot = config[3] / 100.0
			config_time = int(time.time()) + 1 + slot

			## Is this command from the future?
			if pipeline_time < config_time:
				### Looks like it, save it for later
				self._pending.append( (config_time, config) )
				config = None

				### Is there something pending?
				try:
					stored_time, stored_config = self._pending[0]
					if pipeline_time >= stored_time:
						config_time, config = self._pending.popleft()
				except IndexError:
					pass
			else:
				### Nope, this is something we can use now
				pass

		else:
			## Is there something pending?
			try:
				stored_time, stored_config = self._pending[0]
				if pipeline_time >= stored_time:
					config_time, config = self._pending.popleft()
			except IndexError:
				#print "No pending configuation at %.1f" % pipeline_time
				pass

		if config:
			self.log.info("Correlator: New configuration received for tuning %i (delta = %.1f subslots)", config[1], (pipeline_time-config_time)*100.0)
			navg, tuning, gain, slot = config
			if tuning != self.tuning:
				self.log.info("Correlator: Not for this tuning, skipping")
				return False

			self.navg = navg
			self.log.info('  Averaging time set')
			self.gain = gain
			self.log.info('  Gain set')

			ACTIVE_COR_CONFIG.set()

			return True

		elif forceUpdate:
			self.log.info("Correlator: New sequence configuration received")

			return True

		else:
			return False

	#@ISC.logException
	def main(self):
		cpu_affinity.set_core(self.core)
		if self.gpu != -1:
			BFSetGPU(self.gpu)
		self.bind_proclog.update({'ncore': 1,
							 'core0': cpu_affinity.get_core(),
							 'ngpu': 1,
							 'gpu0': BFGetGPU(),})

		with self.oring.begin_writing() as oring:
			for iseq in self.iring.read(guarantee=self.guarantee):
				ihdr = json.loads(iseq.header.tostring())

				self.sequence_proclog.update(ihdr)

				self.updateConfig( self.configMessage(), ihdr, iseq.time_tag, forceUpdate=True )

				self.log.info("Correlator: Start of new sequence: %s", str(ihdr))

				nchan  = ihdr['nchan']
				ochan  = nchan / self.chan_decim
				nstand = ihdr['nstand']
				npol   = ihdr['npol']
				igulp_size = self.ntime_gulp*nchan*nstand*npol		# 4+4 complex
				ogulp_size = ochan*nstand*npol*nstand*npol*8			# complex64
				ishape = (self.ntime_gulp,nchan,nstand*npol)
				oshape = (ochan,nstand*npol,nstand*npol)

				ticksPerTime = int(FS) / int(CHAN_BW)
				base_time_tag = iseq.time_tag

				ohdr = ihdr.copy()
				ohdr['nchan'] = ochan
				ohdr['nbit'] = 32
				ohdr['complex'] = True
				ohdr_str = json.dumps(ohdr)

				self.oring.resize(ogulp_size)

				nAccumulate = 0

				prev_time = time.time()
				iseq_spans = iseq.read(igulp_size)
				while not self.iring.writing_ended():
					reset_sequence = False

					navg_seq = self.navg * int(CHAN_BW/100.0)
					navg_seq = int(navg_seq / self.ntime_gulp) * self.ntime_gulp
					gain_act = 1.0 / 2**self.gain / navg_seq
					navg = navg_seq / int(CHAN_BW/100.0)

					ohdr['time_tag'] = base_time_tag
					ohdr['navg']     = navg
					ohdr['gain']     = self.gain
					ohdr_str = json.dumps(ohdr)

					with oring.begin_sequence(time_tag=base_time_tag, header=ohdr_str) as oseq:
						for ispan in iseq_spans:
							if ispan.size < igulp_size:
								continue # Ignore final gulp
							curr_time = time.time()
							acquire_time = curr_time - prev_time
							prev_time = curr_time

							## Setup and load
							idata = ispan.data_view(np.uint8).reshape(ishape)
							if ochan != nchan:
								idata = idata[:,:ochan,:]

							## Fix the type
							bfidata = BFArray(shape=idata.shape, dtype='ci4', native=False, buffer=idata.ctypes.data)

							## Copy
							copy_array(self.tdata, bfidata)

							## Unpack
							Unpack(self.tdata, self.udata)

							## Correlate
							cscale = gain_act if nAccumulate else 0.0
							self.cdata = self.bfcc.matmul(gain_act, None, self.udata.transpose((1,0,2)), cscale, self.cdata)
							nAccumulate += self.ntime_gulp

							curr_time = time.time()
							process_time = curr_time - prev_time
							prev_time = curr_time

							## Dump?
							if nAccumulate == navg_seq:
								with oseq.reserve(ogulp_size) as ospan:
									odata = ospan.data_view(np.complex64).reshape(oshape)

									odata[...] = self.cdata
								nAccumulate = 0
							curr_time = time.time()
							reserve_time = curr_time - prev_time
							prev_time = curr_time

							## Update the base time tag
							base_time_tag += self.ntime_gulp*ticksPerTime

							## Check for an update to the configuration
							if self.updateConfig( self.configMessage(), ihdr, base_time_tag, forceUpdate=False ):
								reset_sequence = True
								break

							curr_time = time.time()
							process_time += curr_time - prev_time
							prev_time = curr_time
							self.perf_proclog.update({'acquire_time': acquire_time,
							                          'reserve_time': reserve_time,
							                          'process_time': process_time,})

					# Reset to move on to the next input sequence?
					if not reset_sequence:
						break

def gen_chips_header(server, nchan, chan0, seq, gbe=0, nservers=6):
	return struct.pack('>BBBBBBHQ',
				    server,
				    gbe,
				    nchan,
				    1,
				    0,
				    nservers,
				    chan0-nchan*(server-1),
				    seq)

class RetransmitOp(object):
	def __init__(self, log, osock, iring, tuning=0, nchan_max=256, ntime_gulp=2500, nbeam_max=1, guarantee=True, core=-1):
		self.log   = log
		self.sock = osock
		self.iring = iring
		self.tuning = tuning
		self.ntime_gulp = ntime_gulp
		self.nbeam_max = nbeam_max
		self.guarantee = guarantee
		self.core = core

		self.bind_proclog = ProcLog(type(self).__name__+"/bind")
		self.in_proclog   = ProcLog(type(self).__name__+"/in")
		self.size_proclog = ProcLog(type(self).__name__+"/size")
		self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
		self.perf_proclog = ProcLog(type(self).__name__+"/perf")

		self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
		self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})

		self.server = int(socket.gethostname().replace('adp', '0'), 10)
		self.nchan_max = nchan_max

	def main(self):
		cpu_affinity.set_core(self.core)
		self.bind_proclog.update({'ncore': 1,
							 'core0': cpu_affinity.get_core(),})

		for iseq in self.iring.read():
			ihdr = json.loads(iseq.header.tostring())

			self.sequence_proclog.update(ihdr)

			self.log.info("Retransmit: Start of new sequence: %s", str(ihdr))

			chan0   = ihdr['chan0']
			nchan   = ihdr['nchan']
			nstand  = ihdr['nstand']
			npol    = ihdr['npol']
			nstdpol = nstand * npol
			igulp_size = self.ntime_gulp*nchan*nstdpol*8		# complex64
			igulp_shape = (self.ntime_gulp,nchan,nstdpol)

			seq0 = ihdr['seq0']
			seq = seq0

			prev_time = time.time()
			with UDPTransmit(sock=self.sock, core=self.core) as udt:
				for ispan in iseq.read(igulp_size):
					if ispan.size < igulp_size:
						continue # Ignore final gulp
					curr_time = time.time()
					acquire_time = curr_time - prev_time
					prev_time = curr_time

					idata = ispan.data_view(np.complex64).reshape(igulp_shape)
					if nstdpol == 2:
						pdata = idata.astype(np.complex128)
					else:
						pdata = idata

					pkts = []
					for t in xrange(0, self.ntime_gulp):
						pktdata = pdata[t,:,:]
						seq_curr = seq + t
						hdr = gen_chips_header(self.server, nchan, chan0, seq_curr, gbe=self.tuning)
						pkt = hdr + pktdata.tostring()
						pkts.append( pkt )

					try:
						udt.sendmany(pkts)
					except Exception as e:
						pass

					seq += self.ntime_gulp

					curr_time = time.time()
					process_time = curr_time - prev_time
					prev_time = curr_time
					self.perf_proclog.update({'acquire_time': acquire_time,
										 'reserve_time': -1,
										 'process_time': process_time,})

			del udt

def gen_cor_header(stand0, stand1, chan0, time_tag, time_tag0, navg, gain):
	sync_word    = 0xDEC0DE5C
	idval        = 0x02
	frame_num    = 0
	id_frame_num = idval << 24 | frame_num
	#if stand == 0 and pol == 0:
	#	print cfreq, bw, gain, time_tag, time_tag0
	#	print nframe_per_sample, nframe_per_packet
	return struct.pack('>IIIhhqihh',
	                   sync_word,
	                   id_frame_num,
	                   0,
	                   chan0,
	                   gain,
	                   time_tag,
	                   navg,
	                   stand0,
	                   stand1)

class PacketizeOp(object):
	# Note: Input data are: [time,beam,pol,iq]
	def __init__(self, log, iring, osock, npkt_gulp=128, core=-1, gpu=-1, max_bytes_per_sec=None):
		self.log   = log
		self.iring = iring
		self.sock  = osock
		self.npkt_gulp = npkt_gulp
		self.core = core
		self.gpu = gpu

		self.bind_proclog = ProcLog(type(self).__name__+"/bind")
		self.in_proclog   = ProcLog(type(self).__name__+"/in")
		self.size_proclog = ProcLog(type(self).__name__+"/size")
		self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
		self.perf_proclog = ProcLog(type(self).__name__+"/perf")

		self.in_proclog.update({'nring':1, 'ring0':self.iring.name})

		if max_bytes_per_sec is None:
			max_bytes_per_sec = 104857600		# default to 100 MB/s
		self.max_bytes_per_sec = max_bytes_per_sec

	def main(self):
		global ACTIVE_COR_CONFIG

		cpu_affinity.set_core(self.core)
		if self.gpu != -1:
			BFSetGPU(self.gpu)
		self.bind_proclog.update({'ncore': 1,
		                          'core0': cpu_affinity.get_core(),
		                          'ngpu': 1,
		                          'gpu0': BFGetGPU(),})

		for iseq in self.iring.read():
			ihdr = json.loads(iseq.header.tostring())

			self.sequence_proclog.update(ihdr)

			self.log.info("Packetizer: Start of new sequence: %s", str(ihdr))

			#print 'PacketizeOp', ihdr
			chan0  = ihdr['chan0']
			nchan  = ihdr['nchan']
			nstand = ihdr['nstand']
			npol   = ihdr['npol']
			navg   = ihdr['navg']
			time_tag0 = iseq.time_tag
			time_tag  = time_tag0
			igulp_size = nchan*nstand*npol*nstand*npol*8
			ishape = (nchan,nstand*npol,nstand*npol)

			ticksPerFrame = int(round(navg*0.01*FS))

			# HACK for verification
			filename = '/data0/test_%s_%020i.cor' % (socket.gethostname(), time_tag0)#time_tag0
			ofile = open(filename, 'wb')

			prev_time = time.time()
			with UDPTransmit(sock=self.sock, core=self.core) as udt:
				for ispan in iseq.read(igulp_size):
					if ispan.size < igulp_size:
						continue # Ignore final gulp
					curr_time = time.time()
					acquire_time = curr_time - prev_time
					prev_time = curr_time

					idata = ispan.data_view(np.complex64).reshape(ishape)
					ldata = idata.copy(space='cuda')
					BFMap('a(i,j-1,j) = Complex<float>(a(i,j,j-1).real, -a(i,j,j-1).imag)', {'a':ldata}, axis_names=('i','j','k'), shape=ldata.shape)
					odata = ldata.copy(space='system')
					odata = odata.reshape(nchan,nstand,2,nstand,2)
					odata = np.swapaxes(odata, 2, 3)

					bytesSent, bytesStart = 0.0, time.time()

					time_tag_cur = time_tag + ticksPerFrame
					for i in xrange(nstand):
						pkts = []
						for j in xrange(i, nstand):
							pktdata = odata[:,j,i,:,:]
							hdr = gen_cor_header(i+1, j+1, chan0, time_tag_cur, time_tag0, navg, 1)
							try:
								pkt = hdr + pktdata.tostring()
								pkts.append( pkt )
							except Exception as e:
								print 'Packing Error', str(e)

						# HACK for verification
						#try:
						#	#if ACTIVE_COR_CONFIG.is_set():
						#	udt.sendmany(pkts)
						#except Exception as e:
						#	print 'Sending Error', str(e)
						#
						#while bytesSent/(time.time()-bytesStart) >= self.max_bytes_per_sec*speed_factor:
						#	time.sleep(0.001)

						# HACK for verification
						if os.path.getsize(filename) < 10*1024**3:
							for pkt in pkts:
								ofile.write(pkt)
							ofile.flush()
						else:
							try:
								ofile.close()
							except:
								pass
					time_tag += ticksPerFrame

					curr_time = time.time()
					process_time = curr_time - prev_time
					prev_time = curr_time
					self.perf_proclog.update({'acquire_time': acquire_time,
										 'reserve_time': -1,
										 'process_time': process_time,})

			del udt

			try:
				ofile.close()
			except:
				pass

def get_utc_start(shutdown_event=None):
	got_utc_start = False
	while not got_utc_start:
		if shutdown_event is not None:
			if shutdown_event.is_set():
				raise RuntimeError("Shutting down without getting the start time")

		try:
			with MCS.Communicator() as adp_control:
				utc_start = adp_control.report('UTC_START')
				# Check for valid timestamp
				utc_start_dt = datetime.datetime.strptime(utc_start, DATE_FORMAT)
			got_utc_start = True
		except Exception as ex:
			print ex
			time.sleep(1)
	#print "UTC_START:", utc_start
	#return utc_start
	return utc_start_dt

def get_numeric_suffix(s):
	i = 0
	while True:
		if len(s[i:]) == 0:
			raise ValueError("No numeric suffix in string '%s'" % s)
		try: return int(s[i:])
		except ValueError: i += 1

def main(argv):
	parser = argparse.ArgumentParser(description='LWA-SV ADP DRX Service')
	parser.add_argument('-f', '--fork',       action='store_true',       help='Fork and run in the background')
	parser.add_argument('-t', '--tuning',     default=0, type=int,       help='DRX tuning (0 or 1)')
	parser.add_argument('-c', '--configfile', default='adp_config.json', help='Specify config file')
	parser.add_argument('-l', '--logfile',    default=None,              help='Specify log file')
	parser.add_argument('-d', '--dryrun',     action='store_true',       help='Test without acting')
	parser.add_argument('-v', '--verbose',    action='count', default=0, help='Increase verbosity')
	parser.add_argument('-q', '--quiet',      action='count', default=0, help='Decrease verbosity')
	args = parser.parse_args()
	tuning = args.tuning

	# Fork, if requested
	if args.fork:
		stderr = '/tmp/%s_%i.stderr' % (os.path.splitext(os.path.basename(__file__))[0], tuning)
		daemonize(stdin='/dev/null', stdout='/dev/null', stderr=stderr)

	config = Adp.parse_config_file(args.configfile)
	ntuning = len(config['drx'])
	drxConfig = config['drx'][tuning]

	log = logging.getLogger(__name__)
	logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
	                              datefmt='%Y-%m-%d %H:%M:%S')
	logFormat.converter = time.gmtime
	if args.logfile is None:
		logHandler = logging.StreamHandler(sys.stdout)
	else:
		logHandler = Adp.AdpFileHandler(config, args.logfile)
	logHandler.setFormatter(logFormat)
	log.addHandler(logHandler)
	verbosity = args.verbose - args.quiet
	if   verbosity >  0: log.setLevel(logging.DEBUG)
	elif verbosity == 0: log.setLevel(logging.INFO)
	elif verbosity <  0: log.setLevel(logging.WARNING)

	short_date = ' '.join(__date__.split()[1:4])
	log.info("Starting %s with PID %i", argv[0], os.getpid())
	log.info("Cmdline args: \"%s\"", ' '.join(argv[1:]))
	log.info("Version:      %s", __version__)
	log.info("Last changed: %s", short_date)
	log.info("Current MJD:  %f", Adp.MCS2.slot2mjd())
	log.info("Current MPM:  %i", Adp.MCS2.slot2mpm())
	log.info("Config file:  %s", args.configfile)
	log.info("Log file:     %s", args.logfile)
	log.info("Dry run:      %r", args.dryrun)

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

	log.info("Waiting to get UTC_START")
	utc_start_dt = get_utc_start(shutdown_event)
	log.info("UTC_START:    %s", utc_start_dt.strftime(DATE_FORMAT))

	hostname = socket.gethostname()
	try:
		server_idx = get_numeric_suffix(hostname) - 1
	except ValueError:
		server_idx = 0 # HACK to allow testing on head node "adp"
	log.info("Hostname:     %s", hostname)
	log.info("Server index: %i", server_idx)

	## Network - input
	pipeline_idx = drxConfig['pipeline_idx']
	iaddr        = config['server']['data_ifaces'][pipeline_idx]
	iport        = config['server']['data_ports' ][pipeline_idx]
	## Network - TBF - data recorder
	recorder_idx = drxConfig['tbf_recorder_idx']
	recConfig    = config['recorder'][recorder_idx]
	oaddr        = recConfig['host']
	oport        = recConfig['port']
	obw          = recConfig['max_bytes_per_sec']
	## Network - COR - data recorder
	recorder_idx = drxConfig['cor_recorder_idx']
	recConfig    = config['recorder'][recorder_idx]
	vaddr        = recConfig['host']
	vport        = recConfig['port']
	vbw          = recConfig['max_bytes_per_sec']
	## Network - T engine
	tengine_idx  = drxConfig['tengine_idx']
	tngConfig    = config['tengine'][tengine_idx]
	taddr        = config['host']['tengines'][tengine_idx]
	tport        = config['server']['data_ports' ][tngConfig['pipeline_idx']]

	nroach_tot = len(config['host']['roaches'])
	nserver    = len(config['host']['servers'])
	nroach, roach0 = nroach_tot, 0
	nbeam = drxConfig['beam_count']
	cores = drxConfig['cpus']
	gpus  = drxConfig['gpus']

	log.info("Src address:  %s:%i", iaddr, iport)
	log.info("TBF address:  %s:%i", oaddr, oport)
	log.info("COR address:  %s:%i", vaddr, vport)
	log.info("TNG address:  %s:%i", taddr, tport)
	log.info("Roaches:      %i-%i", roach0+1, roach0+nroach)
	log.info("Tunings:      %i (of %i)", tuning+1, ntuning)
	log.info("CPUs:         %s", ' '.join([str(v) for v in cores]))
	log.info("GPUs:         %s", ' '.join([str(v) for v in gpus]))

	# Note: Capture uses Bifrost address+socket objects, while output uses
	#         plain Python address+socket objects.
	iaddr = Address(iaddr, iport)
	isock = UDPSocket()
	isock.bind(iaddr)
	isock.timeout = 0.5

	capture_ring = Ring(name="capture-%i" % tuning)
	tbf_ring     = Ring(name="buffer-%i" % tuning)
	tengine_ring = Ring(name="tengine-%i" % tuning)
	vis_ring     = Ring(name="vis-%i" % tuning, space='cuda')

	tbf_buffer_secs = int(round(config['tbf']['buffer_time_sec']))

	oaddr = Address(oaddr, oport)
	osock = UDPSocket()
	osock.connect(oaddr)

	vaddr = Address(vaddr, vport)
	vsock = UDPSocket()
	vsock.connect(vaddr)

	taddr = Address(taddr, tport)
	tsock = UDPSocket()
	tsock.connect(taddr)

	nchan_max = int(round(drxConfig['capture_bandwidth']/CHAN_BW/nserver))
	tbf_bw_max    = obw/nserver/ntuning
	cor_bw_max    = vbw/nserver/ntuning

	# TODO:  Figure out what to do with this resize
	GSIZE = 500
	ogulp_size = GSIZE *nchan_max*256*2
	obuf_size  = tbf_buffer_secs*25000 *nchan_max*256*2
	tbf_ring.resize(ogulp_size, obuf_size)

	ops.append(CaptureOp(log, fmt="chips", sock=isock, ring=capture_ring,
	                     nsrc=nroach, src0=roach0, max_payload_size=9000,
	                     buffer_ntime=GSIZE, slot_ntime=25000, core=cores.pop(0),
	                     utc_start=utc_start_dt))
	ops.append(CopyOp(log, capture_ring, tbf_ring,
	                  ntime_gulp=GSIZE, #ntime_buf=25000*tbf_buffer_secs,
	                  guarantee=False, core=cores.pop(0)))
	ops.append(TriggeredDumpOp(log=log, osock=osock, iring=tbf_ring,
	                           ntime_gulp=GSIZE, ntime_buf=int(25000*tbf_buffer_secs/2500)*2500,
	                           tuning=tuning, nchan_max=nchan_max,
	                           core=cores.pop(0),
	                           max_bytes_per_sec=tbf_bw_max))
	ops.append(BeamformerOp(log=log, iring=capture_ring, oring=tengine_ring,
	                        tuning=tuning, ntime_gulp=GSIZE,
	                        nchan_max=nchan_max, nbeam_max=nbeam,
	                        core=cores.pop(0), gpu=gpus.pop(0)))
	ops.append(RetransmitOp(log=log, osock=tsock, iring=tengine_ring,
	                        tuning=tuning, ntime_gulp=50,
	                        nbeam_max=nbeam,
	                        core=cores.pop(0)))
	#ops.append(CorrelatorOp(log=log, iring=capture_ring, oring=vis_ring,
	#                        tuning=tuning, ntime_gulp=GSIZE,
	#                        nchan_max=nchan_max,
	#                        core=3 if tuning == 0 else 10, gpu=tuning))
	#ops.append(PacketizeOp(log=log, iring=vis_ring, osock=vsock,
	#                       npkt_gulp=1,
	#                       core=3 if tuning == 0 else 10, gpu=tuning,
	#                       max_bytes_per_sec=cor_bw_max)))

	threads = [threading.Thread(target=op.main) for op in ops]

	log.info("Launching %i thread(s)", len(threads))
	for thread in threads:
		#thread.daemon = True
		thread.start()
	while not shutdown_event.is_set():
		signal.pause()
	log.info("Shutdown, waiting for threads to join")
	for thread in threads:
		thread.join()
	log.info("All done")
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
