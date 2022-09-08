#!/usr/bin/env python

from __future__ import print_function

# Set Backend
import matplotlib
matplotlib.use("Agg") # noqa

# Core Python Includes
import os
import sys
import json
import time
import signal
import logging
import argparse
import threading
import numpy as np
from collections import deque
from scipy.fftpack import fft

from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, FK5
from astropy.constants import c as speed_of_light

import datetime
import ctypes
from . import MCS2 as MCS

# Profiling Includes
import cProfile
import pstats

# Bifrost Includes
import bifrost
import bifrost.affinity
from bifrost.address import Address as BF_Address
from bifrost.udp_socket import UDPSocket as BF_UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, UDPCapture
from bifrost.ring import Ring
from bifrost.unpack import unpack as Unpack
from bifrost.quantize import quantize as Quantize
from bifrost.proclog import ProcLog
from bifrost.libbifrost import bf
from bifrost.fft import Fft
from bifrost.linalg import LinAlg

from bifrost.ndarray import memset_array, copy_array
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()  # noqa

# LWA Software Library Includes
from lsl.reader.ldp import TBNFile, TBFFile
from lsl.common.stations import lwa1, lwasv

#Optimized Bifrost blocks for EPIC
from bifrost.VGrid import VGrid
from bifrost.XGrid import XGrid
from bifrost.aCorr import aCorr

# some py2/3 compatibility
if sys.version_info.major < 3:
    range = xrange  # noqa

# Trigger Processing

TRIGGER_ACTIVE = threading.Event()

DATE_FORMAT = "%Y_%m_%dT%H_%M_%S"


def get_utc_start():
    got_utc_start = False
    while not got_utc_start:
        try:
            with MCS.Communicator() as adp_control:
                utc_start = adp_control.report('UTC_START')
                # Check for valid timestamp
                utc_start_dt = datetime.datetime.strptime(utc_start, DATE_FORMAT)
            got_utc_start = True
        except Exception as ex:
            print(ex)
            time.sleep(0.1)
    return utc_start_dt


# Profiling
def enable_thread_profiling():
    """Monkey-patch Thread.run to enable global profiling.

    Each thread creates a local profiler; statistics are pooled
    to the global stats object on run completion.

    """
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
    """Retreive stats from the thread."""
    stats = getattr(threading.Thread, "stats", None)
    if stats is None:
        raise ValueError(
            "Thread profiling was not enabled, or no threads finished running."
        )
    return stats


# Direct Fourier Transform Matrix
def form_dft_matrix(lmn_vector, antenna_location, antenna_phases, nchan, npol, nstand):
    """Create the DFT Matrix.

    Parameters
    ----------
    lmn_vector : array_like
        Vector coordinates in (l, m, n) space over which the DFT will be performed.
        Has shape (npoints, 3)
    antenna_location : array_like
        Array of atenna locations of the form (nchan, npol, 3, nstand)
    antenna_phases : array_like
        Array of per antenna phases. Has shape (nchan, npol, nstand)
    nchan : int
        Number of channels
    npol : int
        Number of polarizations
    nstand : int
        Number of stands in the array

    Returns
    -------
    normalized DFT matrix for given inputs

    """
    # lm_matrix, shape = [...,2] , where the last dimension is an l/m pair.
    lmn_vector[:, 2] = 1.0 - np.sqrt(
        1.0 - lmn_vector[:, 0] ** 2 - lmn_vector[:, 1] ** 2
    )
    dft_matrix = np.zeros(
        (nchan, npol, lmn_vector.shape[0], nstand), dtype=np.complex64
    )
    # DFT phase factors
    # Both polarisations are at the same physical location, only phases differ.
    dft_matrix[:, :] = np.exp(
        2j * np.pi * (np.dot(lmn_vector, antenna_location[0, 0]))
    )
    # Can put the antenna phases in as well because maths
    dft_matrix *= antenna_phases.transpose([0, 2, 1])[:, :, np.newaxis, :] / nstand

    return dft_matrix


# Frequency-Dependent Locations
def Generate_DFT_Locations(lsl_locs, frequencies, ntime, nchan, npol):
    """
    Generate locations used in DFT transformation.

    Parameters
    ----------
    lsl_locs : np.ndarray
        Array of stand locations. Has shape (3, nstand)
    frequencies : np.ndarray
        Array of frequencies in the observation.
    ntime : int
        Number of times.
    nchan : int
        Number of channels/frequencies.
    npol : int
        Number of polarizations

    Returns
    -------
    Array of DFT locations of shape (nchan, npol, 3, nstand)

    """
    lsl_locs = lsl_locs.T
    lsl_locs = lsl_locs.copy()
    chan_wavelengths = speed_of_light.value / frequencies

    dft_locs = lsl_locs[np.newaxis, np.newaxis, :, :] / chan_wavelengths[:, np.newaxis, np.newaxis, np.newaxis]

    dft_locs = np.broadcast_to(dft_locs, (nchan, npol, 3, lsl_locs.shape[1])).copy()
    return dft_locs


def GenerateLocations(
    lsl_locs, frequencies, ntime, nchan, npol, grid_size=64, grid_resolution=20 / 60.
):
    """
    Generate locations of stands compatible with DFT code.

    Parameters
    ----------
    lsl_locs : np.ndarray
        Array of stand locations. Has shape (3, nstand)
    frequencies : np.ndarray
        Array of frequencies in the observation.
    ntime : int
        Number of times.
    nchan : int
        Number of channels/frequencies.
    npol : int
        Number of polarizations
    grid_size : int
        The desired size of the DFT grid.
    grid_resolution : int
        The gridding resolution in degrees.

    Returns
    -------
    delta
        The sampling length of the DFT or resolution in image space.
    locc
        The lsl locations with shape matching expected DFT input shape and scaled
    sll
        The sampling length of the DFT or resolution in image space.

    """
    delta = (2 * grid_size * np.sin(np.pi * grid_resolution / 360)) ** -1
    chan_wavelengths = speed_of_light.value / frequencies
    sample_grid = chan_wavelengths * delta
    sll = sample_grid[0] / chan_wavelengths[0]
    lsl_locs = lsl_locs.T

    lsl_locsf = lsl_locs[:, np.newaxis, np.newaxis, :] / sample_grid[np.newaxis, np.newaxis, :, np.newaxis]
    lsl_locsf -= np.min(lsl_locsf, axis=3, keepdims=True)

    # Centre locations slightly
    lsl_locsf += (grid_size - np.max(lsl_locsf, axis=3, keepdims=True)) / 2.

    # add ntime axis
    locc = np.broadcast_to(lsl_locsf, (ntime, 3, npol, nchan, lsl_locs.shape[1])).transpose(1, 0, 3, 4, 2).copy()
    return delta, locc, sll


# EPIC
class TBNOfflineCaptureOp(object):
    def __init__(self, log, oring, filename, chan_bw=25000, profile=False, core=-1):
        self.log = log
        self.oring = oring
        self.filename = filename
        self.core = core
        self.profile = profile
        self.chan_bw = 25000

        self.bind_proclog = ProcLog(type(self).__name__ + "/bind")
        self.out_proclog = ProcLog(type(self).__name__ + "/out")
        self.size_proclog = ProcLog(type(self).__name__ + "/size")
        self.sequence_proclog = ProcLog(type(self).__name__ + "/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__ + "/perf")
        self.out_proclog.update({"nring": 1, "ring0": self.oring.name})

        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def main(self):
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        self.bind_proclog.update(
            {"ncore": 1, "core0": bifrost.affinity.get_core()}
        )

        idf = TBNFile(self.filename)
        cfreq = idf.get_info("freq1")
        srate = idf.get_info("sample_rate")
        tInt, tStart, data = idf.read(0.1, time_in_samples=True)

        # Setup the ring metadata and gulp sizes
        ntime = data.shape[1]
        nstand, npol = data.shape[0] // 2, 2
        oshape = (ntime, nstand, npol)
        ogulp_size = ntime * nstand * npol * 8  # complex64
        self.oring.resize(ogulp_size, buffer_factor=10)

        self.size_proclog.update({"nseq_per_gulp": ntime})

        # Build the initial ring header
        ohdr = {}
        ohdr["time_tag"] = tStart
        ohdr["seq0"] = 0
        ohdr["chan0"] = int((cfreq - srate / 2) / self.chan_bw)
        ohdr["nchan"] = 1
        ohdr["cfreq"] = cfreq
        ohdr["bw"] = srate
        ohdr["nstand"] = nstand
        ohdr["npol"] = npol
        ohdr["nbit"] = 8
        ohdr["complex"] = True
        ohdr["axes"] = "time,stand,pol"
        ohdr_str = json.dumps(ohdr)

        # Fill the ring using the same data over and over again
        with self.oring.begin_writing() as oring:
            with oring.begin_sequence(time_tag=tStart, header=ohdr_str) as oseq:
                prev_time = time.time()
                if self.profile:
                    spani = 0
                while not self.shutdown_event.is_set():
                    # Get the current section to use
                    try:
                        _, _, next_data = idf.read(0.1, time_in_samples=True)
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

                        # Setup and load
                        idata = data

                        odata = ospan.data_view(np.complex64).reshape(oshape)
                        # Transpose and reshape to time by stand by pol
                        idata = idata.transpose((1, 0))
                        idata = idata.reshape((ntime, nstand, npol))

                        # Save
                        odata[...] = idata

                    data = next_data

                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update(
                        {
                            "acquire_time": acquire_time,
                            "reserve_time": reserve_time,
                            "process_time": process_time,
                        }
                    )
                    if self.profile:
                        spani += 1
                        if spani >= 10:
                            sys.exit()
                            break
        print("TBNFillerOp - Done")
        os.kill(os.getpid(), signal.SIGTERM)


class FDomainOp(object):
    def __init__(
        self,
        log,
        iring,
        oring,
        ntime_gulp=2500,
        nchan_out=1,
        profile=False,
        core=-1,
        gpu=-1
    ):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.nchan_out = nchan_out
        self.core = core
        self.gpu = gpu
        self.profile = profile

        self.nchan_out = nchan_out

        self.bind_proclog = ProcLog(type(self).__name__ + "/bind")
        self.in_proclog = ProcLog(type(self).__name__ + "/in")
        self.out_proclog = ProcLog(type(self).__name__ + "/out")
        self.size_proclog = ProcLog(type(self).__name__ + "/size")
        self.sequence_proclog = ProcLog(type(self).__name__ + "/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__ + "/perf")

        self.in_proclog.update({"nring": 1, "ring0": self.iring.name})
        self.out_proclog.update({"nring": 1, "ring0": self.oring.name})
        self.size_proclog.update({"nseq_per_gulp": self.ntime_gulp})
        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def main(self):
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update(
            {
                "ncore": 1,
                "core0": bifrost.affinity.get_core(),
                "ngpu": 1,
                "gpu0": BFGetGPU(),
            }
        )

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=True):
                ihdr = json.loads(iseq.header.tobytes())

                self.sequence_proclog.update(ihdr)
                print("FDomainOp: Config - %s" % ihdr)

                # Setup the ring metadata and gulp sizes
                nchan = self.nchan_out
                nstand = ihdr["nstand"]
                npol = ihdr["npol"]

                igulp_size = self.ntime_gulp * 1 * nstand * npol * 8  # complex64
                ishape = (self.ntime_gulp // nchan, nchan, nstand, npol)
                ogulp_size = self.ntime_gulp * 1 * nstand * npol * 1  # ci4
                oshape = (self.ntime_gulp // nchan, nchan, nstand, npol)
                # self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size, buffer_factor=5)

                # Set the output header
                ohdr = ihdr.copy()
                ohdr["nchan"] = nchan
                ohdr["nbit"] = 4
                ohdr["axes"] = "time,chan,stand,pol"
                ohdr_str = json.dumps(ohdr)

                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    print("FDomain Out Sequence!")
                    iseq_spans = iseq.read(igulp_size)
                    while not self.iring.writing_ended():
                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue  # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time

                            if self.profile:
                                spani = 0

                            with oseq.reserve(ogulp_size) as ospan:
                                curr_time = time.time()
                                reserve_time = curr_time - prev_time
                                prev_time = curr_time

                                # Setup and load
                                idata = ispan.data_view(np.complex64).reshape(ishape)

                                odata = ospan.data_view(np.int8).reshape(oshape)

                                # FFT, shift, and phase
                                fdata = fft(idata, axis=1)
                                fdata = np.fft.fftshift(fdata, axes=1)
                                fdata = bifrost.ndarray(fdata, space="system")

                                # Quantization
                                try:
                                    Quantize(fdata, qdata, scale=1. / np.sqrt(nchan))
                                except NameError:
                                    qdata = bifrost.ndarray(shape=fdata.shape, native=False, dtype="ci4")
                                    Quantize(fdata, qdata, scale=1. / np.sqrt(nchan))

                                # Save
                                odata[...] = qdata.copy(space="cuda_host").view(np.int8).reshape(oshape)

                                if self.profile:
                                    spani += 1
                                    if spani >= 10:
                                        sys.exit()
                                        break

                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update(
                                {
                                    "acquire_time": acquire_time,
                                    "reserve_time": reserve_time,
                                    "process_time": process_time,
                                }
                            )
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

        self.bind_proclog = ProcLog(type(self).__name__ + "/bind")
        self.out_proclog = ProcLog(type(self).__name__ + "/out")
        self.size_proclog = ProcLog(type(self).__name__ + "/size")
        self.sequence_proclog = ProcLog(type(self).__name__ + "/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__ + "/perf")
        self.out_proclog.update({"nring": 1, "ring0": self.oring.name})

        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def main(self):
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        self.bind_proclog.update(
            {"ncore": 1, "core0": bifrost.affinity.get_core()}
        )

        idf = TBFFile(self.filename)
        srate = idf.get_info("sample_rate")
        chans = np.round(idf.get_info("freq1") / srate).astype(np.int32)
        chan0 = int(chans[0])
        nchan = len(chans)
        tInt, tStart, data = idf.read(0.1, time_in_samples=True)

        # Setup the ring metadata and gulp sizes
        ntime = data.shape[2]
        nstand, npol = data.shape[0] / 2, 2
        oshape = (ntime, nchan, nstand, npol)
        ogulp_size = ntime * nchan * nstand * npol * 1  # ci4
        self.oring.resize(ogulp_size)

        self.size_proclog.update({"nseq_per_gulp": ntime})

        # Build the initial ring header
        ohdr = {}
        ohdr["time_tag"] = tStart
        ohdr["seq0"] = 0
        ohdr["chan0"] = chan0
        ohdr["nchan"] = nchan
        ohdr["cfreq"] = (chan0 + 0.5 * (nchan - 1)) * srate
        ohdr["bw"] = nchan * srate
        ohdr["nstand"] = nstand
        ohdr["npol"] = npol
        ohdr["nbit"] = 4
        ohdr["complex"] = True
        ohdr["axes"] = "time,chan,stand,pol"
        ohdr_str = json.dumps(ohdr)

        # Fill the ring using the same data over and over again
        with self.oring.begin_writing() as oring:
            with oring.begin_sequence(time_tag=tStart, header=ohdr_str) as oseq:
                prev_time = time.time()
                if self.profile:
                    spani = 0
                while not self.shutdown_event.is_set():
                    # Get the current section to use
                    try:
                        _, _, next_data = idf.read(0.1, time_in_samples=True)
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

                        # Setup and load
                        idata = data

                        odata = ospan.data_view(np.int8).reshape(oshape)

                        # Transpose and reshape to time by channel by stand by pol
                        idata = idata.transpose((2, 1, 0))
                        idata = idata.reshape((ntime, nchan, nstand, npol))
                        idata = idata.copy()

                        # Quantization
                        try:
                            Quantize(idata, qdata, scale=1. / np.sqrt(nchan))
                        except NameError:
                            qdata = bifrost.ndarray(shape=idata.shape, native=False, dtype="ci4")
                            Quantize(idata, qdata, scale=1.0)

                        # Save
                        odata[...] = qdata.copy(space="cuda_host").view(np.int8).reshape(oshape)

                    data = next_data

                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update(
                        {
                            "acquire_time": acquire_time,
                            "reserve_time": reserve_time,
                            "process_time": process_time,
                        }
                    )
                    if self.profile:
                        spani += 1
                        if spani >= 10:
                            sys.exit()
                            break
        print("TBFFillerOp - Done")
        os.kill(os.getpid(), signal.SIGTERM)


# For when we don't need to care about doing the F-Engine ourself.
# TODO: Implement this come implementation time...
FS = 196.0e6
CHAN_BW = 25.0e3
ADP_EPOCH = datetime.datetime(1970, 1, 1)


class FEngineCaptureOp(object):
    """Receive Fourier Spectra from LWA FPGA."""

    def __init__(self, log, *args, **kwargs):
        self.log = log
        self.args = args
        self.kwargs = kwargs
        self.utc_start = self.kwargs["utc_start"]
        del self.kwargs["utc_start"]
        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def seq_callback(
        self, seq0, chan0, nchan, nsrc, time_tag_ptr, hdr_ptr, hdr_size_ptr
    ):
        timestamp0 = int((self.utc_start - ADP_EPOCH).total_seconds())
        time_tag0 = timestamp0 * int(FS)
        time_tag = time_tag0 + seq0 * (int(FS) // int(CHAN_BW))
        print("++++++++++++++++ seq0     =", seq0)
        print("                 time_tag =", time_tag)
        time_tag_ptr[0] = time_tag
        hdr = {
            "time_tag": time_tag,
            "seq0": seq0,
            "chan0": chan0,
            "nchan": nchan,
            "cfreq": (chan0 + 0.5 * (nchan - 1)) * CHAN_BW,
            "bw": nchan * CHAN_BW,
            "nstand": nsrc * 16,
            # 'stand0':   src0*16, # TODO: Pass src0 to the callback too(?)
            "npol": 2,
            "complex": True,
            "nbit": 4,
            "axes": "time,chan,stand,pol",
        }
        print("******** CFREQ:", hdr["cfreq"])
        hdr_str = json.dumps(hdr).encode()
        # TODO: Can't pad with NULL because returned as C-string
        # hdr_str = json.dumps(hdr).ljust(4096, '\0')
        # hdr_str = json.dumps(hdr).ljust(4096, ' ')
        self.header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0] = ctypes.cast(self.header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0

    def main(self):
        seq_callback = PacketCaptureCallback()

        seq_callback.set_chips(self.seq_callback)
        with UDPCapture(*self.args,
                        sequence_callback=seq_callback,
                        **self.kwargs) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
        del capture


class DecimationOp(object):
    def __init__(
        self,
        log,
        iring,
        oring,
        ntime_gulp=2500,
        nchan_out=1,
        npol_out=2,
        guarantee=True,
        core=-1,
        gpu=-1,
    ):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.nchan_out = nchan_out
        self.npol_out = npol_out
        self.guarantee = guarantee
        self.core = core
        self.gpu = gpu

        self.bind_proclog = ProcLog(type(self).__name__ + "/bind")
        self.in_proclog = ProcLog(type(self).__name__ + "/in")
        self.out_proclog = ProcLog(type(self).__name__ + "/out")
        self.size_proclog = ProcLog(type(self).__name__ + "/size")
        self.sequence_proclog = ProcLog(type(self).__name__ + "/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__ + "/perf")

        self.in_proclog.update({"nring": 1, "ring0": self.iring.name})
        self.out_proclog.update({"nring": 1, "ring0": self.oring.name})
        self.size_proclog.update({"nseq_per_gulp": self.ntime_gulp})

    def main(self):
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update(
            {
                "ncore": 1,
                "core0": bifrost.affinity.get_core(),
                "ngpu": 1,
                "gpu0": BFGetGPU(),
            }
        )

        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tobytes())

                self.sequence_proclog.update(ihdr)

                self.log.info("Decimation: Start of new sequence: %s", str(ihdr))

                nchan = ihdr["nchan"]
                nstand = ihdr["nstand"]
                npol = ihdr["npol"]
                chan0 = ihdr["chan0"]

                igulp_size = self.ntime_gulp * nchan * nstand * npol * 1  # ci4
                ishape = (self.ntime_gulp, nchan, nstand, npol)
                ogulp_size = self.ntime_gulp * self.nchan_out * nstand * self.npol_out * 1  # ci4
                oshape = (self.ntime_gulp, self.nchan_out, nstand, self.npol_out)
                self.iring.resize(igulp_size, buffer_factor= 5)
                self.oring.resize(ogulp_size, buffer_factor= 10)  # , obuf_size)

                do_truncate = True
                act_chan_bw = CHAN_BW
                if nchan % self.nchan_out == 0:
                    do_truncate = False
                    act_chan_bw = CHAN_BW * (nchan // self.nchan_out)
                    chan0 = chan0 + 0.5 * (nchan // self.nchan_out - 1)
                    self.log.info("Decimation: Running in averaging mode")
                else:
                    self.log.info("Decimation: Running in truncation mode")
                self.log.info("Decimation: Channel bandwidth is %.3f kHz", act_chan_bw/1e3)
                
                ohdr = ihdr.copy()
                ohdr["chan0"] = chan0
                ohdr["nchan"] = self.nchan_out
                ohdr["npol"] = self.npol_out
                ohdr["cfreq"] = chan0 * CHAN_BW + 0.5 * (self.nchan_out - 1) * act_chan_bw
                ohdr["bw"] = self.nchan_out * act_chan_bw
                ohdr_str = json.dumps(ohdr)

                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    for ispan in iseq.read(igulp_size):
                        if ispan.size < igulp_size:
                            continue  # Ignore final gulp
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time

                        with oseq.reserve(ogulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time

                            idata = ispan.data_view(np.uint8).reshape(ishape)
                            odata = ospan.data_view(np.uint8).reshape(oshape)

                            if do_truncate:
                                sdata = idata[:, :self.nchan_out, :, :]
                                if self.npol_out != npol:
                                    sdata = sdata[:, :, :, :self.npol_out]
                            else:
                                try:
                                    copy_array(gdata, idata)
                                except NameError:
                                    gdata = idata.copy(space='cuda')
                                    adata = bifrost.zeros(
                                        shape=(self.ntime_gulp, self.nchan_out, nstand, self.npol_out),
                                        dtype='u8',
                                        space='cuda'
                                    )
                                
                                bifrost.map("""
                                signed char re, im;
                                #pragma unroll
                                for(int l=0; l<{npol_out}; l++) {{
                                    re = ((signed char) (b(i,j*{navg},k,l) & 0xF0)) / 16;
                                    im = ((signed char) ((b(i,j*{navg},k,l) & 0x0F) << 4)) / 16;
                                    #pragma unroll
                                    for(int m=1; m<{navg}; m++) {{
                                        re += ((signed char) (b(i,j*{navg}+m,k,l) & 0xF0)) / 16;
                                        im += ((signed char) ((b(i,j*{navg}+m,k,l) & 0x0F) << 4)) / 16;
                                    }}
                                    re /= {navg};
                                    im /= {navg};
                                    a(i,j,k,l) = ((re * 16) & 0xF0) | ((im * 16) >> 4);
                                }}
                                """.format(npol_out=self.npol_out, navg=nchan // self.nchan_out),
                                {'a': adata, 'b': gdata},
                                axis_names=('i', 'j', 'k'),
                                shape=adata.shape[:3])
                                
                                try:
                                    copy_array(sdata, adata)
                                except NameError:
                                    sdata = adata.copy(space='system')

                            odata[...] = sdata

                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update(
                                {
                                    "acquire_time": acquire_time,
                                    "reserve_time": reserve_time,
                                    "process_time": process_time,
                                }
                            )

                if not do_truncate:
                    try:
                        del cdata
                        del sdata
                    except NameError:
                        pass



class CalibrationOp(object):
    def __init__(self, log, iring, oring, *args, **kwargs):
        pass


class MOFFCorrelatorOp(object):
    def __init__(
        self,
        log,
        iring,
        oring,
        station,
        grid_size,
        grid_resolution,
        ntime_gulp=2500,
        accumulation_time=10000,
        core=-1,
        gpu=-1,
        remove_autocorrs=False,
        benchmark=False,
        profile=False,
        *args,
        **kwargs
    ):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.accumulation_time = accumulation_time

        self.station = station
        locations = np.array([(ant.stand.x, ant.stand.y, ant.stand.z) for ant in self.station.antennas[::2]])
        if self.station == lwasv:
            locations[[i for i, a in enumerate(self.station.antennas[::2]) if a.stand.id == 256], :] = 0.0
        elif self.station == lwa1:
            locations[[i for i, a in enumerate(self.station.antennas[::2]) if a.stand.id in (35, 257, 258, 259, 260)], :] = 0.0
        self.locations = locations
        
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
            
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution

        self.core = core
        self.gpu = gpu
        self.remove_autocorrs = remove_autocorrs
        self.benchmark = benchmark
        self.newflag = True
        self.profile = profile

        self.bind_proclog = ProcLog(type(self).__name__ + "/bind")
        self.in_proclog = ProcLog(type(self).__name__ + "/in")
        self.out_proclog = ProcLog(type(self).__name__ + "/out")
        self.size_proclog = ProcLog(type(self).__name__ + "/size")
        self.sequence_proclog = ProcLog(type(self).__name__ + "/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__ + "/perf")

        self.in_proclog.update({"nring": 1, "ring0": self.iring.name})
        self.out_proclog.update({"nring": 1, "ring0": self.oring.name})
        self.size_proclog.update({"nseq_per_gulp": self.ntime_gulp})

        self.ant_extent = 1

        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def main(self):
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update(
            {
                "ncore": 1,
                "core0": bifrost.affinity.get_core(),
                "ngpu": 1,
                "gpu0": BFGetGPU(),
            }
        )

        runtime_history = deque([], 50)
        accum = 0
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=True):
                ihdr = json.loads(iseq.header.tobytes())
                self.sequence_proclog.update(ihdr)
                self.log.info("MOFFCorrelatorOp: Config - %s" % ihdr)
                chan0 = ihdr["chan0"]
                nchan = ihdr["nchan"]
                bw = ihdr["bw"]
                act_chan_bw = bw / nchan
                nstand = ihdr["nstand"]
                npol = ihdr["npol"]
                self.newflag = True
                accum = 0

                igulp_size = self.ntime_gulp * nchan * nstand * npol * 1  # ci4
                itshape = (self.ntime_gulp, nchan, nstand, npol)

                freq = chan0 * CHAN_BW + np.arange(nchan) * act_chan_bw
                locname = "locations_%s_%i_%i_%i_%i_%i_%i_%i_%.6f.npz" % (self.station.name, chan0, self.ntime_gulp, nchan, nstand, npol, self.ant_extent, self.grid_size, self.grid_resolution)
                locname = os.path.join(self.cache_dir, locname)
                try:
                    loc_data = np.load(locname)
                    sampling_length = loc_data['sampling_length'].item()
                    locs = loc_data['locs'][...]
                    sll = loc_data['sll'].item()
                except OSError:
                    sampling_length, locs, sll = GenerateLocations(
                        self.locations,
                        freq,
                        self.ntime_gulp,
                        nchan,
                        npol,
                        grid_size=self.grid_size,
                        grid_resolution=self.grid_resolution,
                    )
                    np.savez(locname, sampling_length=sampling_length, locs=locs, sll=sll)
                try:
                    copy_array(self.locs, bifrost.ndarray(locs.astype(np.int32)))
                except AttributeError:
                    self.locs = bifrost.ndarray(locs.astype(np.int32), space="cuda")

                ohdr = ihdr.copy()
                ohdr["nbit"] = 64

                ms_per_gulp = 1e3 * self.ntime_gulp / CHAN_BW
                new_accumulation_time = np.ceil(self.accumulation_time / ms_per_gulp) * ms_per_gulp
                if new_accumulation_time != self.accumulation_time:
                    self.log.warning(
                        "Adjusting accumulation time from %.3f ms to %.3f ms",
                        self.accumulation_time,
                        new_accumulation_time,
                    )
                    self.accumulation_time = new_accumulation_time

                ohdr["npol"] = npol ** 2  # Because of cross multiplying shenanigans
                ohdr["grid_size_x"] = self.grid_size
                ohdr["grid_size_y"] = self.grid_size
                ohdr["axes"] = "time,chan,pol,gridy,gridx"
                ohdr["sampling_length_x"] = sampling_length
                ohdr["sampling_length_y"] = sampling_length
                ohdr["accumulation_time"] = self.accumulation_time
                ohdr["FS"] = FS
                ohdr["latitude"] = self.station.lat * 180. / np.pi
                ohdr["longitude"] = self.station.lon * 180. / np.pi
                ohdr["telescope"] = self.station.name.upper()
                ohdr["data_units"] = "UNCALIB"
                if ohdr["npol"] == 1:
                    ohdr["pols"] = ["xx"]
                elif ohdr["npol"] == 2:
                    ohdr["pols"] = ["xx", "yy"]
                elif ohdr["npol"] == 4:
                    ohdr["pols"] = ["xx", "xy", "yx", "yy"]
                else:
                    raise ValueError(
                        "Cannot write fits file without knowing polarization list"
                    )
                ohdr_str = json.dumps(ohdr)

                # Setup the kernels to include phasing terms for zenith
                # Phases are Ntime x Nchan x Nstand x Npol x extent x extent
                phasename = "phases_%s_%i_%i_%i_%i_%i_%i.npy" % (self.station.name, chan0, self.ntime_gulp, nchan, nstand, npol, self.ant_extent)
                phasename = os.path.join(self.cache_dir, phasename)
                try:
                    phases = np.load(phasename)
                except OSError:
                    freq.shape += (1, 1)
                    phases = np.zeros(
                        (self.ntime_gulp, nchan, nstand, npol, self.ant_extent, self.ant_extent),
                        dtype=np.complex64
                    )
                    for i in range(nstand):
                        # X
                        a = self.station.antennas[2 * i + 0]
                        delay = a.cable.delay(freq) - a.stand.z / speed_of_light.value
                        phases[:, :, i, 0, :, :] = np.exp(2j * np.pi * freq * delay)
                        phases[:, :, i, 0, :, :] /= np.sqrt(a.cable.gain(freq))
                        if npol == 2:
                            # Y
                            a = self.station.antennas[2 * i + 1]
                            delay = a.cable.delay(freq) - a.stand.z / speed_of_light.value
                            phases[:, :, i, 1, :, :] = np.exp(2j * np.pi * freq * delay)
                            phases[:, :, i, 1, :, :] /= np.sqrt(a.cable.gain(freq))
                        # Explicit bad and suspect antenna masking - this will
                        # mask an entire stand if either pol is bad
                        if (
                            self.station.antennas[2 * i + 0].combined_status < 33
                            or self.station.antennas[2 * i + 1].combined_status < 33
                        ):
                            phases[:, :, i, :, :, :] = 0.0
                        # Explicit outrigger masking - we probably want to do
                        # away with this at some point
                        if (
                            (self.station == lwasv and a.stand.id == 256)
                            or (self.station == lwa1 and a.stand.id in (35, 257, 258, 259, 260))
                        ):
                            phases[:, :, i, :, :, :] = 0.0
                    phases = phases.conj()
                    np.save(phasename, phases)
                phases = bifrost.ndarray(phases)
                
                try:
                    copy_array(gphases, phases)
                except NameError:
                    gphases = phases.copy(space="cuda")

                oshape = (1, nchan, npol ** 2, self.grid_size, self.grid_size)
                ogulp_size = nchan * npol ** 2 * self.grid_size * self.grid_size * 8
                self.iring.resize(igulp_size, buffer_factor=10)
                self.oring.resize(ogulp_size, buffer_factor=10)
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    iseq_spans = iseq.read(igulp_size)
                    while not self.iring.writing_ended():
                        reset_sequence = False

                        if self.profile:
                            spani = 0

                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue  # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time

                            if self.benchmark is True:
                                print(" ------------------ ")

                            # Correlator
                            # Setup and load
                            idata = ispan.data_view(np.uint8).reshape(itshape)
                            # Fix the type
                            udata = bifrost.ndarray(
                                shape=itshape,
                                dtype="ci4",
                                native=False,
                                buffer=idata.ctypes.data,
                            )

                            if self.benchmark is True:
                                time1 = time.time()

                            udata = udata.copy(space="cuda")
                            if self.benchmark is True:
                                time1a = time.time()
                                print("  Input copy time: %f" % (time1a - time1))

                            # Make sure we have a place to put the gridded data
                            # Gridded Antennas
                            try:
                                gdata = gdata.reshape(
                                    self.ntime_gulp, nchan, npol, self.grid_size, self.grid_size
                                )
                                memset_array(gdata, 0)
                            except NameError:
                                gdata = bifrost.zeros(
                                    shape=(self.ntime_gulp, nchan, npol, self.grid_size, self.grid_size),
                                    dtype=np.complex64,
                                    space="cuda",
                                )

                            # Grid the Antennas
                            if self.benchmark is True:
                                timeg1 = time.time()
                            try:
                                
                                bf_vgrid.execute(udata, gdata)
                            except NameError:
                                
                                bf_vgrid = VGrid()
                                bf_vgrid.init(self.locs, gphases, self.grid_size, polmajor=False)
                                bf_vgrid.execute(udata, gdata)
                            gdata = gdata.reshape(
                                self.ntime_gulp * nchan * npol, self.grid_size, self.grid_size
                            )
                            if self.benchmark is True:
                                timeg2 = time.time()
                                print("  Grid time: %f" % (timeg2 - timeg1))

                            # Inverse transform

                            if self.benchmark is True:
                                timefft1 = time.time()
                            try:

                                bf_fft.execute(gdata, gdata, inverse=True)
                            except NameError:

                                bf_fft = Fft()
                                bf_fft.init(gdata, gdata, axes=(1, 2))
                                bf_fft.execute(gdata, gdata, inverse=True)
                            gdata = gdata.reshape(
                                1, self.ntime_gulp, nchan, npol, self.grid_size, self.grid_size
                            )
                            if self.benchmark is True:
                                timefft2 = time.time()
                                print("  FFT time: %f" % (timefft2 - timefft1))

                            if self.newflag is True:
                                try:
                                    crosspol = crosspol.reshape(
                                        self.ntime_gulp, nchan, npol ** 2, self.grid_size, self.grid_size
                                    )
                                    accumulated_image = accumulated_image.reshape(
                                        1, nchan, npol ** 2, self.grid_size, self.grid_size
                                    )
                                    memset_array(crosspol, 0)
                                    memset_array(accumulated_image, 0)

                                except NameError:
                                    crosspol = bifrost.zeros(
                                        shape=(self.ntime_gulp, nchan, npol ** 2, self.grid_size, self.grid_size),
                                        dtype=np.complex64,
                                        space="cuda",
                                    )
                                    accumulated_image = bifrost.zeros(
                                        shape=(1, nchan, npol ** 2, self.grid_size, self.grid_size),
                                        dtype=np.complex64,
                                        space="cuda",
                                    )
                                self.newflag = False

                            if self.remove_autocorrs is True:

                                # Setup everything for the autocorrelation calculation.
                                try:
                                    # If one isn't allocated, then none of them are.
                                    autocorrs = autocorrs.reshape(
                                        self.ntime_gulp, nchan, nstand, npol ** 2
                                    )
                                    autocorr_g = autocorr_g.reshape(
                                        nchan * npol ** 2, self.grid_size, self.grid_size
                                    )
                                except NameError:
                                    autocorrs = bifrost.ndarray(
                                        shape=(self.ntime_gulp, nchan, nstand, npol ** 2),
                                        dtype=np.complex64,
                                        space="cuda",
                                    )
                                    autocorrs_av = bifrost.zeros(
                                        shape=(1, nchan, nstand, npol ** 2),
                                        dtype=np.complex64,
                                        space="cuda",
                                    )
                                    autocorr_g = bifrost.zeros(
                                        shape=(1, nchan, npol ** 2, self.grid_size, self.grid_size),
                                        dtype=np.complex64,
                                        space="cuda",
                                    )
                                    autocorr_lo = bifrost.ndarray(
                                        np.ones(
                                            shape=(3, 1, nchan, nstand, npol ** 2),
                                            dtype=np.int32
                                        ) * self.grid_size // 2,
                                        space="cuda",
                                    )
                                    autocorr_il = bifrost.ndarray(
                                        np.ones(
                                            shape=(1, nchan, nstand, npol ** 2, self.ant_extent, self.ant_extent),
                                            dtype=np.complex64
                                        ),
                                        space="cuda",
                                    )

                                try:
                                     bf_auto.execute(udata, autocorrs)
                                except NameError:
                                     bf_auto = aCorr()
                                     bf_auto.init(self.locs, polmajor=False)
                                     bf_auto.execute(udata, autocorrs)
                                autocorrs = autocorrs.reshape(
                                    self.ntime_gulp, nchan, nstand, npol ** 2
                                )

                            try:
                                
                                bf_gmul.execute(gdata, crosspol)
                            except NameError:
                                
                                bf_gmul = XGrid()
                                bf_gmul.init(self.grid_size, polmajor=False)
                                bf_gmul.execute(gdata, crosspol)
                            crosspol = crosspol.reshape(
                                self.ntime_gulp, nchan, npol ** 2, self.grid_size, self.grid_size
                            )
                            gdata = gdata.reshape(
                                1, self.ntime_gulp, nchan, npol, self.grid_size, self.grid_size
                            )

                            # Increment
                            accum += 1e3 * self.ntime_gulp / CHAN_BW

                            if accum >= self.accumulation_time:

                                bifrost.reduce(crosspol, accumulated_image, op="sum")
                                if self.remove_autocorrs is True:
                                    # Reduce along time axis.
                                    bifrost.reduce(autocorrs, autocorrs_av, op="sum")
                                    # Grid the autocorrelations.
                                    autocorr_g = autocorr_g.reshape(
                                        1, nchan, npol ** 2, self.grid_size, self.grid_size
                                    )
                                    #try:
                                    #    bf_romein_autocorr.execute(autocorrs_av, autocorr_g)
                                    #except NameError:
                                    #    bf_romein_autocorr = Romein()
                                    #    bf_romein_autocorr.init(
                                    #        autocorr_lo, autocorr_il, self.grid_size, polmajor=False
                                    #    )
                                    #    bf_romein_autocorr.execute(autocorrs_av, autocorr_g)
                                    try:
                                        bf_vgrid_autocorr.execute(autocorrs_av, autocorr_g)
                                    except NameError:
                                        bf_vgrid_autocorr = VGrid()
                                        bf_vgrid_autocorr.init(
                                            autocorr_lo, autocorr_il, self.grid_size, polmajor=False
                                        )
                                        bf_vgrid_autocorr.execute(autocorrs_av, autocorr_g)
                                    autocorr_g = autocorr_g.reshape(1 * nchan * npol ** 2, self.grid_size, self.grid_size)
                                    # autocorr_g = romein_float(autocorrs_av,autocorr_g,autocorr_il,autocorr_lx,autocorr_ly,autocorr_lz,self.ant_extent,self.grid_size,nstand,nchan*npol**2)
                                    # Inverse FFT
                                    try:
                                        ac_fft.execute(autocorr_g, autocorr_g, inverse=True)
                                    except NameError:
                                        ac_fft = Fft()
                                        ac_fft.init(autocorr_g, autocorr_g, axes=(1, 2), apply_fftshift=True)
                                        ac_fft.execute(autocorr_g, autocorr_g, inverse=True)

                                    accumulated_image = accumulated_image.reshape(nchan, npol ** 2, self.grid_size, self.grid_size)
                                    autocorr_g = autocorr_g.reshape(nchan, npol ** 2, self.grid_size, self.grid_size)
                                    bifrost.map(
                                        "a(i,j,k,l) -= b(i,j,k,l)",
                                        {"a": accumulated_image, "b": autocorr_g},
                                        axis_names=("i", "j", "k", "l"),
                                        shape=(nchan, npol ** 2, self.grid_size, self.grid_size),
                                    )

                                curr_time = time.time()
                                process_time = curr_time - prev_time
                                prev_time = curr_time

                                with oseq.reserve(ogulp_size) as ospan:
                                    odata = ospan.data_view(np.complex64).reshape(oshape)
                                    accumulated_image = accumulated_image.reshape(oshape)
                                    odata[...] = accumulated_image
                                    bifrost.device.stream_synchronize()

                                curr_time = time.time()
                                reserve_time = curr_time - prev_time
                                prev_time = curr_time

                                self.newflag = True
                                accum = 0

                                if self.remove_autocorrs is True:
                                    autocorr_g = autocorr_g.reshape(oshape)
                                    memset_array(autocorr_g, 0)
                                    memset_array(autocorrs, 0)
                                    memset_array(autocorrs_av, 0)

                            else:
                                process_time = 0.0
                                reserve_time = 0.0

                            curr_time = time.time()
                            process_time += curr_time - prev_time
                            prev_time = curr_time

                            # TODO: Autocorrs using Romein??
                            # Output for gridded electric fields.
                            # gdata = gdata.reshape(self.ntime_gulp,nchan,2,self.grid_size,self.grid_size)
                            # image/autos, time, chan, pol, gridx, grid.
                            # accumulated_image = accumulated_image.reshape(oshape)

                            if self.benchmark is True:
                                time2 = time.time()
                                print("-> GPU Time Taken: %f" % (time2 - time1))

                                runtime_history.append(time2 - time1)
                                print("-> Average GPU Time Taken: %f (%i samples)" % (1.0 * sum(runtime_history) / len(runtime_history), len(runtime_history)))
                            if self.profile:
                                spani += 1
                                if spani >= 10:
                                    sys.exit()
                                    break

                            self.perf_proclog.update(
                                {
                                    "acquire_time": acquire_time,
                                    "reserve_time": reserve_time,
                                    "process_time": process_time,
                                }
                            )

                        # Reset to move on to the next input sequence?
                        if not reset_sequence:
                            break


# An alternative correlation step which uses a direct fourier transform.
# This makes use of the DFT as a linear operator, and the high locality
# of a matrix multiplication to form sky images with perfect wide field correction.
class MOFF_DFT_CorrelatorOp(object):
    def __init__(
        self,
        log,
        iring,
        oring,
        station,
        skymodes=64,
        ntime_gulp=2500,
        accumulation_time=10000,
        core=-1,
        gpu=-1,
        benchmark=False,
        profile=False,
        *args,
        **kwargs
    ):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.accumulation_time = accumulation_time

        # Setup Antennas
        self.station = station
        locations = np.array([(ant.stand.x, ant.stand.y, ant.stand.z) for ant in self.station.antennas[::2]])
        #if self.station == lwasv:
        #    locations[[i for i, a in enumerate(self.station.antennas[::2]) if a.stand.id == 256], :] = 0.0
        #elif self.station == lwa1:
        #    locations[[i for i, a in enumerate(self.station.antennas[::2]) if a.stand.id in (35, 257, 258, 259, 260)], :] = 0.0
        self.locations = locations

        # LinAlg

        self.LinAlgObj = LinAlg()

        # Setup Direct Fourier Transform Matrix
        self.dftm = None
        self.skymodes1d = skymodes
        self.skymodes = skymodes ** 2
        self.core = core
        self.gpu = gpu
        self.benchmark = benchmark
        self.newflag = True
        self.profile = profile

        self.bind_proclog = ProcLog(type(self).__name__ + "/bind")
        self.in_proclog = ProcLog(type(self).__name__ + "/in")
        self.out_proclog = ProcLog(type(self).__name__ + "/out")
        self.size_proclog = ProcLog(type(self).__name__ + "/size")
        self.sequence_proclog = ProcLog(type(self).__name__ + "/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__ + "/perf")

        self.in_proclog.update({"nring": 1, "ring0": self.iring.name})
        self.out_proclog.update({"nring": 1, "ring0": self.oring.name})
        self.size_proclog.update({"nseq_per_gulp": self.ntime_gulp})

        self.ant_extent = 1

        self.shutdown_event = threading.Event()

    def shutdown(self):
        self.shutdown_event.set()

    def main(self):
        if self.core != -1:
            bifrost.affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update(
            {
                "ncore": 1,
                "core0": bifrost.affinity.get_core(),
                "ngpu": 1,
                "gpu0": BFGetGPU(),
            }
        )

        runtime_history = deque([], 50)
        accum = 0
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=True):
                ihdr = json.loads(iseq.header.tobytes())
                self.sequence_proclog.update(ihdr)
                self.log.info("MOFFCorrelatorOp: Config - %s" % ihdr)
                chan0 = ihdr["chan0"]
                nchan = ihdr["nchan"]
                nstand = ihdr["nstand"]
                npol = ihdr["npol"]
                bw = ihdr["bw"]
                act_chan_bw = bw / nchan
                self.newflag = True
                accum = 0

                igulp_size = self.ntime_gulp * nchan * nstand * npol * 1  # ci4
                itshape = (self.ntime_gulp, nchan, nstand, npol)

                # Sample locations at right u/v/w values
                freq = chan0 * CHAN_BW + np.arange(nchan) * act_chan_bw
                locs = Generate_DFT_Locations(
                    self.locations, freq, self.ntime_gulp, nchan, npol
                )

                try:
                    copy_array(self.locs, bifrost.ndarray(locs.astype(np.int32)))
                except AttributeError:
                    self.locs = bifrost.ndarray(locs.astype(np.int32), space="cuda")

                ohdr = ihdr.copy()
                ohdr["nbit"] = 64

                ms_per_gulp = 1e3 * self.ntime_gulp / CHAN_BW
                new_accumulation_time = np.ceil(self.accumulation_time / ms_per_gulp) * ms_per_gulp
                if new_accumulation_time != self.accumulation_time:
                    self.log.warning("Adjusting accumulation time from %.3f ms to %.3f ms",
                                     self.accumulation_time, new_accumulation_time)
                    self.accumulation_time = new_accumulation_time

                ohdr["npol"] = npol ** 2  # Because of cross multiplying shenanigans
                ohdr["skymodes"] = self.skymodes
                ohdr["grid_size_x"] = self.skymodes1d
                ohdr["grid_size_y"] = self.skymodes1d
                ohdr["axes"] = "time,chan,pol,gridy,gridx"
                ohdr["accumulation_time"] = self.accumulation_time
                ohdr["FS"] = FS
                ohdr["latitude"] = self.station.lat * 180. / np.pi
                ohdr["longitude"] = self.station.lon * 180. / np.pi
                ohdr["telescope"] = self.station.name.upper()
                ohdr["data_units"] = "UNCALIB"
                if ohdr["npol"] == 1:
                    ohdr["pols"] = ["xx"]
                elif ohdr["npol"] == 2:
                    ohdr["pols"] = ["xx", "yy"]
                elif ohdr["npol"] == 4:
                    ohdr["pols"] = ["xx", "xy", "yx", "yy"]
                else:
                    raise ValueError("Cannot write fits file without knowing polarization list")
                ohdr_str = json.dumps(ohdr)

                # Setup the kernels to include phasing terms for zenith
                # Phases are Nchan x Nstand x Npol
                # freq.shape += (1,)
                phases = np.zeros((nchan, nstand, npol), dtype=np.complex64)
                for i in range(nstand):
                    # X
                    a = self.station.antennas[2 * i + 0]
                    delay = a.cable.delay(freq) - a.stand.z / speed_of_light.value
                    phases[:, i, 0] = np.exp(2j * np.pi * freq * delay)
                    phases[:, i, 0] /= np.sqrt(a.cable.gain(freq))
                    if npol == 2:
                        # Y
                        a = self.station.antennas[2 * i + 1]
                        delay = a.cable.delay(freq) - a.stand.z / speed_of_light.value
                        phases[:, i, 1] = np.exp(2j * np.pi * freq * delay)
                        phases[:, i, 1] /= np.sqrt(a.cable.gain(freq))
                    # Explicit bad and suspect antenna masking - this will
                    # mask an entire stand if either pol is bad
                    if (
                        self.station.antennas[2 * i + 0].combined_status < 33
                        or self.station.antennas[2 * i + 1].combined_status < 33
                    ):
                        phases[:, i, :] = 0.0
                    # Explicit outrigger masking - we probably want to do
                    # away with this at some point
                    #if (
                    #    (self.station == lwasv and a.stand.id == 256)
                    #    or (self.station == lwa1 and a.stand.id in (35, 257, 258, 259, 260))
                    #):
                    #    phases[:, i, :] = 0.0
                phases = bifrost.ndarray(phases)

                # Setup DFT Transform Matrix

                lm_matrix = np.zeros(shape=(self.skymodes1d, self.skymodes1d, 3))
                lm_step = 2.0 / self.skymodes1d
                i, j = np.meshgrid(np.arange(self.skymodes1d), np.arange(self.skymodes1d))
                # this builds a 3 x 64 x 64 matrix, need to transpose axes to [2, 1, 0] to get correct
                #  64 x 64 x 3 shape
                lm_matrix = np.asarray([i * lm_step - 1.0, j * lm_step - 1.0, np.zeros_like(j)])
                lm_matrix = np.fft.fftshift(lm_matrix, axes=(1,2))
                lm_vector = lm_matrix.transpose([1, 2, 0]).reshape((self.skymodes, 3))

                self.dftm = bifrost.ndarray(
                    form_dft_matrix(lm_vector, locs, phases, nchan, npol, nstand)
                )

                # self.dftm = bifrost.ndarray(np.tile(self.dftm[np.newaxis,:],(nchan,1,1,1)))
                dftm_cu = self.dftm.copy(space="cuda")
                # sys.exit(1)

                oshape = (nchan, npol ** 2, self.skymodes, 1)
                ogulp_size = nchan * npol**2 * self.skymodes * 8
                self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size, buffer_factor=5)
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    iseq_spans = iseq.read(igulp_size)
                    while not self.iring.writing_ended():
                        reset_sequence = False

                        if self.profile:
                            spani = 0

                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue  # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time

                            if self.benchmark is True:
                                print(" ------------------ ")

                            # Correlator
                            # Setup and load
                            idata = ispan.data_view(np.uint8).reshape(itshape)
                            # Fix the type
                            tdata = bifrost.ndarray(shape=itshape, dtype="ci4", native=False, buffer=idata.ctypes.data)

                            if self.benchmark is True:
                                time1 = time.time()
                            # chan pol stand
                            tdata = tdata.transpose((1, 3, 2, 0))
                            tdata = tdata.reshape(nchan * npol, nstand, self.ntime_gulp)
                            tdata = tdata.copy()

                            tdata = tdata.copy(space="cuda")
                            # Unpack
                            try:
                                udata = udata.reshape(*tdata.shape)
                                Unpack(tdata, udata)
                            except NameError:
                                udata = bifrost.ndarray(shape=tdata.shape, dtype=np.complex64, space="cuda")
                                Unpack(tdata, udata)
                            # Phase
                            # bifrost.map('a(i,j,k,l) *= b(j,k,l)',
                            #            {'a':udata, 'b':gphases}, axis_names=('i','j','k','l'), shape=udata.shape)

                            dftm_cu = dftm_cu.reshape(nchan * npol, self.skymodes, nstand)

                            # Perform DFT Matrix Multiplication
                            try:
                                gdata = gdata.reshape(nchan * npol, self.skymodes, self.ntime_gulp)
                                memset_array(data, 0)
                                gdata = self.LinAlgObj.matmul(1.0, dftm_cu, udata, 0.0, gdata)
                            except NameError:
                                gdata = bifrost.zeros(
                                    shape=(nchan * npol, self.skymodes, self.ntime_gulp),
                                    dtype=np.complex64,
                                    space="cuda"
                                )
                                memset_array(gdata, 0)
                                gdata = self.LinAlgObj.matmul(1.0, dftm_cu, udata, 0.0, gdata)

                            gdata = gdata.reshape(1, nchan, npol, self.skymodes, self.ntime_gulp)

                            # Setup matrices for cross-multiplication and accumulation
                            if self.newflag is True:
                                try:
                                    gdatas = gdatas.reshape(nchan, npol ** 2, self.skymodes, self.ntime_gulp)
                                    accumulated_image = accumulated_image.reshape(nchan, npol ** 2, self.skymodes, 1)
                                    memset_array(gdatas, 0)
                                    memset_array(accumulated_image, 0)

                                except NameError:
                                    gdatas = bifrost.zeros(
                                        shape=(nchan, npol ** 2, self.skymodes, self.ntime_gulp),
                                        dtype=np.complex64,
                                        space="cuda"
                                    )
                                    accumulated_image = bifrost.zeros(
                                        shape=(nchan, npol ** 2, self.skymodes, 1),
                                        dtype=np.complex64,
                                        space="cuda"
                                    )
                                self.newflag = False

                            bifrost.map("a(i,j,k,l) += (b(0,i,j/2,k,l) * b(0,i,j%2,k,l).conj())",
                                        {"a": gdatas, "b": gdata},
                                        axis_names=("i", "j", "k", "l"),
                                        shape=(nchan, npol ** 2, self.skymodes, self.ntime_gulp))

                            # Make sure we have a place to put the gridded data
                            # Gridded Antennas
                            # Increment
                            accum += 1e3 * self.ntime_gulp / CHAN_BW

                            if accum >= self.accumulation_time:

                                bifrost.reduce(gdatas, accumulated_image, op="sum")

                                curr_time = time.time()
                                process_time = curr_time - prev_time
                                prev_time = curr_time

                                with oseq.reserve(ogulp_size) as ospan:
                                    odata = ospan.data_view(np.complex64).reshape(oshape)
                                    accumulated_image = accumulated_image.reshape(oshape)
                                    # gdatass = gdatass.reshape(oshape)
                                    # odata[...] = gdatass
                                    odata[...] = accumulated_image
                                    bifrost.device.stream_synchronize()


                                curr_time = time.time()
                                reserve_time = curr_time - prev_time
                                prev_time = curr_time

                                self.newflag = True
                                accum = 0

                            else:
                                process_time = 0.0
                                reserve_time = 0.0

                            curr_time = time.time()
                            process_time += curr_time - prev_time
                            prev_time = curr_time

                            if self.benchmark is True:
                                time2 = time.time()
                                print("-> GPU Time Taken: %f" % (time2 - time1))

                                runtime_history.append(time2 - time1)
                                print("-> Average GPU Time Taken: %f (%i samples)" % (1.0 * sum(runtime_history) / len(runtime_history), len(runtime_history)))
                            if self.profile:
                                spani += 1
                                if spani >= 10:
                                    sys.exit()
                                    break

                            self.perf_proclog.update(
                                {
                                    "acquire_time": acquire_time,
                                    "reserve_time": reserve_time,
                                    "process_time": process_time,
                                }
                            )

                        # Reset to move on to the next input sequence?
                        if not reset_sequence:
                            break


class TriggerOp(object):
    def __init__(
        self,
        log,
        iring,
        ints_per_analysis=1,
        threshold=6.0,
        elevation_limit=20.0,
        core=-1,
        gpu=-1,
        *args,
        **kwargs
    ):
        self.log = log
        self.iring = iring
        self.ints_per_file = ints_per_analysis
        self.threshold = threshold
        self.elevation_limit = elevation_limit * np.pi / 180.0

        self.core = core
        self.gpu = gpu

        self.bind_proclog = ProcLog(type(self).__name__ + "/bind")
        self.in_proclog = ProcLog(type(self).__name__ + "/in")
        self.size_proclog = ProcLog(type(self).__name__ + "/size")
        self.sequence_proclog = ProcLog(type(self).__name__ + "/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__ + "/perf")

        self.in_proclog.update({"nring": 1, "ring0": self.iring.name})
        self.size_proclog.update({"nseq_per_gulp": 1})

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
        self.bind_proclog.update(
            {
                "ncore": 1,
                "core0": bifrost.affinity.get_core(),
                "ngpu": 1,
                "gpu0": BFGetGPU(),
            }
        )

        for iseq in self.iring.read(guarantee=True):
            ihdr = json.loads(iseq.header.tobytes())
            fileid = 0

            self.sequence_proclog.update(ihdr)
            self.log.info("TriggerOp: Config - %s" % ihdr)

            nchan = ihdr["nchan"]
            npol = ihdr["npol"]
            grid_size_x = ihdr["grid_size_x"]
            grid_size_y = ihdr["grid_size_y"]
            grid_size = max([grid_size_x, grid_size_y])
            sampling_length_x = ihdr["sampling_length_x"]
            sampling_length_y = ihdr["sampling_length_y"]
            sampling_length = max([sampling_length_x, sampling_length_y])
            print(
                "Channel no: %d, Polarisation no: %d, Grid no: %d, Sampling: %.3f"
                % (nchan, npol, grid_size, sampling_length)
            )

            x, y = np.arange(grid_size_x), np.arange(grid_size_y)
            x, y = np.meshgrid(x, y)
            rho = np.sqrt((x - grid_size_x / 2) ** 2 + (y - grid_size_y / 2) ** 2)
            mask = np.where(
                rho <= grid_size * sampling_length * np.cos(self.elevation_limit),
                False,
                True
            )

            igulp_size = nchan * npol * grid_size_x * grid_size_y * 8
            ishape = (nchan, npol, grid_size_x, grid_size_y)
            image = []
            image_history = deque([], MAX_HISTORY)

            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            nints = 0

            for ispan in iseq_spans:
                if ispan.size < igulp_size:
                    continue  # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time

                idata = ispan.data_view(np.complex64).reshape(ishape)
                itemp = idata.copy(space="cuda_host")
                image.append(itemp)
                nints += 1
                if nints >= self.ints_per_file:
                    image = np.fft.fftshift(image, axes=(3, 4))
                    image = image[:, :, :, ::-1, :]
                    # NOTE:  This just uses the first polarization (XX) for now.
                    #        In the future we probably want to use Stokes I (if
                    #        possible) to beat the noise down a little.
                    image = image[:, :, 0, :, :].real.sum(axis=0).sum(axis=0)
                    unix_time = (
                        ihdr["time_tag"] / FS
                        + ihdr["accumulation_time"] * 1e-3 * fileid * self.ints_per_file
                    )

                    if len(image_history) == MAX_HISTORY:
                        # The transient detection is based on a differencing the
                        # current image (image) with a moving average of the last
                        # N images (image_background).  This is roughly like what
                        # is done at LWA1/LWA-SV to find events in the LASI images.
                        image_background = np.median(image_history, axis=0)
                        image_diff = np.ma.array(image - image_background, mask=mask)
                        peak, mid, rms = image_diff.max(), image_diff.mean(), image_diff.std()
                        print("-->", peak, mid, rms, "@", (peak - mid) / rms)
                        if (peak - mid) > self.threshold * rms:
                            print(
                                "Trigger Set at %.3f with S/N %f"
                                % (unix_time, (peak - mid) / rms,)
                            )
                            TRIGGER_ACTIVE.set()

                    image_history.append(image)
                    image = []
                    nints = 0
                    fileid += 1


class SaveOp(object):
    def __init__(
        self,
        log,
        iring,
        filename,
        core=-1,
        gpu=-1,
        cpu=False,
        profile=False,
        ints_per_file=1,
        out_dir="",
        triggering=False,
        *args,
        **kwargs
    ):
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

        self.bind_proclog = ProcLog(type(self).__name__ + "/bind")
        self.in_proclog = ProcLog(type(self).__name__ + "/in")
        self.size_proclog = ProcLog(type(self).__name__ + "/size")
        self.sequence_proclog = ProcLog(type(self).__name__ + "/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__ + "/perf")

        self.in_proclog.update({"nring": 1, "ring0": self.iring.name})
        self.size_proclog.update({"nseq_per_gulp": 1})

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
        self.bind_proclog.update(
            {
                "ncore": 1,
                "core0": bifrost.affinity.get_core(),
                "ngpu": 1,
                "gpu0": BFGetGPU(),
            }
        )

        image_history = deque([], MAX_HISTORY)

        for iseq in self.iring.read(guarantee=True):
            ihdr = json.loads(iseq.header.tobytes())
            fileid = 0

            self.sequence_proclog.update(ihdr)
            self.log.info("SaveOp: Config - %s" % ihdr)

            cfreq = ihdr["cfreq"]
            nchan = ihdr["nchan"]
            npol = ihdr["npol"]
            grid_size_x = ihdr["grid_size_x"]
            grid_size_y = ihdr["grid_size_y"]
            grid_size = max([grid_size_x, grid_size_y])
            print(
                "Channel no: %d, Polarisation no: %d, Grid no: %d"
                % (nchan, npol, grid_size)
            )

            igulp_size = nchan * npol * grid_size_x * grid_size_y * 8
            ishape = (nchan, npol, grid_size_x, grid_size_y)
            image = []

            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            nints = 0

            dump_counter = 0

            # some constant header information
            primary_hdu = fits.PrimaryHDU()

            primary_hdu.header["TELESCOP"] = ihdr["telescope"]
            # grab the time from the 0th file for the primary header
            # before dumping
            primary_hdu.header["DATE-OBS"] = Time(
                ihdr["time_tag"] / ihdr["FS"] + 1e-3 * ihdr["accumulation_time"] / 2.0,
                format="unix",
                precision=6,
            ).isot

            primary_hdu.header["BUNIT"] = ihdr["data_units"]
            primary_hdu.header["BSCALE"] = 1e0
            primary_hdu.header["BZERO"] = 0e0
            primary_hdu.header["EQUINOX"] = 2000.0
            primary_hdu.header["EXTNAME"] = "PRIMARY"
            primary_hdu.header["GRIDDIMX"] = ihdr["grid_size_x"]
            primary_hdu.header["GRIDDIMY"] = ihdr["grid_size_y"]
            primary_hdu.header["DGRIDX"] = ihdr["sampling_length_x"]
            primary_hdu.header["DGRIDY"] = ihdr["sampling_length_y"]
            primary_hdu.header["INTTIM"] = ihdr["accumulation_time"] * 1e-3
            primary_hdu.header["INTTIMU"] = "SECONDS"
            primary_hdu.header["CFREQ"] = ihdr["cfreq"]
            primary_hdu.header["CFREQU"] = "HZ"

            pol_dict = {"xx": -5, "yy": -6, "xy": -7, "yx": -8}
            pol_nums = [pol_dict[p] for p in ihdr["pols"]]
            pol_order = np.argsort(pol_nums)[::-1]

            dt = TimeDelta(1e-3 * ihdr["accumulation_time"], format="sec")

            dtheta_x = 2 * np.arcsin(0.5 / (ihdr["grid_size_x"] * ihdr["sampling_length_x"]))
            dtheta_y = 2 * np.arcsin(0.5 / (ihdr["grid_size_y"] * ihdr["sampling_length_y"]))

            crit_pix_x = float(ihdr["grid_size_x"] / 2 + 1)
            # Need to correct for shift in center pixel when we flipped dec dimension
            # when writing npz, Only applies for even dimension size
            crit_pix_y = float(ihdr["grid_size_y"] / 2 + 1) - (ihdr["grid_size_x"] + 1) % 2

            delta_x = -dtheta_x * 180.0 / np.pi
            delta_y = dtheta_y * 180.0 / np.pi
            delta_f = ihdr["bw"] / ihdr["nchan"]
            crit_pix_f = (ihdr["nchan"] - 1) * 0.5 + 1  # +1 for FITS numbering

            if self.profile:
                spani = 0

            for ispan in iseq_spans:
                if ispan.size < igulp_size:
                    continue  # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time

                idata = ispan.data_view(np.complex64).reshape(ishape)
                itemp = idata.copy(space="cuda_host")
                image.append(itemp)
                nints += 1

                if nints >= self.ints_per_file:
                    image = np.fft.fftshift(image, axes=(3, 4))
                    image = image[:, :, :, ::-1, :]

                    # Restructure data in preparation to stuff into fits
                    # Now (Ntimes, Npol, Nfreq, y, x)
                    image = image.transpose(0, 2, 1, 3, 4)

                    # Reorder pol for fits convention
                    image = image[:, pol_order, :, :, :]
                    # Break up real/imaginary
                    image = image[
                        :, np.newaxis, :, :, :, :
                    ]  # Now (Ntimes, 2 (complex), Npol, Nfreq, y, x)
                    image = np.concatenate([image.real, image.imag], axis=1)

                    unix_time = (
                        ihdr["time_tag"] / FS
                        + ihdr["accumulation_time"] * 1e-3 * fileid * self.ints_per_file
                    )

                    t0 = Time(
                        unix_time,
                        format="unix",
                        precision=6,
                        location=(ihdr["longitude"], ihdr["latitude"])
                    )

                    time_array = t0 + np.arange(nints) * dt

                    lsts = time_array.sidereal_time("apparent")
                    coords = SkyCoord(
                        lsts.deg, ihdr["latitude"], obstime=time_array, unit="deg"
                    ).transform_to(FK5(equinox="J2000"))

                    hdul = []
                    for im_num, d in enumerate(image):
                        hdu = fits.ImageHDU(data=d)
                        # Time
                        t = time_array[im_num]
                        lst = lsts[im_num]
                        hdu.header["DATETIME"] = t.isot
                        hdu.header["LST"] = lst.hour
                        # Coordinates - sky

                        hdu.header["EQUINOX"] = 2000.0

                        hdu.header["CTYPE1"] = "RA---SIN"
                        hdu.header["CRPIX1"] = crit_pix_x
                        hdu.header["CDELT1"] = delta_x
                        hdu.header["CRVAL1"] = coords[im_num].ra.deg
                        hdu.header["CUNIT1"] = "deg"
                        hdu.header["CTYPE2"] = "DEC--SIN"

                        hdu.header["CRPIX2"] = crit_pix_y

                        hdu.header["CDELT2"] = delta_y
                        hdu.header["CRVAL2"] = coords[im_num].dec.deg
                        hdu.header["CUNIT2"] = "deg"
                        # Coordinates - Freq
                        hdu.header["CTYPE3"] = "FREQ"
                        hdu.header["CRPIX3"] = crit_pix_f
                        hdu.header["CDELT3"] = delta_f
                        hdu.header["CRVAL3"] = ihdr["cfreq"]
                        hdu.header["CUNIT3"] = "Hz"
                        # Coordinates - Stokes parameters
                        hdu.header["CTYPE4"] = "STOKES"
                        hdu.header["CRPIX4"] = 1
                        hdu.header["CDELT4"] = -1
                        hdu.header["CRVAL4"] = pol_nums[pol_order[0]]
                        # Coordinates - Complex
                        hdu.header["CTYPE5"] = "COMPLEX"
                        hdu.header["CRVAL5"] = 1.0
                        hdu.header["CRPIX5"] = 1.0
                        hdu.header["CDELT5"] = 1.0

                        hdul.append(hdu)

                    filename = os.path.join(
                        self.out_dir,
                        "EPIC_{0:3f}_{1:0.3f}MHz.fits".format(unix_time, cfreq / 1e6),
                    )

                    image_history.append((filename, hdul))

                    if TRIGGER_ACTIVE.is_set() or not self.triggering:
                        if dump_counter == 0:
                            dump_counter = 20 + MAX_HISTORY
                        elif dump_counter == 1:
                            TRIGGER_ACTIVE.clear()



                        cfilename, hdus = image_history.popleft()
                        hdulist = fits.HDUList([primary_hdu, *hdus])
                        hdulist.writeto(cfilename, overwrite=True)

                        # np.savez(cfilename, image=cimage, hdr=chdr, image_nums=cimage_nums)
                        print("SaveOp - Image Saved")
                        dump_counter -= 1

                    image = []
                    nints = 0
                    fileid += 1

                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update(
                    {
                        "acquire_time": acquire_time,
                        "reserve_time": -1,
                        "process_time": process_time,
                    }
                )
                if self.profile:
                    spani += 1
                    if spani >= 10:
                        sys.exit()
                        break

class SaveFFTOp(object):
    def __init__(self, log, iring, filename, ntime_gulp=2500, core=-1, *args, **kwargs):
        self.log = log
        self.iring = iring
        self.filename = filename
        self.core = core
        self.ntime_gulp = ntime_gulp

    def main(self):
        # bifrost.affinity.set_core(self.core)

        for iseq in self.iring.read(guarantee=True):

            ihdr = json.loads(iseq.header.tobytes())
            nchan = ihdr["nchan"]
            nstand = ihdr["nstand"]
            npol = ihdr["npol"]

            igulp_size = self.ntime_gulp * 1 * nstand * npol * 2  # ci8
            ishape = (self.ntime_gulp / nchan, nchan, nstand, npol, 2)

            iseq_spans = iseq.read(igulp_size)

            while not self.iring.writing_ended():

                for ispan in iseq_spans:
                    if ispan.size < igulp_size:
                        continue

                    idata = ispan.data_view(np.int8)

                    idata = idata.reshape(ishape)
                    idata = bifrost.ndarray(shape=ishape, dtype="ci4", native=False, buffer=idata.ctypes.data)
                    print(np.shape(idata))
                    np.savez(self.filename + "asdasd.npy", data=idata)
                    print("Wrote to disk")
            break
        print("Save F-Engine Spectra.. done")

def gen_args(return_parser=False):
    parser = argparse.ArgumentParser(
        description="EPIC Correlator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group1 = parser.add_argument_group("Online Data Processing")
    group1.add_argument(
        "--addr", type=str, default="p5p1", help="F-Engine UDP Stream Address"
    )
    group1.add_argument(
        "--port", type=int, default=4015, help="F-Engine UDP Stream Port"
    )
    group1.add_argument(
        "--utcstart",
        type=str,
        default=None,
        help="F-Engine UDP Stream Start Time",
    )

    group2 = parser.add_argument_group("Offline Data Processing")
    group2.add_argument(
        "--offline", action="store_true", help="Load TBN data from Disk"
    )
    group2.add_argument("--tbnfile", type=str, help="TBN Data Path")
    group2.add_argument(
        "--lwa1", action="store_true", help="TBN data is from LWA1, not LWA-SV"
    )
    group2.add_argument("--tbffile", type=str, help="TBF Data Path")

    group3 = parser.add_argument_group("Processing Options")
    group3.add_argument("--cores", type=str, default="2,3,4,5,6,7", help="Comma separated list of CPU cores to bind to")
    group3.add_argument("--gpu", type=int, default=0, help="GPU to bind to")
    group3.add_argument("--imagesize", type=int, default=64, help="1-D Image Size")
    group3.add_argument(
        "--imageres", type=float, default=1.79057, help="Image pixel size in degrees"
    )
    group3.add_argument(
        "--nts", type=int, default=1000, help="Number of timestamps per span"
    )
    group3.add_argument(
        "--accumulate",
        type=int,
        default=1000,
        help="How many milliseconds to accumulate an image over",
    )
    group3.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Duration of EPIC (seconds)",
    )


    group4 = parser.add_argument_group("Correlation Options")
    group4.add_argument(
        "--dftcorrelation",
        action="store_true",
        help="Use a Direct Fourier Transform to form images",
    )
    group4.add_argument(
        "--dft_skymodes_1D",
        type=int,
        default=64,
        help="How many pixels to simulate per full-sky image using a dft",
    )
    group3.add_argument(
        "--channels", type=int, default=1, help="How many channels to produce"
    )
    group3.add_argument(
        "--singlepol", action="store_true", help="Process only X pol. in online mode"
    )
    group3.add_argument(
        "--removeautocorrs", action="store_true", help="Removes Autocorrelations"
    )

    group4 = parser.add_argument_group("Output")
    group4.add_argument(
        "--ints_per_file",
        type=int,
        default=1,
        help="Number of integrations per output FITS file.",
    )
    group4.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Directory for output files. Default is current directory.",
    )

    group5 = parser.add_argument_group("Self Triggering")
    group5.add_argument(
        "--triggering", action="store_true", help="Enable self-triggering"
    )
    group5.add_argument(
        "--threshold", type=float, default=8.0, help="Self-triggering threshold"
    )
    group5.add_argument(
        "--elevation-limit",
        type=float,
        default=20.0,
        help="Self-trigger minimum elevation limit in degrees",
    )

    group6 = parser.add_argument_group("Benchmarking")
    group6.add_argument("--benchmark", action="store_true", help="benchmark gridder")
    group6.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile on ALL threads. Produces trace for each individual thread",
    )

    args = parser.parse_args()
    # Logging Setup
    # TODO: Set this up properly
    if args.profile:
        enable_thread_profiling()

    if not os.path.isdir(args.out_dir):
        print("Output directory does not exist. Defaulting to current directory.")
        args.out_dir = "."

    #if args.removeautocorrs:
    #    raise NotImplementedError(
    #        "Removing autocorrelations is not yet properly implemented."
    #    )

    if return_parser:
        return args, parser
    else:
        return args


def main(args, parser):

    log = logging.getLogger(__name__)
    logFormat = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logFormat.converter = time.gmtime
    logHandler = logging.StreamHandler(sys.stdout)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)
    log.setLevel(logging.DEBUG)

    # Setup the cores and GPUs to use
    cores = [int(v) for v in args.cores.split(',')]
    gpus = [args.gpu,]*len(cores)

    # Setup the signal handling
    ops = []
    shutdown_event = threading.Event()

    def handle_signal_terminate(signum, frame):
        SIGNAL_NAMES = dict(
            (k, v)
            for v, k in reversed(sorted(signal.__dict__.items()))
            if v.startswith("SIG") and not v.startswith("SIG_")
        )
        log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
        try:
            ops[0].shutdown()
            if SIGNAL_NAMES[signum] == "SIGINT":
                print("****Observation is Interrupted****")
                os._exit(0)
            if SIGNAL_NAMES[signum] == "SIGALRM":
                print("****Observation is Complete****")
                os._exit(0)
        except IndexError:
            pass
        shutdown_event.set()

    for sig in [
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGQUIT,
        signal.SIGTERM,
        signal.SIGTSTP,
        signal.SIGALRM
    ]:
        signal.signal(sig, handle_signal_terminate)

    # Setup Rings

    fcapture_ring = Ring(name="capture", space="cuda_host")
    fdomain_ring = Ring(name="fengine", space="cuda_host")
    gridandfft_ring = Ring(name="gridandfft", space="cuda")

    # Setup the station

    lwa_station = lwasv
    if args.lwa1:
        lwa_station = lwa1

    # Setup threads

    if args.offline:
        if args.tbnfile is not None:
            ops.append(
                TBNOfflineCaptureOp(
                    log,
                    fcapture_ring,
                    args.tbnfile,
                    core=cores.pop(0),
                    profile=args.profile,
                )
            )
            ops.append(
                FDomainOp(
                    log,
                    fcapture_ring,
                    fdomain_ring,
                    ntime_gulp=args.nts,
                    nchan_out=args.channels,
                    core=cores.pop(0),
                    gpu=gpus.pop(0),
                    profile=args.profile,
                )
            )

        elif args.tbffile is not None:
            ops.append(
                TBFOfflineCaptureOp(
                    log,
                    fcapture_ring,
                    args.tbffile,
                    core=cores.pop(0),
                    profile=args.profile,
                )
            )
            ops.append(
                DecimationOp(
                    log,
                    fcapture_ring,
                    fdomain_ring,
                    ntime_gulp=args.nts,
                    nchan_out=args.channels,
                    npol_out=1 if args.singlepol else 2,
                    core=cores.pop(0),
                    gpu=gpus.pop(0),
                )
            )
        else:
            raise parser.error(
                "--offline set but no file provided via --tbnfile or --tbffile"
            )
    else:
        if args.utcstart is None:
            utc_start_dt = get_utc_start()
        else:
            utc_start_dt = datetime.datetime.strptime(args.utcstart, DATE_FORMAT)

        # Note: Capture uses Bifrost address+socket objects, while output uses
        #         plain Python address+socket objects.
        iaddr = BF_Address(args.addr, args.port)
        isock = BF_UDPSocket()
        isock.bind(iaddr)
        isock.timeout = 0.5

        ops.append(
            FEngineCaptureOp(
                log,
                fmt="chips",
                sock=isock,
                ring=fcapture_ring,
                nsrc=16,
                src0=0,
                max_payload_size=9000,
                buffer_ntime=args.nts,
                slot_ntime=25000,
                core=cores.pop(0),
                utc_start=utc_start_dt,
            )
        )
        ops.append(
            DecimationOp(
                log,
                fcapture_ring,
                fdomain_ring,
                ntime_gulp=args.nts,
                nchan_out=args.channels,
                npol_out=1 if args.singlepol else 2,
                core=cores.pop(0),
                gpu=gpus.pop(0),
            )
        )

    if args.dftcorrelation:
        ops.append(
            MOFF_DFT_CorrelatorOp(
                log,
                fdomain_ring,
                gridandfft_ring,
                lwa_station,
                skymodes=args.dft_skymodes_1D,
                ntime_gulp=args.nts,
                accumulation_time=args.accumulate,
                core=cores.pop(0),
                gpu=gpus.pop(0),
                benchmark=args.benchmark,
                profile=args.profile,
            )
        )
    else:
        ops.append(
            MOFFCorrelatorOp(
                log,
                fdomain_ring,
                gridandfft_ring,
                lwa_station,
                args.imagesize,
                args.imageres,
                ntime_gulp=args.nts,
                accumulation_time=args.accumulate,
                remove_autocorrs=args.removeautocorrs,
                core=cores.pop(0),
                gpu=gpus.pop(0),
                benchmark=args.benchmark,
                profile=args.profile,
            )
        )
    if args.triggering:
        ops.append(
            TriggerOp(
                log,
                gridandfft_ring,
                core=cores.pop(0),
                gpu=gpus.pop(0),
                ints_per_analysis=args.ints_per_file,
                threshold=args.threshold,
                elevation_limit=max([0.0, args.elevation_limit]),
            )
        )

    ops.append(
        SaveOp(
            log,
            gridandfft_ring,
            "EPIC_",
            out_dir=args.out_dir,
            core=cores.pop(0),
            gpu=gpus.pop(0),
            cpu=False,
            ints_per_file=args.ints_per_file,
            triggering=args.triggering,
            profile=args.profile,
        )
    )

    threads = [threading.Thread(target=op.main) for op in ops]

    # Go!

    for thread in threads:
        thread.daemon = False
        thread.start()

    signal.alarm(args.duration)


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
    args, parser = gen_args(return_parser=True)
    main(args, parser)
