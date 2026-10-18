"""Shared-memory exchange with epic++ (layouts match src/ex/gain_table.hpp and src/ex/cal_tap.hpp).

GainTableWriter  publishes per-antenna corrections c = 1/g, [nchan][256][2] complex64,
                 which epic++ multiplies into its nominal phases at image boundaries.
VisTableReader   reads the visibility matrices published by the epic++ calibration tap,
                 [nchan][2][256][256] complex64 (R = sum x x^H / nsamp, raw F-engine voltages).
ProjTableWriter  publishes the interference projector (src/ex/proj_table.hpp): per channel
                 qmax pairs of vectors (Ut, Vt) over the 512 inputs (input 2 stand + pol);
                 epic++ replaces every voltage sequence x by x - Ut (Vt^H x) before imaging.
                 Optionally also the Sun's (projected, unit-norm per polarisation) response,
                 which exempts solar bursts from epic++'s impulse detection.
All use a seqlock: an odd `seq` means a write is in progress.
"""
from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass

import numpy as np

NSTAND, NPOL = 256, 2

GAIN_HDR = struct.Struct("<8sIIQQqIIIIQdfI48s")
VIS_HDR = struct.Struct("<8sIIQQqIIIIQQQIIQf28s")
PROJ_HDR = struct.Struct("<8sIIQQqIIIIIIQd48s")
assert GAIN_HDR.size == 128 and VIS_HDR.size == 128 and PROJ_HDR.size == 128
NINPUT = NSTAND * NPOL

# NDP F-engine: one sequence = 8192 samples at 196 MS/s
SEQ_SECONDS = 8192 / 196e6


class GainTableWriter:
    def __init__(self, path: str, chan0: int, nchan: int, solver: str = "streamcal"):
        self.path, self.chan0, self.nchan = path, int(chan0), int(nchan)
        self.size = GAIN_HDR.size + nchan * NSTAND * NPOL * 8
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.fstat(fd).st_size != self.size:
                os.ftruncate(fd, self.size)
            self.mm = mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        old = GAIN_HDR.unpack_from(self.mm, 0)
        self.seq = old[3] + (old[3] & 1)
        self.version = old[4] if old[0] == b"EPICGAIN" else 0
        self.solver = solver.encode()[:47]
        if old[0] != b"EPICGAIN":
            # a valid, inactive header (version 0) until the first table, so the reader
            # sees "nothing new" rather than a malformed file
            self.mm[:GAIN_HDR.size] = GAIN_HDR.pack(b"EPICGAIN", 1, GAIN_HDR.size, self.seq, 0, self.chan0,
                                                    self.nchan, NSTAND, NPOL, 0, 0, 0.0, 0.0, 0, self.solver)

    def publish(self, corr: np.ndarray | None, valid_from_seq: int = 0, t_solved_unix: float | None = None,
                max_abs: float = 64.0):
        """corr: complex [nchan, 256, 2] multiplying the nominal phases (zeros flag), or None to revert."""
        valid = corr is not None
        if valid:
            corr = np.asarray(corr, np.complex64)
            assert corr.shape == (self.nchan, NSTAND, NPOL)
            if not np.all(np.isfinite(corr)) or np.abs(corr).max() > max_abs:
                raise ValueError("gain table has non-finite or out-of-range values")
            mags = np.abs(corr[corr != 0])
            if mags.size and not 0.25 <= np.median(mags) <= 4.0:
                # epic++ rejects these too: large factors overflow its half-precision FFT
                raise ValueError(f"gain table is not normalised (median |c| = {np.median(mags):.3g})")
        self.seq += 1  # odd: writing
        struct.pack_into("<Q", self.mm, 16, self.seq)
        if valid:
            self.mm[GAIN_HDR.size:] = corr.tobytes()
        self.version += 1
        hdr = GAIN_HDR.pack(b"EPICGAIN", 1, GAIN_HDR.size, self.seq, self.version, self.chan0, self.nchan, NSTAND,
                            NPOL, 1 if valid else 0, int(valid_from_seq),
                            float(t_solved_unix if t_solved_unix is not None else time.time()), float(max_abs), 0,
                            self.solver)
        self.mm[:GAIN_HDR.size] = hdr
        self.seq += 1  # even: stable
        struct.pack_into("<Q", self.mm, 16, self.seq)
        self.mm.flush()
        return self.version


class ProjTableWriter:
    """Interference projector for epic++: Ut, Vt complex64 [nchan][qmax][512].

    For each polarisation the components act only on that polarisation's inputs (2 stand + pol);
    unused components are zero. Projection with exact vectors is idempotent: Vt^H Ut = I."""

    def __init__(self, path: str, chan0: int, nchan: int, qmax: int = 4, solver: str = "streamcal"):
        self.path, self.chan0, self.nchan, self.qmax = path, int(chan0), int(nchan), int(qmax)
        self.size = PROJ_HDR.size + (2 * qmax + 1) * nchan * NINPUT * 8
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.fstat(fd).st_size != self.size:
                os.ftruncate(fd, self.size)
            self.mm = mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        old = PROJ_HDR.unpack_from(self.mm, 0)
        self.seq = old[3] + (old[3] & 1)
        self.version = old[4] if old[0] == b"EPICPROJ" else 0
        self.solver = solver.encode()[:47]
        if old[0] != b"EPICPROJ" or old[6] != self.nchan or old[8] != self.qmax:
            self.mm[:PROJ_HDR.size] = PROJ_HDR.pack(b"EPICPROJ", 1, PROJ_HDR.size, self.seq, self.version,
                                                    self.chan0, self.nchan, NINPUT, self.qmax, 0, 0, 0, 0, 0.0,
                                                    self.solver)

    def publish(self, Ut: np.ndarray | None, Vt: np.ndarray | None = None, valid_from_seq: int = 0,
                t_solved_unix: float | None = None, sun: np.ndarray | None = None):
        """Ut, Vt: complex [nchan, qmax, 512], or None to switch the projection off.
        sun: complex [nchan, 512], the Sun's response on the raw inputs after projection, unit norm
        over each polarisation's inputs (zero: none), or None."""
        valid = Ut is not None
        q = 0
        if sun is not None:
            sun = np.asarray(sun, np.complex64)
            assert sun.shape == (self.nchan, NINPUT)
            if not np.all(np.isfinite(sun)):
                raise ValueError("Sun vector has non-finite values")
        if valid:
            Ut, Vt = np.asarray(Ut, np.complex64), np.asarray(Vt, np.complex64)
            assert Ut.shape == Vt.shape == (self.nchan, self.qmax, NINPUT)
            if not (np.all(np.isfinite(Ut)) and np.all(np.isfinite(Vt))):
                raise ValueError("projector has non-finite values")
            used = np.any(Ut != 0, axis=(0, 2))
            q = int(np.flatnonzero(used).max() + 1) if used.any() else 0
        self.seq += 1  # odd: writing
        struct.pack_into("<Q", self.mm, 16, self.seq)
        if valid:
            n = self.nchan * self.qmax * NINPUT * 8
            self.mm[PROJ_HDR.size:PROJ_HDR.size + n] = Ut.tobytes()
            self.mm[PROJ_HDR.size + n:PROJ_HDR.size + 2 * n] = Vt.tobytes()
            if sun is not None:
                o = PROJ_HDR.size + 2 * n
                self.mm[o:o + self.nchan * NINPUT * 8] = sun.tobytes()
        self.version += 1
        flags = (1 if valid else 0) | (2 if valid and sun is not None else 0)
        hdr = PROJ_HDR.pack(b"EPICPROJ", 1, PROJ_HDR.size, self.seq, self.version, self.chan0, self.nchan, NINPUT,
                            self.qmax, q, flags, 0, int(valid_from_seq),
                            float(t_solved_unix if t_solved_unix is not None else time.time()), self.solver)
        self.mm[:PROJ_HDR.size] = hdr
        self.seq += 1  # even: stable
        struct.pack_into("<Q", self.mm, 16, self.seq)
        self.mm.flush()
        return self.version


@dataclass
class VisTable:
    version: int
    chan0: int
    nchan: int
    nsamp: int
    seq_first: int
    seq_last: int
    ngulps: int
    nskipped: int
    gulp_nseq: int
    duty: float
    R: np.ndarray  # [nchan, 2, 256, 256] complex64

    @property
    def t_mid_unix(self):
        return 0.5 * (self.seq_first + self.seq_last + self.gulp_nseq) * SEQ_SECONDS


class VisTableReader:
    def __init__(self, path: str):
        self.path = path
        self.mm = None
        self.last = None

    def _map(self):
        if not os.path.exists(self.path):
            return False
        fd = os.open(self.path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            if size < VIS_HDR.size:
                return False
            self.mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        finally:
            os.close(fd)
        return True

    def poll(self) -> VisTable | None:
        """Return a new table (a copy) if one has been published since the last poll."""
        if self.mm is None and not self._map():
            return None
        for _ in range(3):
            h = VIS_HDR.unpack_from(self.mm, 0)
            if h[0] != b"EPICVIS1" or h[3] & 1:
                return None
            key = (h[4], h[11])  # version, seq_first (the version restarts with epic++)
            if key == self.last:
                return None
            nchan, nstand, npol = h[6], h[7], h[8]
            need = VIS_HDR.size + nchan * npol * nstand * nstand * 8
            if need > len(self.mm):
                self.mm.close()
                self.mm = None
                if not self._map() or need > len(self.mm):
                    return None
            R = np.frombuffer(self.mm, np.complex64, nchan * npol * nstand * nstand, VIS_HDR.size).copy()
            if struct.unpack_from("<Q", self.mm, 16)[0] != h[3]:
                continue  # torn read
            self.last = key
            return VisTable(h[4], h[5], nchan, h[10], h[11], h[12], h[13], h[14], h[15], h[16],
                            R.reshape(nchan, npol, nstand, nstand))
        return None
