"""
Shared-memory image ring between the imager and the all-sky monitor.

The imager (or the sky simulator) writes every image's Stokes I (XX + YY)
into a ring of slots in /dev/shm/<name> and never waits for a reader; the
monitor follows the ring and detects when it falls behind. A crash or restart
of either side therefore never stalls the other. The C++ writer is
src/ex/sky_publisher.hpp; the layout below is shared with it.

File header (4096 bytes, little endian)
  0  char[8]  magic "EPICSKY1"
  8  u32      version (1)
 12  u32      nslots
 16  u64      slot_bytes
 24  u64      data_offset (4096)
 32  u32      max_nchan
 36  u32      max_grid
 40  u64      write_count: images completed since the writer started
 48  u32      writer pid
 56  u64      session: changes when the writer restarts

Slot i at data_offset + i * slot_bytes, 256-byte header then data
  0  u64      lock: 2n + 1 while image n is written, 2n + 2 once complete
  8  u64      seq_start (NDP sequence of the first spectrum)
 16  u32      nseqs (spectra in the image)
 20  u32      chan0 (absolute NDP channel of data channel 0)
 24  u32      nchan
 28  u32      nbin (NDP channels summed per data channel)
 32  u32      grid
 36  u32      phase centre mode (0 plane, 1 zenith)
 40  f64      grid_res (deg)
 48  f64      img_len_ms
256  f32      data[nchan][grid][grid], FITS order ([y][x], as the saved images)
"""
from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass

import numpy as np

MAGIC = b"EPICSKY1"
VERSION = 1
HEADER_BYTES = 4096
SLOT_HEADER = 256
PHASE_MODES = {0: "plane", 1: "zenith"}


def shm_path(name: str) -> str:
    return name if name.startswith("/") else os.path.join("/dev/shm", name)


def slot_bytes_for(max_nchan: int, max_grid: int) -> int:
    n = SLOT_HEADER + 4 * max_nchan * max_grid * max_grid
    return (n + 4095) // 4096 * 4096


@dataclass
class ImageMeta:
    index: int  # image number in this writer session
    seq_start: int
    nseqs: int
    chan0: int
    nchan: int
    nbin: int
    grid: int
    phase_mode: str
    grid_res: float
    img_len_ms: float

    @property
    def unix(self) -> float:
        from .geometry import seq_to_unix
        return seq_to_unix(self.seq_start)


class RingWriter:
    def __init__(self, name: str, nslots: int, max_nchan: int, max_grid: int):
        self.path = shm_path(name)
        self.nslots = nslots
        self.slot_bytes = slot_bytes_for(max_nchan, max_grid)
        size = HEADER_BYTES + nslots * self.slot_bytes
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, size)
            self.mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        self.count = 0
        hdr = struct.pack("<8sIIQQIIQIIQ", MAGIC, VERSION, nslots, self.slot_bytes,
                          HEADER_BYTES, max_nchan, max_grid, 0, os.getpid(), 0,
                          time.time_ns())
        self.mm[: len(hdr)] = hdr
        self.max_nchan, self.max_grid = max_nchan, max_grid

    def write(self, data: np.ndarray, seq_start: int, nseqs: int, chan0: int, nbin: int = 1,
              grid_res: float = 1.056, img_len_ms: float = 83.592, phase_mode: str = "plane"):
        nchan, gy, gx = data.shape
        assert gy == gx and nchan <= self.max_nchan and gx <= self.max_grid
        n = self.count
        off = HEADER_BYTES + (n % self.nslots) * self.slot_bytes
        self.mm[off: off + 8] = struct.pack("<Q", 2 * n + 1)
        mode = 1 if phase_mode == "zenith" else 0
        self.mm[off + 8: off + 56] = struct.pack("<QIIIIIIdd", seq_start, nseqs, chan0, nchan,
                                                 nbin, gx, mode, grid_res, img_len_ms)
        dst = np.frombuffer(self.mm, np.float32, nchan * gy * gx, off + SLOT_HEADER)
        dst[:] = data.reshape(-1)
        self.mm[off: off + 8] = struct.pack("<Q", 2 * n + 2)
        self.count = n + 1
        self.mm[40:48] = struct.pack("<Q", self.count)

    def close(self, unlink: bool = False):
        self.mm.close()
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


class RingReader:
    """
    Follows a ring. read() returns (meta, data) for the next image, or None if
    there is none yet. Images overwritten before they were read are counted in
    self.dropped; a writer restart (new session) resets the position.
    """

    def __init__(self, name: str):
        self.path = shm_path(name)
        self.mm = None
        self.next = None
        self.session = None
        self.dropped = 0
        self.restarts = 0

    def _open(self) -> bool:
        if self.mm is not None:
            return True
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return False
        try:
            size = os.fstat(fd).st_size
            if size < HEADER_BYTES:
                return False
            self.mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        magic, ver, self.nslots, self.slot_bytes, self.data_offset = struct.unpack_from(
            "<8sIIQQ", self.mm, 0)
        if magic != MAGIC or ver != VERSION:
            self.mm.close()
            self.mm = None
            raise ValueError(f"{self.path} is not an EPIC sky ring")
        return True

    def _header(self):
        count, = struct.unpack_from("<Q", self.mm, 40)
        session, = struct.unpack_from("<Q", self.mm, 56)
        return count, session

    def read(self):
        if not self._open():
            return None
        count, session = self._header()
        if session != self.session:
            if self.session is not None:
                self.restarts += 1
            self.session = session
            self.next = max(count - 1, 0)  # start with the newest image
        if count < self.next:  # writer restarted within the same session id
            self.next = count
        if count == self.next:
            return None
        if count - self.next > self.nslots - 1:  # overrun: skip to the oldest safe slot
            skip = count - (self.nslots - 1) - self.next
            self.dropped += skip
            self.next += skip
        n = self.next
        off = self.data_offset + (n % self.nslots) * self.slot_bytes
        lock1, = struct.unpack_from("<Q", self.mm, off)
        if lock1 != 2 * n + 2:
            if lock1 > 2 * n + 2:  # already overwritten
                self.dropped += 1
                self.next += 1
            return None
        (seq_start, nseqs, chan0, nchan, nbin, grid, mode, grid_res,
         img_len_ms) = struct.unpack_from("<QIIIIIIdd", self.mm, off + 8)
        data = np.frombuffer(self.mm, np.float32, nchan * grid * grid,
                             off + SLOT_HEADER).copy().reshape(nchan, grid, grid)
        lock2, = struct.unpack_from("<Q", self.mm, off)
        self.next += 1
        if lock2 != lock1:  # overwritten while copying
            self.dropped += 1
            return None
        meta = ImageMeta(n, seq_start, nseqs, chan0, nchan, nbin, grid,
                         PHASE_MODES.get(mode, "plane"), grid_res, img_len_ms)
        return meta, data

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None
