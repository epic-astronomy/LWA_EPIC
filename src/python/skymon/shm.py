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
 56  u64      gain_version: version of the calibration gain table the image
              was made with (epic++ --gain_table; 0: nominal phases, no table)
64  u32      layout: 0 the image below; 1 the monitor's pixel spectra
              data[nchan][npix], regridded on the GPU by the imager (FX
              imaging) with the pixel table this monitor writes
              (SkyTableWriter)
 68  u32      npix (layout 1)
 72  u64      table_version (layout 1): the pixel table the spectra follow
 80  u32      peel_mask (layout 1): bit s set when the imager subtracted bright
              source PEEL_NAMES[s] from the spectra (EPIC_FX_PEEL=sky); its
              peak in image units at its own position, per channel, then
              follows the spectra: peaks[len(PEEL_NAMES)][nchan] float32
 84  u32      peel_term (layout 1, with peel_mask): 1 when the subtracted term,
              the peeled sources as they appear at each pixel, follows the
              peaks: term[nchan][npix] float32; spectra + term is the unpeeled
              sky, which the display accumulation (sky map, spectrograms,
              calibrator measurements) keeps
256  f32      data[nchan][grid][grid], FITS order ([y][x], as the saved images)

Pixel table /dev/shm/<name>_table, written by the monitor and read by the
imager (src/ex/sky_table.hpp, which documents the layout): the image position
of every pixel slot at two times, between which both interpolate for each
image (geometry.SkyGrid.taps).
"""
from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass

import numpy as np

MAGIC = b"EPICSKY1"
# the imager's bright sources, in the order of their bits in peel_mask (sky_sources.hpp)
PEEL_NAMES = ("CygA", "CasA", "TauA", "VirA", "Sun")
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
    gain_version: int = 0
    layout: int = 0  # 1: data are pixel spectra [nchan][npix] (FX imaging)
    npix: int = 0
    table_version: int = 0
    peel_mask: int = 0
    # [len(PEEL_NAMES)][nchan]: the peeled sources' peaks (image units), or None
    peel_peaks: np.ndarray | None = None
    # [nchan][npix]: the peeled sources at each pixel (spectra + term: unpeeled), or None
    peel_term: np.ndarray | None = None

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
              grid_res: float = 1.056, img_len_ms: float = 83.592, phase_mode: str = "plane",
              gain_version: int = 0):
        nchan, gy, gx = data.shape
        assert gy == gx and nchan <= self.max_nchan and gx <= self.max_grid
        n = self.count
        off = HEADER_BYTES + (n % self.nslots) * self.slot_bytes
        self.mm[off: off + 8] = struct.pack("<Q", 2 * n + 1)
        mode = 1 if phase_mode == "zenith" else 0
        self.mm[off + 8: off + 64] = struct.pack("<QIIIIIIddQ", seq_start, nseqs, chan0, nchan,
                                                 nbin, gx, mode, grid_res, img_len_ms,
                                                 gain_version)
        dst = np.frombuffer(self.mm, np.float32, nchan * gy * gx, off + SLOT_HEADER)
        dst[:] = data.reshape(-1)
        self.mm[off: off + 8] = struct.pack("<Q", 2 * n + 2)
        self.count = n + 1
        self.mm[40:48] = struct.pack("<Q", self.count)

    def write_pixels(self, data: np.ndarray, seq_start: int, nseqs: int, chan0: int, grid: int,
                     table_version: int, nbin: int = 1, grid_res: float = 1.056,
                     img_len_ms: float = 83.592, gain_version: int = 0, peaks: dict | None = None,
                     term: np.ndarray | None = None):
        """Pixel spectra data [nchan][npix] (layout 1), as the FX imager publishes them;
        peaks: {name in PEEL_NAMES: [nchan]} of the sources peeled from them; term
        [nchan][npix]: those sources at each pixel (written only with peaks)"""
        nchan, npix = data.shape
        assert nchan <= self.max_nchan
        n = self.count
        off = HEADER_BYTES + (n % self.nslots) * self.slot_bytes
        self.mm[off: off + 8] = struct.pack("<Q", 2 * n + 1)
        self.mm[off + 8: off + 64] = struct.pack("<QIIIIIIddQ", seq_start, nseqs, chan0, nchan,
                                                 nbin, grid, 0, grid_res, img_len_ms, gain_version)
        mask, pk = 0, np.zeros((len(PEEL_NAMES), nchan), np.float32)
        for name, v in (peaks or {}).items():
            i = PEEL_NAMES.index(name)
            mask |= 1 << i
            pk[i] = v
        has_term = bool(mask) and term is not None
        self.mm[off + 64: off + 88] = struct.pack("<IIQII", 1, npix, table_version, mask, int(has_term))
        n_term = nchan * npix if has_term else 0
        dst = np.frombuffer(self.mm, np.float32, nchan * npix + pk.size + n_term, off + SLOT_HEADER)
        dst[: nchan * npix] = data.reshape(-1)
        dst[nchan * npix: nchan * npix + pk.size] = pk.reshape(-1)
        if has_term:
            dst[nchan * npix + pk.size:] = np.asarray(term, np.float32).reshape(-1)
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
         img_len_ms, gain_version) = struct.unpack_from("<QIIIIIIddQ", self.mm, off + 8)
        layout, npix, table_version, peel_mask, has_term = struct.unpack_from("<IIQII", self.mm, off + 64)
        peaks = term = None
        if layout == 1:
            data = np.frombuffer(self.mm, np.float32, nchan * npix,
                                 off + SLOT_HEADER).copy().reshape(nchan, npix)
            if peel_mask:
                o = off + SLOT_HEADER + 4 * nchan * npix
                peaks = np.frombuffer(self.mm, np.float32, len(PEEL_NAMES) * nchan, o).copy().reshape(-1, nchan)
                if has_term:
                    term = np.frombuffer(self.mm, np.float32, nchan * npix,
                                         o + 4 * len(PEEL_NAMES) * nchan).copy().reshape(nchan, npix)
        else:
            layout, npix, table_version, peel_mask = 0, 0, 0, 0
            data = np.frombuffer(self.mm, np.float32, nchan * grid * grid,
                                 off + SLOT_HEADER).copy().reshape(nchan, grid, grid)
        lock2, = struct.unpack_from("<Q", self.mm, off)
        self.next += 1
        if lock2 != lock1:  # overwritten while copying
            self.dropped += 1
            return None
        meta = ImageMeta(n, seq_start, nseqs, chan0, nchan, nbin, grid,
                         PHASE_MODES.get(mode, "plane"), grid_res, img_len_ms, gain_version,
                         layout, npix, table_version, peel_mask if peaks is not None else 0, peaks, term)
        return meta, data

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None


class SkyTableWriter:
    """
    The monitor's pixel table for the imager (src/ex/sky_table.hpp): with FX
    imaging the imager regrids every image to these pixel slots on the GPU
    and publishes their spectra (ring layout 1) instead of the image.
    """

    MAGIC = b"EPICSKYT"
    HEADER = 128

    def __init__(self, name: str):
        self.path = shm_path(name)
        self.mm = None
        self.capacity = 0
        self.version = 0

    def _map(self, capacity: int):
        import mmap

        size = self.HEADER + 2 * capacity * 2 * 8
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, size)
            self.mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        self.mm[0:8] = self.MAGIC
        struct.pack_into("<IIQ", self.mm, 8, 1, capacity, 0)
        self.capacity = capacity

    def write(self, grid: int, grid_res: float, t_a: float, t_b: float, xy_a, xy_b):
        """Positions xy_a at t_a and xy_b at t_b ([P, 2], 1-based FITS pixels, NaN for a free slot)"""
        xy_a = np.ascontiguousarray(xy_a, np.float64)
        xy_b = np.ascontiguousarray(xy_b, np.float64)
        P = xy_a.shape[0]
        if self.mm is None or P != self.capacity:
            if self.mm is not None:
                self.mm.close()
            self._map(P)
        seq, = struct.unpack_from("<Q", self.mm, 16)
        seq += 1 if seq % 2 == 0 else 2  # odd: being written
        struct.pack_into("<Q", self.mm, 16, seq)
        self.version += 1
        struct.pack_into("<QII", self.mm, 24, self.version, int(grid), 0)
        struct.pack_into("<ddd", self.mm, 40, float(grid_res), float(t_a), float(t_b))
        n = P * 2 * 8
        self.mm[self.HEADER:self.HEADER + n] = xy_a.tobytes()
        self.mm[self.HEADER + n:self.HEADER + 2 * n] = xy_b.tobytes()
        struct.pack_into("<Q", self.mm, 16, seq + 1)  # even: complete
        return self.version

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None
