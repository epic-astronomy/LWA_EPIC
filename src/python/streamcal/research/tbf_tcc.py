"""Correlate whole TBF captures (every sample, every channel) with the Tensor-Core Correlator.

    python tbf_tcc.py FILE OUTDIR [--chan-chunk 132] [--chan-stride 1]

Pilot data for kernel-StEFCal on real LST-pooled data (section 8.10, step 2): the Memo 12
captures (/FileStore/epic0/memo12_Data/rawfiles, 2023, ADP, ~38 MHz, 792 channels of
25 kHz, ~1 s each over an 8 h Cyg A transit). tbf_cache.py used 10,000 samples of 16
channels; this uses all ~25,000 samples of all channels.

A TBF frame is 6168 bytes: a 24-byte header (first channel, big endian, at byte 12;
timetag, big endian, at byte 16) and [12 chan][256 stand][2 pol] bytes of 4+4-bit samples
(real part in the high nibble). Frames are read with a memmap and reordered with numpy
into the TCC i8 layout, then correlated on the GPU with tests/streamcal/tcc/tcc_correlate.

Output OUTDIR/<capture>.npz, the format of tbn_tcc.py:
  R_even, R_odd  [nchan, 2 (XX, YY), 256, 256] complex64, even / odd samples, each / nsamp
  freq [nchan] Hz, chans [nchan] (ADP channel numbers), t (ISO time of the first sample),
  nsamp (per half), nvalid [nchan] (time steps present per channel)
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

from streamcal import tcc  # noqa: E402

FRAME = 6168
HDR = 24
TICKS_PER_SAMPLE = 7840  # 196 MHz / 25 kHz
CHAN_BW = 25e3
T = 16  # TCC i8 samples per block

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("outdir")
ap.add_argument("--chan-chunk", type=int, default=132, help="channels per GPU call (a multiple of 12)")
ap.add_argument("--chan-stride", type=int, default=1, help="keep every Nth channel")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
out = os.path.join(a.outdir, os.path.basename(a.file) + ".npz")
if os.path.exists(out):
    print("exists", out)
    sys.exit(0)

T0 = time.time()
raw = np.memmap(a.file, np.uint8, "r")
nfr = raw.size // FRAME
fr = raw[:nfr * FRAME].reshape(nfr, FRAME)
first = (fr[:, 12].astype(np.int64) << 8) | fr[:, 13]
tt = np.zeros(nfr, np.uint64)
for i in range(8):
    tt = (tt << np.uint64(8)) | fr[:, 16 + i].astype(np.uint64)
t0 = tt.min()
k = ((tt - t0) // np.uint64(TICKS_PER_SAMPLE)).astype(np.int64)
blocks = np.unique(first)
nt = int(k.max()) + 1
nt_half = (nt // 2) // T * T
print(f"{os.path.basename(a.file)}: {nfr} frames, {len(blocks)} channel blocks ({blocks[0]}..{blocks[-1] + 11}), "
      f"{nt} time steps, {nt_half} per half; header scan {time.time() - T0:.0f}s", flush=True)

chans_all = (blocks[:, None] + np.arange(12)[None, :]).ravel()
keep = np.arange(0, len(chans_all), a.chan_stride)
chans = chans_all[keep]
R = np.zeros((2, len(chans), 2, 256, 256), np.complex64)  # half, chan, pol
nvalid = np.zeros(len(chans), np.int64)
bpc = a.chan_chunk // 12
tool = tcc.TOOL
workdir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
for b0 in range(0, len(blocks), bpc):
    T1 = time.time()
    bsel = blocks[b0:b0 + bpc]
    cidx = np.flatnonzero(np.isin(chans_all, (bsel[:, None] + np.arange(12)).ravel()))  # into chans_all
    sel = np.isin(cidx, keep)
    if not sel.any():
        continue
    nc = len(bsel) * 12
    # [time][chan in chunk][512] bytes; missing frames stay zero
    buf = np.zeros((nt, nc, 512), np.uint8)
    for j, b in enumerate(bsel):
        idx = np.flatnonzero(first == b)
        buf[k[idx], j * 12:(j + 1) * 12, :] = fr[idx, HDR:].reshape(-1, 12, 512)
        present = np.zeros(nt, bool)
        present[k[idx]] = True
        for c in range(12):
            gi = np.flatnonzero(keep == cidx[j * 12 + c])
            if gi.size:
                nvalid[gi[0]] = present.sum()
    buf = buf[:, sel]  # only kept channels
    ncs = buf.shape[1]
    for h in (0, 1):
        x = buf[h::2][:nt_half]  # [t, chan, 512]
        s = x.view(np.int8)
        re = s >> 4
        im = (x << 4).view(np.int8) >> 4
        z = np.stack([re, im], -1)  # [t, chan, 512, 2]
        z = z.reshape(nt_half // T, T, ncs, 512, 2).transpose(2, 0, 3, 1, 4)  # [chan, t/T, input, T, 2]
        fin = os.path.join(workdir, f"tbf_tcc_{os.getpid()}.in")
        fout = fin + ".vis"
        np.ascontiguousarray(z).tofile(fin)
        msg = subprocess.run([tool, "i8", "256", str(ncs), str(nt_half), fin, fout], check=True,
                             capture_output=True, text=True).stdout.strip()
        v = np.fromfile(fout, np.int32).reshape(ncs, 256 * 257 // 2, 2, 2, 2)
        os.remove(fin)
        os.remove(fout)
        v = (v[..., 0] + 1j * v[..., 1]).astype(np.complex64)  # [chan, bl, polY, polX]
        iX, iY = np.tril_indices(256)  # bl = X (X + 1) / 2 + Y, Y <= X
        gsel = np.searchsorted(chans, chans_all[cidx[sel]])
        for p in (0, 1):
            Rp = np.zeros((ncs, 256, 256), np.complex64)
            Rp[:, iY, iX] = v[:, :, p, p]  # R[a, b] = sum x_a conj(x_b), a <= b
            Rp[:, iX, iY] = np.conj(v[:, :, p, p])
            R[h, gsel, p] = Rp / nt_half
    print(f"  channels {chans_all[cidx[0]]}..{chans_all[cidx[-1]]}: {time.time() - T1:.0f}s ({msg})", flush=True)

t = Time(t0 / 196e6, format="unix")
tmp = out + ".tmp.npz"
np.savez(tmp, R_even=R[0], R_odd=R[1], freq=chans * CHAN_BW, chans=chans, t=t.isot, nsamp=nt_half, nvalid=nvalid)
os.replace(tmp, out)
print(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB) t={t.isot} in {time.time() - T0:.0f}s", flush=True)
