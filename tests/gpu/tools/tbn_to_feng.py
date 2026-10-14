"""
Convert an LSL TBN capture to the F-engine input of the epic++ imaging kernel, for
tests/gpu/image_file.

Writes <prefix>.feng   uint8, CHAN_MAJOR [chan][seq][stand] CNib2 (4-bit X re/im, Y re/im)
       <prefix>.phases float32 [chan][stand][Xre, Xim, Yre, Yim], gen_phases_lwasv with the
                       epic_geometry phase centre (cable delay - p.r/c, 1/sqrt(cable gain))
       <prefix>.json   chan0 (in imager channels of FS / 8192), nchan, nseq, unix time of the first sample

The TBN band (100 kHz) is channelised into 25 kHz channels with a 4-point FFT.
usage: python tbn_to_feng.py <src/python> <tbn file> <prefix>
"""
import json
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import epic_geometry as eg  # noqa: E402
from lsl.common.stations import lwasv  # noqa: E402
from lsl.reader.ldp import TBNFile  # noqa: E402

tbn, prefix = sys.argv[2], sys.argv[3]
BW, SOL = 25e3, 299792458.0  # TBN is channelised here into 25 kHz channels
NDP_BW = 196.0e6 / 8192      # the imager's channel width (constants.h BANDWIDTH)

idf = TBNFile(tbn)
cf, sr = idf.get_info('freq1'), idf.get_info('sample_rate')
nfr = idf.get_info('nframe') // idf.get_info('nantenna')
_, tstart, data = idf.read(nfr * 512 / sr * 0.999)
idf.close()
nin = data.shape[0]
nsamp = data.shape[1] // 4 * 4
spec = np.fft.fft(data[:, :nsamp].reshape(nin, -1, 4), axis=2)          # input, seq, chan
chans = np.rint((cf + np.fft.fftfreq(4, 1 / sr)) / BW).astype(int)
order = np.argsort(chans)
chans, spec = chans[order], spec[:, :, order]
nchan, nseq = len(chans), spec.shape[1]
assert np.all(np.diff(chans) == 1)
# The imager takes channel i to be at (chan0 + i) * NDP_BW. Choose chan0 so
# that the 25 kHz channels' frequencies are matched best; the residual
# (< ~10 kHz at ~50 MHz) changes positions by < 0.02 px.
freq = chans * BW
chan0 = int(np.rint(np.mean(freq / NDP_BW - np.arange(nchan))))

# 4-bit quantisation: rms ~2 counts, clipped to +/-7
scale = 2.0 / np.sqrt(np.mean(np.abs(spec) ** 2) / 2)
q = lambda v: np.clip(np.rint(v * scale), -7, 7).astype(np.int8)
re, im = q(spec.real), q(spec.imag)
nib = ((re.astype(np.uint8) & 0xF) << 4) | (im.astype(np.uint8) & 0xF)   # CNib: re high, im low nibble
# [input][seq][chan] -> [chan][seq][stand][pol]; TBN inputs are in station antenna order (X, Y per stand)
nstand = nin // 2
feng = nib.reshape(nstand, 2, nseq, nchan).transpose(3, 2, 0, 1).copy()
feng.tofile(prefix + '.feng')

# Calibration phases as gen_phases_lwasv, with the epic_geometry phase centre
pc, _, _ = eg.array_frame()
used = eg.used_stands()
ph = np.zeros((nchan, nstand, 4), np.float32)
for i in range(nstand):
    if not used[i]:
        continue
    for pol in (0, 1):
        a = lwasv.antennas[2 * i + pol]
        d = a.cable.delay(freq) - np.dot(pc, (a.stand.x, a.stand.y, a.stand.z)) / SOL
        v = np.exp(2j * np.pi * freq * d) / np.sqrt(a.cable.gain(freq))
        ph[:, i, 2 * pol] = v.real
        ph[:, i, 2 * pol + 1] = v.imag
ph.tofile(prefix + '.phases')
t0 = float(tstart.unix) if hasattr(tstart, 'unix') else float(tstart)
json.dump(dict(chan0=chan0, nchan=nchan, nseq=nseq, unix=t0, phase_center=eg.phase_center_mode()),
          open(prefix + '.json', 'w'))
print(f"{prefix}: chan0 {chan0}, {nchan} channels, {nseq} spectra, clip fraction "
      f"{np.mean(np.abs(spec.real * scale) > 7.5):.4f}")
