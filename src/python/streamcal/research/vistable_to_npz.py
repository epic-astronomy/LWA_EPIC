"""Convert recorded GPU-tap tables (epic++ --caltap_record_dir) into the capture format of the
real-data research tools (tbf_tcc.py: R_even, R_odd, freq, chans, t, nsamp).

    python vistable_to_npz.py TABLE_DIR OUT_DIR [--pair 1] [--stride 1] [--chan-stride 1]

Each table vis_<version>.bin is a 128-byte EPICVIS1 header and complex64 R[nchan][2][256][256]
(XX, YY), each / nsamp, integrated over --caltap_interval. Two consecutive tables are
independent noise realisations of nearly the same sky, so they stand in for the even and odd
halves of a TBF capture: pseudo-capture k uses tables 2k (as R_even) and 2k+1 (as R_odd), with t
at the middle of the pair. --pair 0 writes one file per table with R_even = R_odd = the table.
NDP sequence numbers count spectra of 8192 / 196 MHz from the Unix epoch.
"""
import argparse
import glob
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

from streamcal import geometry as G  # noqa: E402

HDR = struct.Struct("<8sIIQQqIIIIQQQIIQf28s")
SEQ_S = 8192 / 196e6


def read(fn, chan_stride):
    with open(fn, "rb") as f:
        raw = f.read(128)
        (magic, layout, hbytes, seq, version, chan0, nchan, nstand, npol, flags, nsamp, seq_first, seq_last,
         ngulps, nskipped, gulp_nseq, duty, note) = HDR.unpack(raw)
        assert magic == b"EPICVIS1", fn
        R = np.fromfile(f, np.complex64, nchan * npol * nstand * nstand).reshape(nchan, npol, nstand, nstand)
    keep = np.arange(0, nchan, chan_stride)
    t_mid = (seq_first + seq_last + gulp_nseq) / 2 * SEQ_S
    return dict(R=R[keep], chans=chan0 + keep, t=t_mid, nsamp=nsamp, ngulps=ngulps, nskipped=nskipped,
                version=version, note=note.split(b"\0")[0].decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables")
    ap.add_argument("out")
    ap.add_argument("--pair", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth pseudo-capture")
    ap.add_argument("--chan-stride", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.tables, "vis_*.bin")))
    groups = [files[i:i + 2] for i in range(0, len(files) - 1, 2)] if a.pair else [[f] for f in files]
    for k, grp in enumerate(groups[::a.stride]):
        tabs = [read(f, a.chan_stride) for f in grp]
        e, o = tabs[0], tabs[-1]
        t = Time(np.mean([x["t"] for x in tabs]), format="unix")
        out = os.path.join(a.out, f"vt_{e['version']:06d}.npz")
        np.savez(out, R_even=e["R"], R_odd=o["R"], chans=e["chans"], freq=e["chans"] * G.CHAN_BW,
                 t=t.isot, nsamp=int(min(x["nsamp"] for x in tabs)),
                 ngulps=np.array([x["ngulps"] for x in tabs]), nskipped=np.array([x["nskipped"] for x in tabs]))
        print(out, t.isot, f"nsamp {e['nsamp']}, gulps {[x['ngulps'] for x in tabs]}, {e['note']}", flush=True)


if __name__ == "__main__":
    main()
