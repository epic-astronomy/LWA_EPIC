"""Cut a few channels out of the full TBF correlation files (tbf_tcc.py output).

    python tbf_subset.py OUTDIR CHANS FILE.npz [FILE.npz ...]

CHANS: ADP channel numbers, comma separated (for example 1236,1428,1620,1812).
Writes OUTDIR/<name>.npz with the same keys (R_even, R_odd [nchan, 2, 256, 256], freq,
chans, t, nsamp, nvalid) restricted to those channels, so that dd_kernelcal_real.py and
diffuse_eval.py read 4 channels instead of 792 (1.7 GB per capture).
"""
import os
import sys

import numpy as np


def main(outdir, chans, files):
    os.makedirs(outdir, exist_ok=True)
    want = [int(c) for c in chans.split(",")]
    for fn in files:
        z = np.load(fn)
        idx = [int(np.flatnonzero(z["chans"] == c)[0]) for c in want]
        out = {k: z[k][idx] for k in ("R_even", "R_odd", "freq", "chans", "nvalid")}
        out.update(t=z["t"], nsamp=z["nsamp"])
        np.savez(os.path.join(outdir, os.path.basename(fn)), **out)
        print(os.path.basename(fn), str(z["t"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
