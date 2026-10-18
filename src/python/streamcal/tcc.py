"""Correlate with the ASTRON Tensor-Core Correlator through tests/streamcal/tcc/tcc_correlate.

correlate_i8(x) takes integer-valued complex samples x[chan, recv, pol, t] (|re|, |im| <= 127)
and returns R[chan, polY, polX, recv, recv] = sum_t x[recv_i, polY] conj(x[recv_j, polX]),
full Hermitian in the receiver indices. Conventions were fixed against numpy
(tests/streamcal/test_tcc.py).
"""
import os
import subprocess
import tempfile

import numpy as np

TOOL = os.environ.get("TCC_CORRELATE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                                    "tests", "streamcal", "tcc", "tcc_correlate"))


def to_tcc_i8(x, T=16):
    """x[chan, recv, pol, t] complex (integer valued) -> bytes in TCC i8 layout [chan][t/T][recv][pol][T][re, im]."""
    nch, nr, npol, nt = x.shape
    nt = nt // T * T
    re = np.clip(np.rint(x.real[..., :nt]), -127, 127).astype(np.int8)
    im = np.clip(np.rint(x.imag[..., :nt]), -127, 127).astype(np.int8)
    z = np.stack([re, im], -1).reshape(nch, nr, npol, nt // T, T, 2)
    return np.ascontiguousarray(z.transpose(0, 3, 1, 2, 4, 5)).tobytes(), nt


def correlate_i8(x, workdir=None):
    nch, nr, npol, _ = x.shape
    buf, nt = to_tcc_i8(x)
    workdir = workdir or ("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir())
    fin = tempfile.NamedTemporaryFile(dir=workdir, suffix=".tccin", delete=False)
    fin.write(buf)
    fin.close()
    fout = fin.name + ".vis"
    try:
        out = subprocess.run([TOOL, "i8", str(nr), str(nch), str(nt), fin.name, fout], check=True,
                             capture_output=True, text=True).stdout
        v = np.fromfile(fout, np.int32).reshape(nch, nr * (nr + 1) // 2, 2, 2, 2)
    finally:
        os.remove(fin.name)
        if os.path.exists(fout):
            os.remove(fout)
    v = v[..., 0] + 1j * v[..., 1]  # [chan, bl, polY, polX]
    iX, iY = np.tril_indices(nr)  # baseline = recvX (recvX + 1) / 2 + recvY with recvY <= recvX: row-major lower triangle
    R = np.zeros((nch, 2, 2, nr, nr), complex)
    # convention (verified): v[bl, pY, pX] = sum x[recvY, pY] conj(x[recvX, pX])
    R[:, :, :, iY, iX] = v.transpose(0, 2, 3, 1)
    R[:, :, :, iX, iY] = np.conj(v.transpose(0, 3, 2, 1))
    return R, nt, out.strip()
