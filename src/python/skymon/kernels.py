"""
ctypes bindings to the compiled kernels in _kernels.c.

The shared library is built on first use into a per-host cache directory
(gcc -O3 -march=x86-64-v3 -fopenmp), so the same checkout on the shared file system serves
every worker. ctypes releases the GIL during each call.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import socket
import subprocess
from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("_kernels.c")
_lib = None

f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")


def _cache_dir() -> Path:
    base = os.environ.get("SKYMON_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "skymon")
    return Path(base) / socket.gethostname()


def build(force: bool = False) -> Path:
    src = _SRC.read_bytes()
    tag = hashlib.sha1(src).hexdigest()[:12]
    out = _cache_dir() / f"skymon_kernels_{tag}.so"
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    cmd = ["gcc", "-O3", "-march=x86-64-v3", "-ffast-math", "-fno-finite-math-only",
           "-fopenmp", "-shared", "-fPIC", "-o", str(tmp), str(_SRC), "-lm"]
    subprocess.run(cmd, check=True)
    os.replace(tmp, out)
    return out


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(str(build()))
        L.skymon_fdmt.argtypes = [f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, i32p, f32p,
                                  ctypes.c_int, ctypes.c_int]
        L.skymon_fdmt.restype = ctypes.c_int
        L.skymon_search.argtypes = [f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, i32p,
                                    ctypes.c_int, ctypes.c_int, i32p, ctypes.c_int, f32p,
                                    ctypes.c_int, ctypes.c_float, f32p, i32p, f64p, i64p,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.skymon_search.restype = ctypes.c_int
        L.skymon_regrid.argtypes = [f32p, ctypes.c_int, ctypes.c_int, i32p, f32p, ctypes.c_int,
                                    ctypes.c_int, f32p, ctypes.c_int]
        L.skymon_regrid.restype = None
        u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
        fnull = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
        L.skymon_normalise.argtypes = [f32p, ctypes.c_int, ctypes.c_int, f32p, f32p, f32p, i32p, u8p,
                                       ctypes.c_void_p, i32p, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                       ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int]
        L.skymon_normalise.restype = ctypes.c_int
        L.skymon_tier_push.argtypes = [f32p, ctypes.c_int, ctypes.c_int, f32p, ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                       ctypes.c_void_p, ctypes.c_int]
        L.skymon_tier_push.restype = None
        _lib = L
    return _lib


def flatten_plan(plan) -> np.ndarray:
    init_nd, levels = plan
    out = list(init_nd)
    for level in levels:
        for nd, d_lo, d_up in level:
            out.append(nd)
            out.extend(int(v) for v in d_lo)
            out.extend(int(v) for v in d_up)
    return np.asarray(out, dtype=np.int32)


def fdmt(data: np.ndarray, flat_plan: np.ndarray, max_delay: int, nthreads: int = 1,
         ptile: int = 16) -> np.ndarray:
    C, W, P = data.shape
    out = np.empty((max_delay + 1, W, P), dtype=np.float32)
    rows = lib().skymon_fdmt(np.ascontiguousarray(data, np.float32), C, W, P, flat_plan, out,
                             nthreads, ptile)
    assert rows == max_delay + 1, (rows, max_delay)
    return out


def search(data, flat_plan, max_delay, row0, nb, widths, sigma, seg, clip=5.0, nthreads=1,
           ptile=16, detrend=True, zerodm=False):
    """Returns best_snr [P, nseg], best_info [P, nseg, 3], sumsq, cnt [D + 1, nw]"""
    C, W, P = data.shape
    nw = len(widths)
    nseg = (nb + seg - 1) // seg
    best = np.empty((P, nseg), np.float32)
    info = np.empty((P, nseg, 3), np.int32)
    sumsq = np.empty((max_delay + 1, nw), np.float64)
    cnt = np.empty((max_delay + 1, nw), np.int64)
    lib().skymon_search(data, C, W, P, flat_plan, row0, nb, np.asarray(widths, np.int32), nw,
                        np.ascontiguousarray(sigma, np.float32), seg, clip, best, info, sumsq,
                        cnt, nthreads, ptile, int(detrend), int(zerodm))
    return best, info, sumsq, cnt


def regrid(img: np.ndarray, idx: np.ndarray, w: np.ndarray, out: np.ndarray, nthreads=1):
    C = img.shape[0]
    NN = img.shape[-1] * img.shape[-2] if img.ndim == 3 else img.shape[1]
    lib().skymon_regrid(np.ascontiguousarray(img, np.float32), C, NN,
                        np.ascontiguousarray(idx, np.int32), np.ascontiguousarray(w, np.float32),
                        idx.shape[0], idx.shape[1], out, nthreads)
    return out


def _ptr(a):
    return None if a is None else a.ctypes.data_as(ctypes.c_void_p)


def normalise(raw, mu, var, prev, age, live, acc, sub, warmup, a_mu, a_var, clip, med=None,
              nthreads=1, chan_flag_sigma=math.inf, chan_mask=None, flags=None, spread=None,
              held=None, max_hold=0):
    """
    In-place normalisation of raw [C, P] with interference flagging; see
    skymon_normalise in _kernels.c. Optional arrays: med, spread [C] float32
    and flags [C] uint8 (outputs), chan_mask [C] uint8 (channels to set to
    zero), held [C, P] uint16 (state of the withheld statistics updates).
    """
    C, P = raw.shape
    for a, dt, shape in ((med, np.float32, (C,)), (spread, np.float32, (C,)),
                         (flags, np.uint8, (C,)), (chan_mask, np.uint8, (C,)),
                         (held, np.uint16, (C, P))):
        if a is not None:
            assert a.dtype == dt and a.shape == shape and a.flags.c_contiguous, (a.dtype, a.shape)
    return lib().skymon_normalise(raw, C, P, mu, var, prev, age, live, _ptr(acc), sub, len(sub),
                                  warmup, a_mu, a_var, clip, _ptr(med),
                                  float(min(chan_flag_sigma, 3e38)), _ptr(chan_mask), _ptr(flags),
                                  _ptr(spread), _ptr(held), int(max_hold), nthreads)


def tier_push(x, base, hist, pos, a_hp, dec=None, nthreads=1):
    C, W, P = hist.shape
    assert hist.flags.c_contiguous and x.shape == (C, P)
    lib().skymon_tier_push(x, C, P, base, _ptr(hist), W, pos, a_hp, _ptr(dec), nthreads)
