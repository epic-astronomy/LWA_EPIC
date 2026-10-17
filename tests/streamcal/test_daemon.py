"""End-to-end test of the streamcal daemon on synthetic calibration-tap tables.

Writes visibility tables in the epic++ CalTap layout (raw F-engine convention:
x = (eps / c_cable) s), runs the daemon for a few intervals, reads the gain
table it publishes and checks that corr * eps is constant across stands, i.e.
that epic++ (which applies geometry x cable x corr) would undo the true gains.

    PYTHONPATH=src/python python tests/streamcal/test_daemon.py /dev/shm/streamcal_e2e
"""
import os
import struct
import sys
import tempfile

import numpy as np
from astropy.time import Time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "python"))
from streamcal import daemon as D, forward as F, geometry as G, shm  # noqa: E402


def wishart(R, T, rng):
    n = R.shape[0]
    L = np.linalg.cholesky(R + 1e-9 * np.trace(R).real / n * np.eye(n))
    A = np.tril((rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2), -1)
    A[np.diag_indices(n)] = np.sqrt(rng.gamma(T - np.arange(n), 1.0))
    LA = L @ A
    return LA @ LA.conj().T / T


def write_vis(path, version, chan0, nchan, R, nsamp, seq_first, seq_last, gulp_nseq=1000):
    hdr = shm.VIS_HDR.pack(b"EPICVIS1", 1, 128, 2 * version, version, chan0, nchan, 256, 2, 0, nsamp, seq_first,
                           seq_last, 10, 0, gulp_nseq, nsamp / ((seq_last - seq_first + gulp_nseq) or 1), b"test")
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(R.astype(np.complex64).tobytes())


def main(d):
    os.makedirs(d, exist_ok=True)
    vis, gains = f"{d}/vis", f"{d}/gains"
    for p in (vis, gains):
        if os.path.exists(p):
            os.remove(p)
    rng = np.random.default_rng(7)
    chan0, nchan, stride = 3104, 128, 16
    t0 = Time("2026-10-16T06:00:00")
    a = D.argparse.Namespace(vis=vis, gains=gains, dry_run=False, nside=64, model_refresh=60.0, chan_stride=stride,
                             umin=1.0, niter=40, max_chi2=3.0, min_power=0.3, max_power=3.0, flag_amp=0.15,
                             max_abs=32.0, q_phase_deg=2.0, q_delay_ns=0.05, interval=10.0, save="",
                             passes=12, passes_warm=4, workers=4, no_positions=False, fixed_time="")
    sc = D.StreamCal(a)
    ant = sc.ant
    n = len(ant)
    solve_idx = np.arange(stride // 2, nchan, stride)
    freq = (chan0 + solve_idx) * G.CHAN_BW
    # true residual gains: amplitude, phase offset, residual delay, per pol
    amp = np.exp(rng.normal(0, 0.15, (2, n)))
    ph0 = rng.uniform(-1, 1, (2, n)) * np.radians(60)
    tau = rng.normal(0, 3e-9, (2, n))
    dead = rng.choice(n, 2, replace=False)
    eps = amp[:, :, None] * np.exp(1j * (ph0[:, :, None] + 2 * np.pi * (freq - freq.mean())[None, None, :] * tau[:, :, None]))
    eps[:, dead, :] *= 0.01
    seq_s = 8192 / 196e6
    nsamp = int(0.05 * 10 / seq_s)  # 5% tap over 10 s
    truth = [F.build_snapshot(t0, freq / G.CHAN_BW, pol=p, nside=128, extra_sources=F.random_extra_sources())
             for p in ("XX", "YY")]
    errs = []
    for k in range(6):
        t = t0 + k * 10 / 86400
        R = np.zeros((nchan, 2, 256, 256), np.complex64)
        for p in range(2):
            for j, ci in enumerate(solve_idx):
                M = truth[p].matrix(ant, j)
                M = M + 0.15 * np.real(np.diag(M)).mean() * np.eye(n)
                cc = D.cable_correction(freq[j], p)[ant]
                e = eps[p, :, j] / cc
                Rt = e[:, None] * M * np.conj(e)[None, :]
                R[ci, p][np.ix_(ant, ant)] = wishart(Rt, nsamp, rng)
        seq_first = int(t.unix / seq_s) - 120000
        write_vis(vis, k + 1, chan0, nchan, R, nsamp, seq_first, seq_first + 238000)
        assert sc.step(), "daemon did not pick up the table"
        # read what epic++ would read
        raw = np.fromfile(gains, np.uint8)
        hdr = shm.GAIN_HDR.unpack_from(raw, 0)
        corr = np.frombuffer(raw[128:].tobytes(), np.complex64).reshape(nchan, 256, 2)
        err = []
        for p in range(2):
            live = np.setdiff1d(np.arange(n), dead)
            prod = corr[solve_idx][:, ant[live], p] * eps[p, live, :].T  # should be constant per channel
            prod = prod / np.mean(prod, axis=1, keepdims=True)
            err.append(np.degrees(np.sqrt(np.mean(np.angle(prod) ** 2))))
            err.append(100 * np.std(np.abs(prod)))
            flagged = np.all(corr[:, ant[dead], p] == 0)
            err.append(flagged)
        errs.append(err)
        print(f"interval {k}: gain version {hdr[4]}; X: phase err {err[0]:.2f} deg, amp err {err[1]:.1f}%, "
              f"dead flagged {err[2]};  Y: {err[3]:.2f} deg, {err[4]:.1f}%, dead flagged {err[5]}", flush=True)
    last = errs[-1]
    ok = last[0] < 3 and last[3] < 3 and last[1] < 5 and last[4] < 5 and last[2] and last[5]
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()))
