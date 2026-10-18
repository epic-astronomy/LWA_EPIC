"""GPU interference stage (src/ex/rfi_stage.cu) against numpy, through the shared-memory projector.

Python writes a projector with streamcal.shm.ProjTableWriter and a gulp with injected interference
and impulses; rfi/rfi_stage_driver (rfi/build.sh) reads the projector with ProjTableReader, runs
RfiStage and writes the processed gulp. Checks:
  1. a table that is not a projector pair is rejected;
  2. no projector, no impulse detection: the output equals the input;
  3. projection: the interference is removed to the quantisation level and the output matches the
     float projection to within the dithered rounding (mean error ~ 0, rms ~ sqrt(1/6));
  4. impulse detection: exactly the injected sequences are zeroed, in every stream;
  5. Sun exemption: a short solar burst (a strong signal along the Sun's response) is zeroed
     with the impulses when the table has no Sun vector, and kept when it has one.

    PYTHONPATH=src/python python tests/streamcal/test_rfi_stage.py WORKDIR
"""
import os
import subprocess
import sys

import numpy as np

from streamcal import shm

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "rfi", "rfi_stage_driver")
NCH, NSEQ, NSTREAM, NS = 8, 256, 2, 256
rng = np.random.default_rng(7)
work = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/streamcal_rfi_test"
os.makedirs(work, exist_ok=True)
proj, gulp_f, out_f = (os.path.join(work, n) for n in ("proj", "gulp.bin", "out.bin"))


def unpack(b):
    b = b.astype(np.uint8)
    return ((b.view(np.int8) >> 4) + 1j * ((b << 4).view(np.int8) >> 4)).astype(np.complex64)


def pack(z):
    re = np.clip(np.rint(z.real), -7, 7).astype(np.int8)
    im = np.clip(np.rint(z.imag), -7, 7).astype(np.int8)
    return (((re.astype(np.uint8) & 0xF) << 4) | (im.astype(np.uint8) & 0xF)).astype(np.uint8)


def run(sigma):
    if os.path.exists(out_f):
        os.remove(out_f)
    r = subprocess.run([DRIVER, proj, gulp_f, out_f, str(NCH), str(NSEQ), str(NSTREAM), str(sigma)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


# interference subspace per channel and polarisation (k = 2 in X, 3 in Y), input powers D
kp = (2, 3)
Ut = np.zeros((NCH, 4, 512), np.complex64)
Vt = np.zeros_like(Ut)
x = np.zeros((NCH, NSEQ, 512), np.complex64)
for c in range(NCH):
    for p in (0, 1):
        k = kp[p]
        U, _ = np.linalg.qr(rng.standard_normal((NS, k)) + 1j * rng.standard_normal((NS, k)))
        d = rng.uniform(4, 12, NS)
        dh = np.sqrt(d)
        Ut[c, :k, p::2] = (dh[:, None] * U).T
        Vt[c, :k, p::2] = (U / dh[:, None]).T
        noise = (rng.standard_normal((NSEQ, NS)) + 1j * rng.standard_normal((NSEQ, NS))) / np.sqrt(2)
        amp = 3.0 * (rng.standard_normal((NSEQ, k)) + 1j * rng.standard_normal((NSEQ, k))) / np.sqrt(2)
        x[c, :, p::2] = dh[None, :] * (0.8 * noise + amp @ U.T * np.sqrt(NS) / 4)
imp = [50, 51, 200]
x[:, imp, :] *= 4.0
gulp = pack(x)
gulp.tofile(gulp_f)
xq = unpack(gulp)
ok = True

# 1. a corrupted table is rejected
w = shm.ProjTableWriter(proj, 3168, NCH, qmax=4, solver="test")
w.publish(Ut, 2 * Vt)
rc, msg = run(0)
print("corrupted table:", "rejected" if rc == 3 else f"ACCEPTED ({msg.strip()})")
ok &= rc == 3 and "not a projector pair" in msg

# 2. projection off, no impulse detection: identity
w.publish(None)
rc, msg = run(0)
out = np.fromfile(out_f, np.uint8).reshape(gulp.shape) if rc == 0 else None
same = out is not None and np.array_equal(out, gulp)
print("identity:", msg.strip(), "->", "exact" if same else "DIFFERENT")
ok &= same

# 3. projection
w.publish(Ut, Vt)
rc, msg = run(0)
y = unpack(np.fromfile(out_f, np.uint8).reshape(gulp.shape))
print("projection:", msg.strip())
keep = np.setdiff1d(np.arange(NSEQ), imp)
before, after, err = [], [], []
for c in range(NCH):
    for p in (0, 1):
        k = kp[p]
        V, U = Vt[c, :k, p::2], Ut[c, :k, p::2]
        xs, ys = xq[c][keep][:, p::2], y[c][keep][:, p::2]
        sx, sy = xs @ V.conj().T, ys @ V.conj().T  # [seq, k]
        before.append(np.mean(np.abs(sx) ** 2))
        after.append(np.mean(np.abs(sy) ** 2))
        ref = xs - sx @ U
        live = np.abs(ref.real) < 6.5
        live &= np.abs(ref.imag) < 6.5
        err.append((ys - ref)[live])
err = np.concatenate(err)
supp = np.median(np.array(after) / np.array(before))
mean_err, rms_err = abs(err.mean()), np.sqrt(np.mean(np.abs(err) ** 2) / 2)
print(f"  interference power in the projected subspace: {supp:.2e} of the input; error mean {mean_err:.3f}, "
      f"rms per component {rms_err:.3f} (dithered rounding: {np.sqrt(1 / 6):.3f})")
ok &= supp < 0.02 and mean_err < 0.02 and 0.3 < rms_err < 0.5

# 4. impulse detection, with the projection
rc, msg = run(6.0)
z = np.fromfile(out_f, np.uint8).reshape(gulp.shape)
zero_rows = sorted(set(np.flatnonzero(np.all(z == 0, axis=(0, 2)))))
print("impulses:", msg.strip(), "-> zeroed sequences", zero_rows, "injected", imp)
nflag = int(msg.split("flagged")[1].split()[0]) if "flagged" in msg else -1
ok &= zero_rows == imp and nflag == len(imp) * NSTREAM

# 5. Sun exemption. Lower base level, so the burst does not saturate the 4-bit samples.
burst = [120, 121, 122, 123]
x5 = 0.5 * x.copy()
x5[:, imp, :] *= 4.0  # impulses: x was already scaled by 4 there, now 8x the base
sun = np.zeros((NCH, 512), np.complex64)
for c in range(NCH):
    for p in (0, 1):
        dh = np.abs(Ut[c, 0, p::2]) / np.maximum(np.abs(Vt[c, 0, p::2]), 1e-30)  # = d
        dh = np.sqrt(dh)
        v = dh * np.exp(2j * np.pi * rng.uniform(size=NS))  # a compact source: phase-only times sqrt(power)
        k = kp[p]
        vp = v - Ut[c, :k, p::2].T @ (Vt[c, :k, p::2].conj() @ v)
        sun[c, p::2] = vp / np.linalg.norm(vp)
        amp = 1.6 * (rng.standard_normal(len(burst)) + 1j * rng.standard_normal(len(burst))) / np.sqrt(2)
        x5[c][np.ix_(burst, np.arange(p, 512, 2))] += amp[:, None] * v[None, :]
g5 = pack(x5)
gulp5_f = os.path.join(work, "gulp5.bin")
g5.tofile(gulp5_f)
for with_sun in (False, True):
    w.publish(Ut, Vt, sun=sun if with_sun else None)
    r = subprocess.run([DRIVER, proj, gulp5_f, out_f, str(NCH), str(NSEQ), str(NSTREAM), "6"], capture_output=True,
                       text=True)
    z = np.fromfile(out_f, np.uint8).reshape(g5.shape)
    zr = sorted(set(np.flatnonzero(np.all(z == 0, axis=(0, 2)))))
    want = sorted(imp) if with_sun else sorted(imp + burst)
    print(f"solar burst, {'with' if with_sun else 'without'} Sun vector: zeroed {zr}, expected {want}")
    ok &= zr == want

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
