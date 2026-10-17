"""Summarise beam_fit.py output: correction tables, hold-out metrics, A-team scales, consistency.

    python beam_fit_report.py OUT.npy [OUT2.npy ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from streamcal import beamcorr as BC  # noqa: E402

ELS = (20, 30, 45, 60, 75)
AZS = ((0, "N"), (90, "E"), (180, "S"), (270, "W"))


def ctable(p, terms):
    out = {}
    for el in ELS:
        out[el] = [float(BC.correction(BC.enu_from_azel(az, el)[None], p, terms)[0]) for az, _ in AZS]
    return out


def corr_grid(p1, p2, terms, elmin=20):
    """rms of log C difference over the sky above elmin (equal solid angle) and correlation."""
    el = np.linspace(elmin, 90, 40)
    az = np.linspace(0, 360, 72, endpoint=False)
    E, A = np.meshgrid(el, az)
    w = np.cos(np.radians(E)).ravel()
    enu = BC.enu_from_azel(A.ravel(), E.ravel())
    Y = BC.basis(enu, terms)
    l1, l2 = p1 @ Y, p2 @ Y
    rms = np.sqrt(np.sum(w * (l1 - l2) ** 2) / w.sum())
    c = np.sum(w * l1 * l2) / np.sqrt(np.sum(w * l1 ** 2) * np.sum(w * l2 ** 2))
    return rms, c


def drift(rows, tag, nm):
    s, e = [], []
    for x in rows:
        if nm in x["names"] and nm not in x[tag]["frozen"]:
            s.append(x[tag]["scales"][nm])
            e.append(x["el"][x["names"].index(nm)])
    if len(s) < 3:
        return None
    s, e = np.array(s), np.array(e)
    slope = np.polyfit(e / 10, s, 1)[0]
    mad = 1.4826 * np.median(np.abs(s - np.median(s)))
    return np.median(s), slope, mad, len(s), e.min(), e.max()


def report(fn):
    res = np.load(fn, allow_pickle=True).item()
    meta = res["meta"]
    terms = tuple(meta["terms"])
    names = [BC.NAMES[j] for j in terms]
    print(f"\n######## {fn}: basis {meta['basis']} {names}, diffuse {meta['diffuse']} nside {meta['nside']}, "
          f"sim {meta['sim']}")
    keys = [k for k in res if k != "meta"]
    keys.sort(key=lambda k: (int(k.split("_")[1]), int(k.split("_")[0])))
    print("\n## p (sum half), even/odd")
    for k in keys:
        v = res[k]
        line = f"{v['freq'] / 1e6:5.1f} {'XY'[int(k.split('_')[1])]}  sum [{' '.join(f'{x:+.3f}' for x in v['fit']['sum']['p'])}]"
        if "even" in v["fit"]:
            rms, c = corr_grid(v["fit"]["even"]["p"], v["fit"]["odd"]["p"], terms)
            line += (f"  even [{' '.join(f'{x:+.3f}' for x in v['fit']['even']['p'])}]"
                     f"  odd [{' '.join(f'{x:+.3f}' for x in v['fit']['odd']['p'])}]  even-odd rms dlogC {rms:.3f}")
        if meta["sim"]:
            if len(terms) == 6:
                rms, c = corr_grid(v["fit"]["sum"]["p"], np.asarray(meta["sim_p"]), terms)
                line += f"  vs truth: rms dlogC {rms:.3f} (el>20)"
        print(line)
    print("\n## C at (el; N E S W), sum half")
    for k in keys:
        v = res[k]
        t = ctable(v["fit"]["sum"]["p"], terms)
        print(f"{v['freq'] / 1e6:5.1f} {'XY'[int(k.split('_')[1])]}  " +
              "  ".join(f"{el}: {' '.join(f'{c:.2f}' for c in t[el])}" for el in ELS))
    if meta["sim"] and len(terms) == 6:
        t = ctable(np.asarray(meta["sim_p"]), terms)
        print("truth  " + "  ".join(f"{el}: {' '.join(f'{c:.2f}' for c in t[el])}" for el in ELS))
    print("\n## channel-to-channel agreement (rms dlogC el>20, correlation)")
    for pol in (0, 1):
        kk = [k for k in keys if int(k.split("_")[1]) == pol]
        for i in range(len(kk)):
            for j in range(i + 1, len(kk)):
                rms, c = corr_grid(res[kk[i]]["fit"]["sum"]["p"], res[kk[j]]["fit"]["sum"]["p"], terms)
                print(f"  {'XY'[pol]} {res[kk[i]]['freq'] / 1e6:.1f}-{res[kk[j]]['freq'] / 1e6:.1f}: rms {rms:.3f} corr {c:+.2f}")
    print("\n## hold-out (test) and training: medians, uncorrected -> corrected")
    print(f"{'MHz':>5} {'pol':>3} {'set':>5} | {'op chi2':>13} {'op resid':>15} {'op xerr':>15} | "
          f"{'fix chi2':>13} {'fix resid':>15} {'fix xerr':>15}")
    agg = {}
    for k in keys:
        v = res[k]
        for lab in ("test", "train"):
            rr = [x for x in v["rows"] if x["test"] == (lab == "test")]
            m = {f"{tg}_{q}": float(np.median([x[tg][q] for x in rr])) for tg in ("op0", "op1", "fix0", "fix1")
                 for q in ("chi2", "resid", "xerr")}
            agg.setdefault(lab, []).append((k, m))
            print(f"{v['freq'] / 1e6:5.1f} {'XY'[int(k.split('_')[1])]:>3} {lab:>5} | "
                  f"{m['op0_chi2']:.3f}->{m['op1_chi2']:.3f} {m['op0_resid']:.4f}->{m['op1_resid']:.4f} "
                  f"{m['op0_xerr']:.4f}->{m['op1_xerr']:.4f} | {m['fix0_chi2']:.3f}->{m['fix1_chi2']:.3f} "
                  f"{m['fix0_resid']:.4f}->{m['fix1_resid']:.4f} {m['fix0_xerr']:.4f}->{m['fix1_xerr']:.4f}")
    for lab in ("test", "train"):
        allrows = [x for k in keys for x in res[k]["rows"] if x["test"] == (lab == "test")]
        s = "  ".join(f"{tg} chi2 {np.median([x[tg]['chi2'] for x in allrows]):.3f} resid "
                      f"{np.median([x[tg]['resid'] for x in allrows]):.4f} xerr {np.median([x[tg]['xerr'] for x in allrows]):.4f}"
                      for tg in ("op0", "op1", "fix0", "fix1"))
        print(f"ALL {lab}: {s}")
        nb = sum(1 for k, m in agg[lab] if m["op1_xerr"] < m["op0_xerr"])
        print(f"   {lab}: op xerr improved in {nb}/{len(agg[lab])} channel-pol; "
              f"fix xerr improved in {sum(1 for k, m in agg[lab] if m['fix1_xerr'] < m['fix0_xerr'])}/{len(agg[lab])}")
    print("\n## X/Y asymmetry: excess chi2 and xerr, X / Y (op, all snapshots)")
    for k in keys:
        if not k.endswith("_0"):
            continue
        ky = k[:-1] + "1"
        if ky not in res:
            continue
        out = []
        for tg in ("op0", "op1"):
            ex = [np.median([x[tg]["chi2"] for x in res[kk]["rows"]]) - 1 for kk in (k, ky)]
            xe = [np.median([x[tg]["xerr"] for x in res[kk]["rows"]]) for kk in (k, ky)]
            out.append(f"{tg}: excess X {ex[0]:.3f} Y {ex[1]:.3f} ratio {ex[0] / ex[1]:.2f}; xerr X {xe[0]:.4f} Y {xe[1]:.4f} ratio {xe[0] / xe[1]:.2f}")
        print(f"{res[k]['freq'] / 1e6:5.1f}  " + " | ".join(out))
    print("\n## shared scales (training) and per-snapshot op scales: median, slope per 10 deg el, robust sd, n, el range")
    for k in keys:
        v = res[k]
        print(f"{v['freq'] / 1e6:5.1f} {'XY'[int(k.split('_')[1])]}  shared p=0 "
              f"{ {a: round(b, 3) for a, b in v['fit0']['s'].items()} }  with C { {a: round(b, 3) for a, b in v['fit']['sum']['s'].items()} }")
        for nm in ("CasA", "CygA", "TauA", "VirA"):
            d0 = drift(v["rows"], "op0", nm)
            d1 = drift(v["rows"], "op1", nm)
            f = lambda d: "n/a" if d is None else f"{d[0]:.2f} slope {d[1]:+.3f} sd {d[2]:.2f} (n {d[3]}, el {d[4]:.0f}-{d[5]:.0f})"  # noqa: E731
            print(f"      {nm}: {f(d0)}  ->  {f(d1)}")


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        report(fn)
