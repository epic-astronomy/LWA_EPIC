"""
Is the pattern of detections and non-detections across the bands one that a
real burst can produce?

A burst need not be seen in every band that can see its position:

- its spectrum may be steep, or confined to a few MHz, as is common for
  repeating FRBs;
- scattering grows as nu^-4 and can spread the pulse at the lowest frequencies
  far beyond the search's boxcars, so the S/N falls although the fluence does
  not.

Each band of an event gives a value: the S/N of a detection or of a
followup measurement (any value, with an error of 1), or, where only the
band's blind search covered the arrival, an upper limit at its threshold.
These are compared with three families of models, in units of S/N:

  power law       A (nu / nu_ref)^alpha,                    alpha in [-10, 10]
  band-limited    A exp(-(nu - nu_0)^2 / (2 (f nu_0)^2)),   f in [0.05, 0.3]
  either, scattered: times sqrt(w / sqrt(w^2 + tau(nu)^2)), tau = tau_1GHz (nu / 1 GHz)^-4

The fit gives the smallest chi^2 of each family and the range of parameters
that fit. A pattern no model fits (for instance a bright detection with a
deep limit in a band a few MHz away) is implausible for a burst from the sky.
Events are never discarded on this basis: implausible events are kept and
reviewed last.
"""
from __future__ import annotations

import math

import numpy as np

ALPHAS = np.linspace(-10, 10, 81)
FRACS = np.array([0.05, 0.1, 0.2, 0.3])
NU0S = np.arange(10.0, 100.5, 0.5)
TAUS = np.concatenate([[0.0], np.logspace(-8, -4, 17)])  # tau at 1 GHz, s


def band_values(ev, threshold=8.0):
    """[(name, nu_MHz, kind, value, width_s)] from an event; kind is 'meas' or 'limit'"""
    out = []
    for name, b in ev["bands"].items():
        st = b.get("state")
        if st == "detected":
            c = b["candidate"]
            nu = 0.5 * (c["f_lo_mhz"] + c["f_hi_mhz"])
            out.append((name, nu, "meas", float(c["snr"]), max(c["width_s"], c["tier_dt"])))
        elif st == "followup" and b.get("followup") and "f_lo" in b:
            nu = 0.5 * (b["f_lo"] + b["f_hi"])
            out.append((name, nu, "meas", float(b["followup"]["snr"]), 0.1))
        elif st == "searched" and "f_lo" in b and "beyond" not in str(b.get("reason", "")):
            nu = 0.5 * (b["f_lo"] + b["f_hi"])
            out.append((name, nu, "limit", float(b.get("threshold", threshold)), 0.1))
    return out


def _chi2(obs_m, kinds, obs, model):
    """Best non-negative amplitude for a model shape, and its chi^2"""
    meas = kinds == "meas"
    mm = model[meas]
    denom = float(np.sum(mm * mm))
    a0 = max(float(np.sum(obs[meas] * mm)) / denom, 0.0) if denom > 0 else 0.0
    best = math.inf
    for a in (a0, 0.8 * a0, 0.6 * a0, 0.4 * a0, 0.2 * a0):
        pred = a * model
        c = float(np.sum((obs[meas] - pred[meas]) ** 2))
        over = pred[~meas] - obs[~meas]
        c += float(np.sum(np.clip(over + 1.0, 0, None) ** 2))  # a limit binds 1 sigma below it
        best = min(best, c)
    return best


def characterise(values, chi2_margin=9.0):
    """
    Fit the families to the band values. Returns a dict with the best chi^2 of
    each family, whether any fits (plausible), and the constraints in words.
    """
    if not values:
        return dict(plausible=None, note="no band values")
    nu = np.array([v[1] for v in values])
    kinds = np.array([v[2] for v in values])
    obs = np.array([v[3] for v in values])
    w = np.array([v[4] for v in values])
    n_meas = int(np.sum(kinds == "meas"))
    ok = chi2_margin + len(values)  # generous: the aim is to keep anything a burst could be
    ref = nu[np.argmax(np.where(kinds == "meas", obs, -np.inf))]

    def supp(tau1):
        tau = tau1 * (nu / 1000.0) ** -4
        return np.sqrt(w / np.sqrt(w ** 2 + tau ** 2))

    pl = np.full((len(ALPHAS), len(TAUS)), np.inf)
    for j, t in enumerate(TAUS):
        s = supp(t)
        for i, a in enumerate(ALPHAS):
            pl[i, j] = _chi2(None, kinds, obs, (nu / ref) ** a * s)
    gs = np.full((len(NU0S), len(FRACS), len(TAUS)), np.inf)
    for k, t in enumerate(TAUS):
        s = supp(t)
        for i, n0 in enumerate(NU0S):
            for j, f in enumerate(FRACS):
                gs[i, j, k] = _chi2(None, kinds, obs,
                                    np.exp(-0.5 * ((nu - n0) / (f * n0)) ** 2) * s)
    pl_min, gs_min = float(pl.min()), float(gs.min())
    out = dict(n_values=len(values), n_measured=n_meas, chi2_power_law=round(pl_min, 2),
               chi2_band_limited=round(gs_min, 2), chi2_accept=ok,
               plausible=bool(min(pl_min, gs_min) <= ok),
               values=[dict(band=v[0], nu_mhz=round(v[1], 2), kind=v[2], snr=round(v[3], 2))
                       for v in values])
    notes = []
    a_ok = ALPHAS[np.any(pl <= ok, axis=1)]
    if len(a_ok):
        out["alpha_range"] = [float(a_ok.min()), float(a_ok.max())]
        unscattered = ALPHAS[pl[:, 0] <= ok]
        if len(unscattered):
            notes.append(f"a power law with spectral index {unscattered.min():+.1f} to "
                         f"{unscattered.max():+.1f}")
        else:
            t_ok = TAUS[np.any(pl <= ok, axis=0)]
            notes.append(f"a power law if scattered (tau at 1 GHz >= {t_ok.min() * 1e6:.2g} us)")
    g_ok = np.argwhere(gs <= ok)
    if len(g_ok):
        n0 = NU0S[g_ok[:, 0]]
        fw = FRACS[g_ok[:, 1]]
        out["band_limited_nu0"] = [float(n0.min()), float(n0.max())]
        out["band_limited_frac"] = [float(fw.min()), float(fw.max())]
        notes.append(f"emission confined near {n0.min():.0f}-{n0.max():.0f} MHz "
                     f"(width {100 * fw.min():.0f}-{100 * fw.max():.0f}% of its frequency)")
    if not notes:
        notes.append("no power law, band-limited or scattered spectrum fits")
    out["consistent_with"] = notes
    return out
