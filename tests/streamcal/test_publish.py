"""The streaming calibrator's publication policy (StreamCal.publish_reason).
    PYTHONPATH=src/python python -m pytest tests/streamcal/test_publish.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "python"))
from streamcal import daemon as D  # noqa: E402


def policy(**kw):
    a = argparse.Namespace(publish_warmup=300.0, publish_converging="hold", publish_min_s=60.0,
                           publish_max_s=600.0, publish_phase_deg=1.0, publish_bright_pct=0.1,
                           publish_amp_pct=2.0)
    for k, v in kw.items():
        setattr(a, k, v)
    sc = object.__new__(D.StreamCal)
    sc.a = a
    sc.published = None
    return sc


def run(sc, tables, times):
    """publish_reason over a sequence, publishing as the daemon does; the reasons"""
    out = []
    for c, t in zip(tables, times):
        why = sc.publish_reason(c, t)
        if why is not None:
            sc.published, sc.published_t = c, t
        out.append(why)
    return out


def tables(n, drift_deg, seed=0):
    """n tables [chan, stand, pol] whose phases walk by drift_deg per table"""
    rng = np.random.default_rng(seed)
    ph = np.cumsum(rng.normal(0, np.radians(drift_deg), (n, 8, 32, 2)), axis=0)
    return [np.exp(1j * p).astype(np.complex64) for p in ph]


def test_hold_while_converging_then_one_table():
    sc = policy()
    times = 1000.0 + 10.0 * np.arange(40)
    why = run(sc, tables(40, 3.0), times)
    assert all(w is None for w, t in zip(why, times) if t - times[0] < 300)
    first = next(i for i, w in enumerate(why) if w is not None)
    assert why[first] == "converged" and times[first] - times[0] == 300


def test_minimum_interval_except_for_flags():
    sc = policy(publish_warmup=0.0)
    cs = tables(30, 5.0)  # every table differs by more than the thresholds
    times = 10.0 * np.arange(30)
    why = run(sc, cs, times)
    pub = [t for w, t in zip(why, times) if w is not None]
    assert np.all(np.diff(pub) >= 60)
    # a stand flagged: published at once
    sc2 = policy(publish_warmup=0.0)
    run(sc2, cs[:1], times[:1])
    flagged = cs[1].copy()
    flagged[:, 3, 0] = 0
    assert sc2.publish_reason(flagged, 10.0) == "flags"


def test_every_table_while_converging_on_request():
    sc = policy(publish_converging="every", publish_min_s=0.0)
    times = 10.0 * np.arange(10)
    why = run(sc, tables(10, 3.0), times)
    assert why[0] == "first" and all(w == "converging" for w in why[1:])
