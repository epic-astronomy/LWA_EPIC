"""
Consensus of the all-sky monitors across bands.

Every imaging node runs its own monitor for its own band and stores what it
finds (monitor.py). The combiner looks at all of them together. A burst from
the sky is seen at the same position and DM in every band that can see that
position, at times that follow the dispersion law

    t_bottom(band) = t_inf + K DM / f_lo(band)^2,

so the high bands see it first and the lowest band up to hours later.
Interference near one station does not follow this law.

For each candidate a monitor reports, the combiner opens an event (or joins
the candidate to an open one), and for every other band it

1. predicts when the burst reaches the bottom of that band;
2. checks that the position is above that monitor's elevation limit for the
   whole sweep;
3. sends that monitor a watch (watch.py). The monitor measures the S/N at
   that position, DM and time whatever its value, and stores the data. When
   the arrival is still to come this is done in advance, and the threshold is
   in effect lowered from 8 to confirm_snr.

When every band is resolved, the event is classified:

  confirmed      two or more bands see it (a detection, or a followup with
                 S/N >= confirm_snr)
  band_limited   one band sees it; the S/N measured or limited in the others
                 fits a spectrum a burst can have (steep, confined to a few
                 MHz, or scattered at low frequency; spectrum.py)
  implausible    one band sees it, and no such spectrum fits the others
  unconfirmable  no other band could see it (position not visible, node down,
                 or DM beyond their search)

Nothing is discarded on these grounds: the data of every band stay on disk
(the detection, each followup measurement, and for a band the burst had
already passed, the accumulated spectrum around its arrival), and every event
is marked for review, implausible ones last. A reviewer's verdict (frb, rfi,
unclear) is recorded with the event (POST /api/events/<id>/review).

A decided event also carries the data quality of each band that detected or
measured it (conditions: the worst noise ratio, RFI rate and so on in the
minute before, conditions.py) and `disturbed` when any of those bands was
in a disturbed period.

With two or more detections, the DM and t_inf are refitted to all of them,
which pins the DM far more closely than one 3 MHz band can.

Cross-band coincidence veto. Terrestrial impulsive interference (lightning
sferics, meteor afterglows, sparks) is undispersed: it reaches every band at
the same moment, whereas a burst from the sky reaches the bands seconds to
hours apart. The combiner keeps the new candidates (and RFI) of every node for
coinc_history_s, and calls two candidates from different nodes simultaneous
when

    |t_bottom_a - t_bottom_b| <= coinc_slack_s + tier_dt_a + tier_dt_b + width_a + width_b.

A simultaneous pair is undispersed, and so terrestrial, when the dispersion
law at either candidate's DM puts the two bands further apart in time than
coinc_disp_margin_s plus the same tolerances: a burst at that DM could not
reach both at once. Positions are not compared, because interference in the
near field is imaged at different places in different bands. Bands close in
frequency, or a low DM, predict a gap too small to tell the two apart, and
such pairs are left to the consensus above.

The candidates of all bands involved form one event of kind and status
terrestrial (review not needed), with the reason. A coincident candidate
opens no sky event; a pending sky event whose detections are all coincident
is given status terrestrial with a note, and kept.

The combiner also makes coordinated injections. It asks every monitor to
inject the same burst (position, DM and t_inf), and checks that the events
come out confirmed: a measurement of the whole chain.

Every decided or reopened event also carries external context for the
reviewer (context.py): the Sun's elevation, NOAA SWPC solar radio sweeps and
X-ray flares at its arrival times, and nearby lightning when a source is
configured. These are fetched in the background and never hold up or fail the
combiner.

The combiner needs the monitors, but the monitors never need the combiner.
Its state is in its output directory (/FileStore), so it can be restarted on
any node.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .candidates import elevation_radius, separation_deg
from .dedisp import K_DM
from .geometry import altaz, beam_fwhm_deg, time_to_set
from .spectrum import band_values, characterise

SEEN_STATES = ("detected", "followup")
COINC_KEYS = ("id", "cls", "snr", "dm", "dm_step", "t_bottom", "ra", "dec", "tier_dt", "width_s",
              "f_lo_mhz", "f_hi_mhz")


@dataclass
class CombinerConfig:
    nodes: dict  # name -> monitor URL (status and live data)
    dsns: dict = field(default_factory=dict)  # name -> the node's local database (mailbox)
    out_dir: str = "skymon_combiner"
    poll_s: float = 10.0
    timeout_s: float = 3.0
    token: str | None = None
    confirm_snr: float = 6.0  # a followup at or above this S/N counts as seen
    min_bands: int = 2
    lookback_s: float = 6 * 3600.0  # how far back candidates are read each poll
    keep_hours: float = 48.0
    settle_s: float = 300.0  # wait this long past the last predicted result
    inject_interval_s: float = 0.0  # coordinated injections; 0 disables
    inject_snr: tuple = (15.0, 30.0)
    inject_dm: tuple = (50.0, 1000.0)
    inject_el_min: float = 30.0
    seed: int | None = None
    coinc_history_s: float = 900.0  # candidates kept for the coincidence veto
    coinc_slack_s: float = 1.5  # simultaneity, beyond the tiers' and widths' tolerances
    coinc_disp_margin_s: float = 3.0  # least dispersive gap that makes a pair undispersed
    # external context for the reviewers (context.py); polled only by run()
    context_swpc: bool = True
    lightning_url: str | None = None
    context_interval_s: float = 900.0
    context_refresh_s: float = 24 * 3600.0  # update the context of events decided this recently


def utc(unix: float) -> str:
    return datetime.fromtimestamp(unix, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def fit_dm(dets):
    """
    Least-squares DM and t_inf from detections [(f_lo, t_bottom, sigma_t)] in
    two or more bands: t_bottom = t_inf + DM K / f_lo^2. Returns (dm, t_inf,
    dm_err) or None.
    """
    if len({round(d[0], 3) for d in dets}) < 2:
        return None
    x = np.array([K_DM / d[0] ** 2 for d in dets])
    y = np.array([d[1] for d in dets])
    w = 1.0 / np.array([max(d[2], 1e-3) for d in dets]) ** 2
    A = np.stack([np.ones_like(x), x], 1)
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    t_inf, dm = cov @ (A.T @ (w * y))
    return float(dm), float(t_inf), float(math.sqrt(cov[1, 1]))


class Combiner:
    def __init__(self, cfg: CombinerConfig):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(cfg.seed)
        self.pool = ThreadPoolExecutor(max_workers=max(4, len(cfg.nodes)))
        from .localdb import LocalDB
        self.nodes = {n: dict(name=n, url=u.rstrip("/"), up=False, status=None, last_ok=None,
                              error=None, cursor=None,
                              db=LocalDB(cfg.dsns[n], os.path.join(cfg.out_dir, "spool", n))
                              if cfg.dsns.get(n) else None)
                      for n, u in cfg.nodes.items()}
        self.events: dict[str, dict] = {}
        self.seen: set = set()  # (node, candidate id) already handled
        self.injections: dict[str, dict] = {}
        self.next_inject = None
        self.started = time.time()
        self.recent: list = []  # [(received, node, candidate)] for the coincidence veto
        self.terrestrial: dict = {}  # (node, candidate id) -> terrestrial event id
        os.makedirs(os.path.join(cfg.out_dir, "events"), exist_ok=True)
        from .context import make_service
        self.context = make_service(cfg.context_swpc, cfg.lightning_url,
                                    os.path.join(cfg.out_dir, "context"), cfg.context_interval_s)
        self._context_gen: dict[str, int] = {}  # event id -> context generation attached
        self._load()

    # ------------------------------------------------------------ persistence
    def _event_path(self, ev):
        d = os.path.join(self.cfg.out_dir, "events",
                         datetime.fromtimestamp(ev["t_inf"], timezone.utc).strftime("%Y%m%d"))
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, ev["id"] + ".json")

    def _save(self, ev):
        ev["updated"] = time.time()
        ev["version"] = ev.get("version", 0) + 1
        self._replicate(ev)
        p = self._event_path(ev)
        tmp = p + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(ev, f, indent=1)
            os.replace(tmp, p)
        except OSError as e:  # keep going without the disk
            print(f"combiner: could not write {p}: {e}", flush=True)

    def _replicate(self, ev):
        """Copy the event to every node's database that answers (newer versions win)"""
        body = json.dumps(_finite(ev))
        for node in self.nodes.values():
            if node["db"] is None:
                continue
            node["db"].execute(
                "INSERT INTO events (id, version, status, review, t_inf, state, updated) "
                "VALUES (%s, %s, %s, %s, %s, %s, now()) ON CONFLICT (id) DO UPDATE SET "
                "version = EXCLUDED.version, status = EXCLUDED.status, review = EXCLUDED.review, "
                "t_inf = EXCLUDED.t_inf, state = EXCLUDED.state, updated = now() "
                "WHERE events.version < EXCLUDED.version",
                (ev["id"], ev["version"], ev["status"], ev.get("review"), ev["t_inf"], body))

    def _load_db(self, cutoff):
        """The newest copy of each recent event across the nodes' databases"""
        best = {}
        for node in self.nodes.values():
            if node["db"] is None:
                continue
            rows = node["db"].execute(
                "SELECT id, version, state FROM events WHERE updated > to_timestamp(%s)",
                (cutoff,), fetch=True)
            for eid, ver, state in rows or []:
                if eid not in best or ver > best[eid]["version"]:
                    best[eid] = dict(state, version=ver)
        return best

    def _load(self):
        """Reopen the events of the last keep_hours, so a restart on another node continues"""
        cutoff = time.time() - self.cfg.keep_hours * 3600
        for eid, ev in self._load_db(cutoff).items():
            self.events[eid] = ev
        root = os.path.join(self.cfg.out_dir, "events")
        for day in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            for name in os.listdir(os.path.join(root, day)):
                if not name.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(root, day, name)) as f:
                        ev = json.load(f)
                except (OSError, ValueError):
                    continue
                if ev.get("created", 0) < cutoff:
                    continue
                if ev["id"] in self.events and \
                        self.events[ev["id"]].get("version", 0) >= ev.get("version", 0):
                    continue
                self.events[ev["id"]] = ev
        for ev in self.events.values():
            for node, b in ev["bands"].items():
                if b.get("candidate"):
                    self.seen.add((node, b["candidate"]["id"]))
                for cid in b.get("others", []):
                    self.seen.add((node, cid))
        # the coincidence veto continues where it stopped
        now = time.time()
        for ev in self.events.values():
            if ev.get("kind") == "terrestrial":
                for c in ev.get("candidates", []):
                    self.terrestrial[(c["node"], c["id"])] = ev["id"]
                    self.seen.add((c["node"], c["id"]))
                    if ev["created"] >= now - self.cfg.coinc_history_s:
                        self.recent.append((now, c["node"], {k: c.get(k) for k in COINC_KEYS}))
            elif ev["created"] >= now - self.cfg.coinc_history_s:
                for node, b in ev["bands"].items():
                    if b.get("candidate") and b["candidate"].get("cls") == "candidate":
                        self.recent.append((now, node, {k: b["candidate"].get(k)
                                                        for k in COINC_KEYS}))
        if self.events:
            print(f"combiner: reopened {len(self.events)} events", flush=True)

    def _log_injection(self, rec):
        try:
            with open(os.path.join(self.cfg.out_dir, "injections.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------ talking to monitors
    def _get(self, node, path):
        with urllib.request.urlopen(node["url"] + path, timeout=self.cfg.timeout_s) as r:
            return json.loads(r.read())

    def _post(self, node, path, body):
        req = urllib.request.Request(node["url"] + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        if self.cfg.token:
            req.add_header("X-Skymon-Token", self.cfg.token)
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as r:
            return json.loads(r.read())

    # The mailbox: each call uses the node's database when it has one, else HTTP
    def _status(self, node):
        if node["db"] is not None:
            rows = node["db"].execute(
                "SELECT value FROM status WHERE key = 'monitor' "
                "AND updated > now() - interval '60 seconds'", fetch=True)
            if not rows:
                raise ConnectionError("no recent status in the node's database")
            return rows[0][0]
        return self._get(node, "/api/status")

    def _candidates(self, node, since):
        if node["db"] is not None:
            if node["cursor"] is None:
                rows = node["db"].execute(
                    "SELECT coalesce(min(id) - 1, (SELECT coalesce(max(id), 0) FROM candidates)) "
                    "FROM candidates WHERE created > to_timestamp(%s)", (since,), fetch=True)
                if rows is None:
                    raise ConnectionError("database")
                node["cursor"] = int(rows[0][0] or 0)
            rows = node["db"].execute(
                "SELECT id, record FROM candidates WHERE id > %s AND cls <> 'followup' "
                "AND cls <> 'context' ORDER BY id LIMIT 20000", (node["cursor"],), fetch=True)
            if rows is None:
                raise ConnectionError("database")
            if rows:
                node["cursor"] = int(rows[-1][0])
            return [r[1] for r in rows]
        return self._get(node, "/api/candidates?" + urllib.parse.urlencode(
            dict(since=since, limit=5000)))

    def _send_watch(self, node, body):
        if node["db"] is not None:
            if not node["db"].execute(
                    "INSERT INTO watches (id, event_id, request) VALUES (%s, %s, %s) "
                    "ON CONFLICT (id) DO NOTHING",
                    (body["id"], body.get("event_id"), json.dumps(_finite(body)))):
                raise ConnectionError("database")
            return body
        return self._post(node, "/api/watch", body)

    def _watches(self, node, event_id):
        if node["db"] is not None:
            rows = node["db"].execute(
                "SELECT id, state, result, reason, context_id FROM watches WHERE event_id = %s",
                (event_id,), fetch=True)
            if rows is None:
                raise ConnectionError("database")
            return [dict(id=r[0], state=r[1], result=r[2], reason=r[3], context_id=r[4])
                    for r in rows]
        return self._get(node, "/api/watches?" + urllib.parse.urlencode(dict(event=event_id)))

    def _send_injection(self, node, spec):
        if node["db"] is not None:
            if not node["db"].execute(
                    "INSERT INTO injections (id, spec) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (spec["id"], json.dumps(_finite(spec)))):
                raise ConnectionError("database")
            return dict(accepted=True)
        return self._post(node, "/api/inject", spec)

    def _refresh_nodes(self):
        futs = {n: self.pool.submit(self._status, node) for n, node in self.nodes.items()}
        for n, f in futs.items():
            node = self.nodes[n]
            try:
                st = f.result(timeout=self.cfg.timeout_s + 2)
                ok = st.get("band") is not None
                node.update(up=ok, status=st if ok else node["status"],
                            last_ok=time.time() if ok else node["last_ok"],
                            error=None if ok else "not imaging")
            except Exception as e:
                node.update(up=False, error=type(e).__name__)

    def _band(self, node):
        st = node.get("status") or {}
        b = st.get("band")
        if not b:
            return None
        tiers = st.get("tiers") or []
        return dict(f_lo=b["f_lo"], f_hi=b["f_hi"], el_min=st.get("el_min", 10.0),
                    beam=st.get("beam_deg") or beam_fwhm_deg(0.5 * (b["f_lo"] + b["f_hi"])),
                    dm_max=tiers[-1]["dm_max"] if tiers else 3000.0,
                    latency=max([t["latency_s"] for t in tiers] or [60.0]),
                    threshold=st.get("threshold", 8.0))

    # ------------------------------------------------------------ events
    def _new_event(self, node, c):
        eid = f"ev-{time.strftime('%Y%m%dT%H%M%S', time.gmtime(c['t_inf']))}-" \
              f"{uuid.uuid4().hex[:6]}"
        inj = c.get("injection_id") if c["cls"] == "injection" else None
        ev = dict(id=eid, created=time.time(), kind="injection" if inj else "sky",
                  injection_id=inj, status="pending", ra=c["ra"], dec=c["dec"], dm=c["dm"],
                  dm_err=c["dm_step"], t_inf=c["t_inf"], utc_inf=utc(c["t_inf"]),
                  ref_f_lo=c["f_lo_mhz"], ref_beam=beam_fwhm_deg(c["f_lo_mhz"]), refits=0,
                  combined_snr=None, n_seen=0, bands={}, notes=[])
        self.events[eid] = ev
        print(f"combiner: event {eid} from {node} S/N {c['snr']:.1f} DM {c['dm']:.1f} "
              f"RA {c['ra']:.2f} Dec {c['dec']:.2f}", flush=True)
        return ev

    def _matches(self, ev, c):
        if ev["injection_id"] or (c["cls"] == "injection"):
            return bool(ev["injection_id"]) and ev["injection_id"] == c.get("injection_id")
        f = c["f_lo_mhz"]
        radius = elevation_radius(2.0 * max(ev["ref_beam"], beam_fwhm_deg(f)), c.get("el", 90.0))
        if separation_deg(ev["ra"], ev["dec"], c["ra"], c["dec"]) > radius:
            return False
        tb = ev["t_inf"] + K_DM * ev["dm"] / f ** 2
        lever = K_DM * abs(f ** -2 - ev["ref_f_lo"] ** -2)
        tol = 3 * c["tier_dt"] + 2 * c["width_s"] + 2.0 + lever * max(ev["dm_err"], c["dm_step"])
        dm_tol = max(0.1 * ev["dm"], 3 * max(ev["dm_err"], c["dm_step"]))
        if abs(c["dm"] - ev["dm"]) <= dm_tol and abs(c["t_bottom"] - tb) <= tol:
            return "dm"
        # at the same position during the burst's sweep through this band, but
        # at another DM: either the burst triggering a neighbouring trial DM,
        # or a second burst. Which one is not decided here (_add_other_dm).
        tt = ev["t_inf"] + K_DM * ev["dm"] / c["f_hi_mhz"] ** 2
        return "sweep" if tt - tol <= c["t_bottom"] <= tb + tol else False

    def _add_other_dm(self, ev, node, c):
        """
        A candidate at the event's position, within its sweep, at another DM.
        It is kept with the event and listed for the reviewer as another DM at
        this position (a second burst, such as from a repeating source, looks
        the same as a sidelobe of a bright one at a wrong trial DM). Its data
        stay linked to the event, so they are archived with it; it does not
        replace the band's detection or change the fitted DM.
        """
        b = ev["bands"].setdefault(node, {})
        b.setdefault("others", []).append(c["id"])
        ev.setdefault("other_dm", []).append(dict(
            node=node, id=c["id"], dm=c["dm"], snr=c["snr"], t_bottom=c["t_bottom"],
            ra=c["ra"], dec=c["dec"], dm_step=c["dm_step"], tier_dt=c["tier_dt"],
            width_s=c["width_s"], f_lo_mhz=c["f_lo_mhz"]))
        note = (f"{utc(time.time())} {node} also found S/N {c['snr']:.1f} at DM {c['dm']:.1f} "
                f"at this position during the sweep (a wrong trial DM or a second burst)")
        ev["notes"].append(note)

    def _add_detection(self, ev, node, c):
        b = ev["bands"].setdefault(node, {})
        cand = {k: c.get(k) for k in ("id", "cls", "snr", "dm", "dm_step", "t_inf", "t_bottom",
                                      "ra", "dec", "el", "tier", "tier_dt", "width_s",
                                      "f_lo_mhz", "f_hi_mhz", "injection_id")}
        # the band's data quality in the minute before the detection (conditions.py)
        cand["conditions"] = (c.get("quality") or {}).get("conditions")
        # fluence from the band's calibration (flux.py), when the monitor has one
        fl = c.get("flux") or {}
        cand["fluence_jy_ms"] = fl.get("fluence_jy_ms")
        cand["fluence_err_jy_ms"] = fl.get("fluence_err_jy_ms")
        if b.get("candidate") and b["candidate"]["snr"] >= c["snr"]:
            b.setdefault("others", []).append(c["id"])
        else:
            if b.get("candidate"):
                b.setdefault("others", []).append(b["candidate"]["id"])
            b["candidate"] = cand
        b["state"] = "detected"
        self._refit(ev)
        if ev["status"] != "pending":  # a late detection reopens a decided event
            ev["notes"].append(f"{utc(time.time())} reopened by a detection in {node}")
            ev["status"] = "pending"
            self._attach_context(ev)

    def _attach_context(self, ev):
        """
        Solar reports, lightning and the Sun's elevation for the reviewers.
        Returns True if the context changed. Never fails the combiner.
        """
        try:
            gen = self.context.generation
            ctx = self.context.for_event(ev)
        except Exception as e:
            print(f"combiner: context of {ev['id']} failed: {e!r}", flush=True)
            return False
        self._context_gen[ev["id"]] = gen
        if ctx == ev.get("context"):
            return False
        ev["context"] = ctx
        return True

    def _refit(self, ev):
        """Position from the highest band, DM and t_inf from all detections"""
        dets = [(b["candidate"]["f_lo_mhz"], b["candidate"]["t_bottom"],
                 max(b["candidate"]["tier_dt"], b["candidate"]["width_s"]), b["candidate"])
                for b in ev["bands"].values() if b.get("state") == "detected"]
        if not dets:
            return
        best = max(dets, key=lambda d: d[0])[3]  # smallest beam
        ev.update(ra=best["ra"], dec=best["dec"], ref_f_lo=best["f_lo_mhz"],
                  ref_beam=beam_fwhm_deg(best["f_lo_mhz"]))
        fit = fit_dm([d[:3] for d in dets])
        if fit:
            dm, t_inf, err = fit
            ev.update(dm=dm, t_inf=t_inf, utc_inf=utc(t_inf),
                      dm_err=max(err, 0.25 * min(d[3]["dm_step"] for d in dets)))
            ev["refits"] += 1
        else:
            ev.update(dm=best["dm"], t_inf=best["t_inf"], utc_inf=utc(best["t_inf"]),
                      dm_err=best["dm_step"])

    def _plan(self, ev, now):
        """Predict, check visibility and send watches for the bands still unresolved"""
        for name, node in self.nodes.items():
            b = ev["bands"].setdefault(name, {})
            if b.get("state") in ("detected", "followup", "not_visible", "searched"):
                continue
            band = self._band(node)
            if band is None:
                b.setdefault("state", "node_down")
                continue
            b["threshold"] = band["threshold"]
            f_lo, f_hi = band["f_lo"], band["f_hi"]
            tb = ev["t_inf"] + K_DM * ev["dm"] / f_lo ** 2
            tt = ev["t_inf"] + K_DM * ev["dm"] / f_hi ** 2
            lever = K_DM * abs(f_lo ** -2 - ev["ref_f_lo"] ** -2)
            t_tol = lever * ev["dm_err"] + 2.0
            el_top = float(altaz(ev["ra"], ev["dec"], tt)[0])
            el_bot = float(altaz(ev["ra"], ev["dec"], tb)[0])
            b.update(f_lo=f_lo, f_hi=f_hi, predicted_t_top=tt, predicted_t_bottom=tb,
                     t_tol=t_tol, el_at_arrival=el_bot,
                     deadline=tb + t_tol + band["latency"] + self.cfg.settle_s)
            if min(el_top, el_bot) < band["el_min"]:
                b["state"] = "not_visible"
                continue
            if ev["dm"] > band["dm_max"]:
                b.update(state="searched", reason="DM beyond this band's search range")
                continue
            if not node["up"]:
                b["state"] = "node_down"
                continue
            if now > tb + t_tol + band["latency"] and b.get("state") != "watching":
                # the burst passed this band before the event was known: that
                # monitor's own search at its threshold covered it, and the data
                # for a measurement are gone. The watch still makes the monitor
                # keep the position's accumulated spectrum around the arrival.
                if not b.get("watch_id"):
                    wid = f"{ev['id']}-{name}-ctx"
                    try:
                        self._send_watch(node, dict(
                            id=wid, event_id=ev["id"], ra=ev["ra"], dec=ev["dec"], dm=ev["dm"],
                            dm_err=ev["dm_err"], t_inf=ev["t_inf"], t_tol=t_tol))
                        b["watch_id"] = wid
                    except Exception:
                        pass
                b.update(state="searched", reason="passed this band before the event was "
                                                  "known; covered by its own search")
                continue
            key = (round(ev["dm"], 2), round(ev["t_inf"], 2))
            if b.get("watch_key") == key and b.get("state") == "watching":
                continue
            wid = f"{ev['id']}-{name}-{ev['refits']}"
            try:
                self._send_watch(node, dict(id=wid, event_id=ev["id"], ra=ev["ra"],
                                                    dec=ev["dec"], dm=ev["dm"],
                                                    dm_err=ev["dm_err"], t_inf=ev["t_inf"],
                                                    t_tol=t_tol))
                b.update(state="watching", watch_id=wid, watch_key=key, watch_sent=now,
                         ahead_s=tb - now)
            except Exception as e:
                b.update(state="node_down", error=type(e).__name__)

    def _collect_watches(self, ev):
        for name, b in ev["bands"].items():
            if b.get("watch_id") and not b.get("data_id") and b.get("state") != "watching":
                self._collect_context(ev, name, b)
            if b.get("state") != "watching":
                continue
            node = self.nodes[name]
            if not node["up"]:
                continue
            try:
                ws = self._watches(node, ev["id"])
            except Exception:
                continue
            w = next((x for x in ws if x["id"] == b.get("watch_id")), None)
            if w is None:
                b["state"] = "node_down"  # the monitor restarted and lost the watch
                b["reason"] = "watch lost"
            elif w["state"] == "measured":
                r = w["result"]
                b.update(state="followup", followup=r, data_id=r.get("cand_id"))
            elif w["state"] == "expired":
                b.update(state="searched", reason=w.get("reason"))

    def _collect_context(self, ev, name, b):
        """Note the record a monitor kept for a watch it could not measure"""
        node = self.nodes[name]
        if not node["up"]:
            return
        try:
            ws = self._watches(node, ev["id"])
        except Exception:
            return
        w = next((x for x in ws if x["id"] == b.get("watch_id")), None)
        if w and w.get("context_id"):
            b["data_id"] = w["context_id"]
        elif w and w["state"] == "measured":
            b.update(state="followup", followup=w["result"], data_id=w["result"].get("cand_id"))

    def _decide(self, ev, now):
        bands = ev["bands"]
        open_ = [b for b in bands.values()
                 if b.get("state") in (None, "watching", "node_down")
                 and now < b.get("deadline", 0)]
        if open_:
            return False
        seen, measured = [], []
        for name, b in bands.items():
            if b.get("state") == "detected":
                seen.append(name)
                measured.append(b["candidate"]["snr"])
            elif b.get("state") == "followup":
                measured.append(b["followup"]["snr"])
                if b["followup"]["snr"] >= self.cfg.confirm_snr:
                    seen.append(name)
            elif b.get("state") == "watching":
                b.update(state="searched", reason="no result before the deadline")
        could = [n for n, b in bands.items() if b.get("state") in ("detected", "followup",
                                                                    "searched")]
        spectrum = characterise(band_values(ev))
        # data quality of the bands that measured the burst: a detection made
        # while a band was disturbed (storm, solar activity, interference)
        # calls for a closer look at that band's data
        cond = {}
        for name, b in bands.items():
            if b.get("state") == "detected":
                c = b["candidate"].get("conditions")
            elif b.get("state") == "followup":
                c = b["followup"].get("conditions")
            else:
                continue
            if c is not None:
                cond[name] = c
        if len(seen) >= self.cfg.min_bands:
            status = "confirmed"
        elif len(seen) >= 1 and len(could) >= 2:
            status = "band_limited" if spectrum.get("plausible") else "implausible"
        else:
            status = "unconfirmable"
        ev.update(status=status, n_seen=len(seen), seen=seen, could_see=could,
                  combined_snr=float(np.sum(measured) / math.sqrt(len(measured)))
                  if measured else None,
                  best_snr=float(max(measured)) if measured else None,
                  spectrum=spectrum, decided=now, conditions=cond,
                  disturbed=any(bool(c.get("disturbed")) for c in cond.values()),
                  review=ev.get("review") or ("pending" if ev["kind"] == "sky" else "not needed"),
                  review_priority=dict(confirmed=1, band_limited=2, unconfirmable=3,
                                       implausible=4)[status])
        self._attach_context(ev)
        print(f"combiner: {ev['id']} {status}: seen in {seen or '-'}, could see "
              f"{could or '-'}, DM {ev['dm']:.1f} +- {ev['dm_err']:.2f}", flush=True)
        if ev["injection_id"] and ev["injection_id"] in self.injections:
            inj = self.injections[ev["injection_id"]]
            rec = dict(event="decided", id=inj["id"], status=status, seen=seen, could_see=could,
                       dm=inj["dm"], dm_fit=ev["dm"], snr=inj["snr"])
            inj["result"] = rec
            self._log_injection(rec)
        return True

    # ------------------------------------------------------------ coincidence veto
    @staticmethod
    def _tolerance(a, b):
        return sum(float(x.get(k) or 0.0) for x in (a, b) for k in ("tier_dt", "width_s"))

    def _undispersed(self, a, b):
        """
        Simultaneous in two bands, although the dispersion law at either DM
        would separate them: the predicted gap is that DM times the lever
        K (f_lo_a^-2 - f_lo_b^-2), and a burst of the sky must show it
        """
        tol = self._tolerance(a, b)
        if abs(a["t_bottom"] - b["t_bottom"]) > self.cfg.coinc_slack_s + tol:
            return False
        lever = K_DM * abs(a["f_lo_mhz"] ** -2 - b["f_lo_mhz"] ** -2)
        return lever * max(a["dm"], b["dm"]) > self.cfg.coinc_disp_margin_s + tol

    def _coincidences(self, now):
        """
        Groups of candidates linked by undispersed pairs, over the history.
        The candidates are sorted by arrival, so each is compared only with
        those that follow it within the largest possible tolerance.
        """
        self.recent = [r for r in self.recent if r[0] >= now - self.cfg.coinc_history_s]
        items = sorted(self.recent, key=lambda r: r[2]["t_bottom"])
        if len(items) < 2:
            return []
        reach = self.cfg.coinc_slack_s + 2 * max(
            float(c.get("tier_dt") or 0.0) + float(c.get("width_s") or 0.0) for _, _, c in items)
        parent = {}

        def find(k):
            parent.setdefault(k, k)
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        for i, (_, na, a) in enumerate(items):
            for _, nb, b in items[i + 1:]:
                if b["t_bottom"] - a["t_bottom"] > reach:
                    break
                if na != nb and self._undispersed(a, b):
                    parent[find((na, a["id"]))] = find((nb, b["id"]))
        groups = {}
        recs = {(n, c["id"]): c for _, n, c in items}
        for k in parent:
            groups.setdefault(find(k), []).append((k[0], recs[k]))
        return list(groups.values())

    def _reason(self, members):
        """The pair that the dispersion law separates most, in words"""
        best = None
        for i, (na, a) in enumerate(members):
            for nb, b in members[i + 1:]:
                if na == nb or not self._undispersed(a, b):
                    continue
                dm = max(a["dm"], b["dm"])
                gap = K_DM * dm * abs(a["f_lo_mhz"] ** -2 - b["f_lo_mhz"] ** -2)
                if best is None or gap > best[0]:
                    best = (gap, dm, a, b, na, nb)
        gap, dm, a, b, na, nb = best
        return (f"undispersed: {na} ({a['f_lo_mhz']:.1f} MHz) and {nb} ({b['f_lo_mhz']:.1f} MHz) "
                f"were triggered {abs(a['t_bottom'] - b['t_bottom']):.2f} s apart, whereas a "
                f"burst at DM {dm:.1f} reaches them {gap:.1f} s apart")

    def _veto(self, now, changed):
        """Record undispersed coincidences as terrestrial events, and move out sky events"""
        for members in self._coincidences(now):
            keys = [(n, c["id"]) for n, c in members]
            new = [(n, c) for n, c in members if (n, c["id"]) not in self.terrestrial]
            if not new:
                continue
            eids = sorted({self.terrestrial[k] for k in keys if k in self.terrestrial})
            ev = self.events.get(eids[0]) if eids else None
            if ev is None:
                ev = self._new_terrestrial(members, now)
            for n, c in new:
                self.terrestrial[(n, c["id"])] = ev["id"]
                ev["candidates"].append(dict(node=n, utc_bottom=utc(c["t_bottom"]), **c))
                b = ev["bands"].setdefault(n, dict(state="terrestrial"))
                if b.get("candidate") is None or b["candidate"]["snr"] < c["snr"]:
                    if b.get("candidate"):
                        b.setdefault("others", []).append(b["candidate"]["id"])
                    b["candidate"] = dict(c)
                else:
                    b.setdefault("others", []).append(c["id"])
            cands = ev["candidates"]
            best = max(cands, key=lambda c: c["snr"])
            ev.update(nodes=sorted({c["node"] for c in cands}),
                      t_inf=min(ev["t_inf"], min(c["t_bottom"] for c in cands)),
                      ra=best["ra"], dec=best["dec"], dm=best["dm"], dm_err=best["dm_step"],
                      best_snr=float(best["snr"]),
                      reason=self._reason([(c["node"], c) for c in cands]))
            ev["utc_inf"] = utc(ev["t_inf"])
            ev["n_bands"] = len(ev["nodes"])
            changed.add(ev["id"])
            self._move_out(ev, now, changed)
            print(f"combiner: terrestrial {ev['id']} in {ev['nodes']}: {ev['reason']}", flush=True)

    def _new_terrestrial(self, members, now):
        t0 = min(c["t_bottom"] for _, c in members)
        eid = f"tr-{time.strftime('%Y%m%dT%H%M%S', time.gmtime(t0))}-{uuid.uuid4().hex[:6]}"
        ev = dict(id=eid, created=time.time(), kind="terrestrial", injection_id=None,
                  status="terrestrial", review="not needed", t_inf=t0, utc_inf=utc(t0),
                  ra=None, dec=None, dm=None, dm_err=None, ref_f_lo=None, ref_beam=None,
                  refits=0, combined_snr=None, best_snr=None, n_seen=0, bands={},
                  candidates=[], sky_events=[], notes=[], decided=now)
        self.events[eid] = ev
        return ev

    def _move_out(self, tev, now, changed):
        """
        Give status terrestrial to every pending sky event whose detections are
        all part of terrestrial coincidences. An event with a detection outside
        them (a burst that happened to coincide with interference in another
        band) stays open, with a note.
        """
        mine = {(c["node"], c["id"]) for c in tev["candidates"]}
        for ev in self.events.values():
            if ev["kind"] != "sky" or ev["status"] != "pending":
                continue
            ids = {(n, b["candidate"]["id"]) for n, b in ev["bands"].items()
                   if b.get("state") == "detected" and b.get("candidate")}
            ids |= {(n, cid) for n, b in ev["bands"].items() for cid in b.get("others", [])}
            if not ids & mine:
                continue
            dets = {(n, b["candidate"]["id"]) for n, b in ev["bands"].items()
                    if b.get("state") == "detected" and b.get("candidate")}
            if dets and all(k in self.terrestrial for k in dets):
                ev["notes"].append(f"{utc(now)} moved to terrestrial event {tev['id']}: "
                                   f"{tev['reason']}")
                ev.update(status="terrestrial", review="not needed", terrestrial_event=tev["id"],
                          decided=now)
            elif tev["id"] in ev.setdefault("terrestrial_links", []):
                continue
            else:
                ev["terrestrial_links"].append(tev["id"])
                ev["notes"].append(f"{utc(now)} a candidate is part of terrestrial event "
                                   f"{tev['id']}; other detections keep the event open")
            if ev["id"] not in tev["sky_events"]:
                tev["sky_events"].append(ev["id"])
            changed.add(ev["id"])

    # ------------------------------------------------------------ main loop
    def poll(self, now=None):
        now = time.time() if now is None else now
        self._refresh_nodes()
        since = now - self.cfg.lookback_s
        new = []
        for name, node in self.nodes.items():
            if not node["up"]:
                continue
            try:
                cands = self._candidates(node, since)
            except Exception as e:
                node.update(up=False, error=type(e).__name__)
                continue
            for c in cands:
                if (name, c["id"]) in self.seen:
                    continue
                self.seen.add((name, c["id"]))
                if c["cls"] in ("candidate", "rfi"):
                    self.recent.append((now, name, {k: c.get(k) for k in COINC_KEYS}))
                # only plain candidates open events: not RFI, not candidates at
                # known sources ("known"), and of injections only coordinated ones
                if c["cls"] == "candidate" or (c["cls"] == "injection" and
                                               str(c.get("injection_id", "")).startswith("cinj-")):
                    new.append((name, c))
        with self.lock:
            changed = set()
            self._veto(now, changed)
            for name, c in sorted(new, key=lambda x: -x[1]["snr"]):
                if (name, c["id"]) in self.terrestrial:
                    continue  # recorded with its terrestrial event
                ev, how = None, None
                for e in self.events.values():
                    if e["kind"] == "terrestrial" or e["status"] == "terrestrial":
                        continue
                    if not (e["status"] == "pending" or now - e.get("decided", now) < 3600):
                        continue
                    m = self._matches(e, c)
                    if m == "dm" or m is True:
                        ev, how = e, "dm"
                        break
                    if m == "sweep" and ev is None:
                        ev, how = e, "sweep"
                if ev is None:
                    ev = self._new_event(name, c)
                    self._add_detection(ev, name, c)
                elif how == "sweep":
                    self._add_other_dm(ev, name, c)
                else:
                    self._add_detection(ev, name, c)
                changed.add(ev["id"])
            for ev in list(self.events.values()):
                if ev["status"] != "pending":
                    # the external reports often come in after the decision
                    if now - ev.get("decided", now) < self.cfg.context_refresh_s and \
                            self._context_gen.get(ev["id"]) != self.context.generation and \
                            self._attach_context(ev):
                        changed.add(ev["id"])
                    # the records a monitor keeps for passed arrivals come later
                    if now - ev.get("decided", now) < 7200 and any(
                            b.get("watch_id") and not b.get("data_id")
                            for b in ev["bands"].values()):
                        for name, b in ev["bands"].items():
                            if b.get("watch_id") and not b.get("data_id"):
                                self._collect_context(ev, name, b)
                        changed.add(ev["id"])
                    continue
                self._collect_watches(ev)
                self._plan(ev, now)
                self._decide(ev, now)
                changed.add(ev["id"])
            for eid in changed:
                self._save(self.events[eid])
            cutoff = now - self.cfg.keep_hours * 3600
            for eid in [k for k, e in self.events.items() if e["created"] < cutoff]:
                del self.events[eid]
                self._context_gen.pop(eid, None)
            self.terrestrial = {k: v for k, v in self.terrestrial.items() if v in self.events}
        self._maybe_inject(now)

    def _maybe_inject(self, now):
        """A coordinated injection: the same burst requested from every monitor that is up"""
        if self.cfg.inject_interval_s <= 0:
            return
        if self.next_inject is None:
            self.next_inject = now + self.rng.exponential(self.cfg.inject_interval_s)
        if now < self.next_inject:
            return
        self.next_inject = now + self.rng.exponential(self.cfg.inject_interval_s)
        bands = [(n, self._band(nd)) for n, nd in self.nodes.items() if nd["up"]]
        bands = [(n, b) for n, b in bands if b]
        if len(bands) < 2:
            return
        f_low = min(b["f_lo"] for _, b in bands)
        f_high = max(b["f_hi"] for _, b in bands)
        el_min = max(max(b["el_min"] for _, b in bands), self.cfg.inject_el_min)
        dm = float(math.exp(self.rng.uniform(*np.log(self.cfg.inject_dm))))
        dm = min(dm, min(b["dm_max"] for _, b in bands) * 0.95)
        span = K_DM * dm * (f_low ** -2 - f_high ** -2) + 120
        for _ in range(200):
            ra = float(self.rng.uniform(0, 360))
            dec = float(np.degrees(np.arcsin(self.rng.uniform(-0.6, 1.0))))
            if altaz(ra, dec, now)[0] >= el_min and \
                    float(time_to_set(dec, ra, now, el_min)) > span + 300:
                break
        else:
            return
        t_inf = now + 90.0 - K_DM * dm / f_high ** 2
        snr = float(self.rng.uniform(*self.cfg.inject_snr))
        iid = "cinj-" + uuid.uuid4().hex[:10]
        spec = dict(id=iid, ra=ra, dec=dec, dm=dm, t_inf=t_inf, snr=snr, width_s=0.005)
        sent = []
        for n, _ in bands:
            try:
                self._send_injection(self.nodes[n], spec)
                sent.append(n)
            except Exception:
                pass
        rec = dict(event="requested", nodes=sent, **spec)
        self.injections[iid] = rec
        self._log_injection(rec)
        print(f"combiner: coordinated injection {iid} DM {dm:.1f} RA {ra:.1f} Dec {dec:.1f} "
              f"to {sent}", flush=True)

    def review(self, eid, verdict, by="", note=""):
        """Record a person's verdict on an event; the latest one is the event's verdict"""
        if verdict not in ("frb", "rfi", "unclear"):
            raise ValueError("verdict must be frb, rfi or unclear")
        with self.lock:
            ev = self.events.get(eid)
            if ev is None:
                raise KeyError(eid)
            ev.setdefault("reviews", []).append(dict(verdict=verdict, by=str(by)[:80],
                                                     note=str(note)[:2000], at=time.time()))
            ev["review"] = verdict
            self._save(ev)
            return ev

    def run(self, stop: threading.Event):
        self.context.start(stop)  # background threads; they never block the loop
        while not stop.is_set():
            t0 = time.time()
            try:
                self.poll()
            except Exception as e:  # never stop on one bad poll
                print(f"combiner: poll failed: {e!r}", flush=True)
            stop.wait(max(0.5, self.cfg.poll_s - (time.time() - t0)))

    # ------------------------------------------------------------ queries
    def status(self):
        with self.lock:
            counts = {}
            for e in self.events.values():
                counts[e["status"]] = counts.get(e["status"], 0) + 1
            counts.setdefault("terrestrial", 0)
            injs = [i.get("result", {}).get("status") for i in self.injections.values()]
            to_review = sum(1 for e in self.events.values()
                            if e.get("review") == "pending" and e["status"] != "pending")
            return dict(started=self.started, now=time.time(), events=counts, to_review=to_review,
                        terrestrial=dict(events=sum(1 for e in self.events.values()
                                                    if e["kind"] == "terrestrial"),
                                         candidates=len(self.terrestrial)),
                        context=self.context.status(),
                        nodes=[dict(name=n["name"], up=n["up"], error=n["error"],
                                    last_ok=n["last_ok"], band=(n["status"] or {}).get("band"))
                               for n in self.nodes.values()],
                        coordinated_injections=dict(requested=len(injs),
                                                    confirmed=injs.count("confirmed"),
                                                    band_limited=injs.count("band_limited"),
                                                    implausible=injs.count("implausible"),
                                                    unconfirmable=injs.count("unconfirmable"),
                                                    pending=injs.count(None)))

    def list(self, since=0.0, status=None, kind=None, limit=500, review=None):
        with self.lock:
            ev = [e for e in self.events.values() if e["t_inf"] >= since
                  and (status is None or e["status"] == status)
                  and (kind is None or e["kind"] == kind)
                  and (review is None or e.get("review") == review)]
        ev.sort(key=lambda e: e["t_inf"], reverse=True)
        return ev[:limit]


def serve(comb: Combiner, host="0.0.0.0", port=8096):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}
            parts = [p for p in u.path.split("/") if p]
            try:
                if parts == ["api", "status"]:
                    return self._json(comb.status())
                if parts == ["api", "events"]:
                    return self._json(comb.list(float(q.get("since", 0)), q.get("status"),
                                                q.get("kind"), int(q.get("limit", 500)),
                                                q.get("review")))
                if parts[:2] == ["api", "events"] and len(parts) == 3:
                    e = comb.events.get(parts[2])
                    return self._json(e) if e else self._json(dict(error="unknown event"), 404)
                return self._json(dict(error="not found"), 404)
            except (ValueError, KeyError) as e:
                return self._json(dict(error=str(e)), 400)

        def do_POST(self):
            import hmac
            token = comb.cfg.token
            if token and not hmac.compare_digest(self.headers.get("X-Skymon-Token", ""), token):
                return self._json(dict(error="forbidden"), 403)
            parts = [p for p in urllib.parse.urlparse(self.path).path.split("/") if p]
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
                if parts[:2] == ["api", "events"] and len(parts) == 4 and parts[3] == "review":
                    ev = comb.review(parts[2], body.get("verdict"), body.get("by", ""),
                                     body.get("note", ""))
                    return self._json(dict(id=ev["id"], review=ev["review"],
                                           reviews=ev["reviews"]))
                return self._json(dict(error="not found"), 404)
            except KeyError:
                return self._json(dict(error="unknown event"), 404)
            except (ValueError, TypeError) as e:
                return self._json(dict(error=str(e)), 400)

    httpd = ThreadingHTTPServer((host, port), H)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _finite(o):
    from .localdb import _finite as f
    return f(o)
