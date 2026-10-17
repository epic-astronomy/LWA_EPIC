"""
External context attached to the combiner's events, for the reviewers.

Solar radio bursts (types II and III drift from high to low frequency) and
nearby thunderstorms produce false events. Reports from outside the array
help a reviewer recognise them. For every decided event the combiner attaches

    context = dict(sun_el_deg, sun_sep_deg, solar=[...], lightning={...},
                   sources=[...], unavailable=[...])

  sun_el_deg   elevation of the Sun at LWA-SV at the event's first arrival
  sun_sep_deg  angle between the Sun and the event's position
  solar        NOAA SWPC reports within +-15 min of the event's arrival in
               any of its bands: radio sweeps (RSP: types II, III, IV, ...)
               whose frequency range overlaps the band, and X-ray flares
               (XRA, and the GOES real-time flare list) at any frequency.
               None when SWPC is disabled or has never answered.
  lightning    strikes within +-10 min of the arrivals: the number within
               100 km of LWA-SV, the distance of the nearest one, and whether
               the data covered the whole window. None when no lightning
               source is configured or it has never answered.
  sources      providers with data (fetched now or earlier)
  unavailable  providers that are enabled but have no data

Each provider is polled in its own background thread (every 15 min by
default) with a timeout. The last good data are kept in memory and in
<out_dir>/context/<provider>.json, so a restart without network still has
them. A failure is logged once and never reaches the combiner: without the
network the combiner runs exactly as before, and the context only lists the
provider as unavailable.

Lightning data format (--lightning-url, an http(s) URL or a local file):

    {"strikes": [{"time": "2026-10-17T03:12:45Z", "lat": 34.1, "lon": -106.5}, ...]}

or the list alone. "time" is Unix seconds or ISO 8601 (UTC when no zone is
given); "latitude"/"longitude" and "t" are accepted for "lat"/"lon" and
"time". Strikes within 500 km of LWA-SV are kept for 48 h across fetches, so
a source that serves only the last minutes is enough.

NOAA SWPC data are in the public domain.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.request
import warnings
from datetime import datetime, timezone

from .dedisp import K_DM
from .geometry import LWASV_LAT, LWASV_LON

LWASV_HEIGHT_M = 1477.0
EARTH_RADIUS_KM = 6371.0

SWPC_URLS = dict(
    edited_events="https://services.swpc.noaa.gov/json/edited_events.json",
    goes_flares_7day="https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json",
    goes_flares_latest="https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json",
)
# The solar radio spectrographs reporting to SWPC stop at about 25 MHz; a
# sweep reported down to that limit continues below it.
SPECTROGRAPH_FLOOR_MHZ = 26.0


# ------------------------------------------------------------------ helpers
def parse_time(v):
    """Unix seconds from Unix seconds or ISO 8601 (UTC when no zone is given); None if invalid"""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    s = str(v).strip()
    if not s or s.lower().startswith("unk"):
        return None
    try:
        return float(s)
    except ValueError:
        pass
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def iso(unix):
    if unix is None:
        return None
    return datetime.fromtimestamp(unix, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (haversine)"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def arrivals(ev):
    """
    The event's arrival in each band with a known frequency range:
    [(band, f_lo, f_hi, t_top, t_bottom)], from the event's DM and t_inf
    """
    out = []
    for name, b in (ev.get("bands") or {}).items():
        c = b.get("candidate") or {}
        f_lo = b.get("f_lo") or c.get("f_lo_mhz")
        f_hi = b.get("f_hi") or c.get("f_hi_mhz")
        if not f_lo or not f_hi:
            continue
        out.append((name, float(f_lo), float(f_hi),
                    ev["t_inf"] + K_DM * ev["dm"] / f_hi ** 2,
                    ev["t_inf"] + K_DM * ev["dm"] / f_lo ** 2))
    return out


def first_arrival(ev):
    """Time of the earliest detection, else of the earliest predicted arrival, else t_inf"""
    dets = [b["candidate"]["t_bottom"] for b in (ev.get("bands") or {}).values()
            if b.get("state") == "detected" and b.get("candidate")]
    if dets:
        return min(dets)
    arr = arrivals(ev)
    return min(a[3] for a in arr) if arr else ev["t_inf"]


def sun_position(unix, ra=None, dec=None):
    """
    (elevation of the Sun at LWA-SV, separation from (ra, dec) or None), degrees.
    No download is attempted: the Earth orientation tables bundled with
    astropy are accurate to far better than needed here.
    """
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
    from astropy.time import Time
    from astropy.utils import iers
    loc = EarthLocation(lat=LWASV_LAT * u.deg, lon=LWASV_LON * u.deg,
                        height=LWASV_HEIGHT_M * u.m)
    with iers.conf.set_temp("auto_download", False), \
            iers.conf.set_temp("auto_max_age", None), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = Time(unix, format="unix")
        sun = get_sun(t)
        el = float(sun.transform_to(AltAz(obstime=t, location=loc)).alt.deg)
        sep = None
        if ra is not None and dec is not None:
            # in the geocentric frame: the barycentric direction of the Sun
            # differs from the one seen from the Earth
            sep = float(sun.separation(SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")).deg)
    return el, sep


# ------------------------------------------------------------------ SWPC parsing
def parse_edited_events(payload):
    """
    Radio sweeps (RSP) and X-ray flares (XRA) from SWPC edited_events.json.

    Fields used: type, begin_datetime, max_datetime, end_datetime (UTC, no
    zone), frequency ("025-074", MHz, for RSP; "1-8A" for XRA), particulars1
    (sweep type/intensity such as "III/2", or the flare class such as "C1.2"),
    particulars2 (shock speed in km/s for type II), observatory, region.
    Records that do not parse are skipped.
    """
    out = []
    for e in payload if isinstance(payload, list) else []:
        try:
            typ = str(e.get("type") or "").strip()
            if typ not in ("RSP", "XRA"):
                continue
            begin = parse_time(e.get("begin_datetime"))
            if begin is None:
                continue
            end = parse_time(e.get("end_datetime"))
            peak = parse_time(e.get("max_datetime"))
            rec = dict(source="swpc", type=typ, cls=(e.get("particulars1") or "").strip() or None,
                       begin=begin, max=peak, end=end if end is not None and end >= begin
                       else (peak if peak is not None else begin),
                       observatory=(e.get("observatory") or "").strip() or None,
                       region=e.get("region"), f_mhz=None)
            if typ == "RSP":
                lo, _, hi = str(e.get("frequency") or "").partition("-")
                try:
                    rec["f_mhz"] = [float(lo), float(hi)]
                except ValueError:
                    rec["f_mhz"] = None  # an unknown range matches any band
                cls = rec["cls"] or ""
                rec["sweep"] = cls.split("/")[0] or None
                if rec["sweep"] == "II" and e.get("particulars2"):
                    try:
                        rec["speed_km_s"] = float(e["particulars2"])
                    except (TypeError, ValueError):
                        pass
            out.append(rec)
        except (AttributeError, TypeError):
            continue
    return out


def parse_goes_flares(payload):
    """
    X-ray flares from the GOES lists (xray-flares-7-day.json, -latest.json).

    Fields used: begin_time, max_time, end_time (ISO 8601 with Z, or "Unk"
    while the flare is in progress), max_class, satellite.
    """
    out = []
    for e in payload if isinstance(payload, list) else []:
        try:
            begin = parse_time(e.get("begin_time"))
            if begin is None:
                continue
            peak = parse_time(e.get("max_time"))
            end = parse_time(e.get("end_time"))
            out.append(dict(source="goes", type="XRA", cls=e.get("max_class") or None,
                            begin=begin, max=peak,
                            end=end if end is not None and end >= begin else None,
                            observatory=f"G{e['satellite']}" if e.get("satellite") else None,
                            region=None, f_mhz=None, ongoing=end is None))
        except (AttributeError, TypeError):
            continue
    return out


def match_solar(records, ev, window_s=900.0, now=None):
    """
    The solar reports within window_s of the event's arrival in any of its
    bands: a sweep must also overlap the band in frequency. Each match lists
    the bands it overlaps. A flare still in progress lasts until now.
    """
    now = time.time() if now is None else now
    arr = arrivals(ev)
    out, seen = [], set()
    for r in records:
        begin = r["begin"]
        end = r.get("end")
        if end is None:
            end = max(begin, now) if r.get("ongoing") else begin
        bands = []
        for name, f_lo, f_hi, t_top, t_bot in arr:
            if end < t_top - window_s or begin > t_bot + window_s:
                continue
            if r["type"] == "RSP" and r.get("f_mhz"):
                lo, hi = r["f_mhz"]
                if lo <= SPECTROGRAPH_FLOOR_MHZ:
                    lo = 0.0
                if hi < f_lo or lo > f_hi:
                    continue
            bands.append(name)
        if not bands:
            continue
        # the same flare is reported by two GOES satellites and in both lists
        key = (r["type"], round(begin / 60)) if r["type"] == "XRA" else \
            (r["type"], begin, r.get("observatory"), r.get("cls"))
        if key in seen:
            continue
        seen.add(key)
        m = dict(type=r["type"], cls=r.get("cls"), begin=iso(begin), max=iso(r.get("max")),
                 end=iso(r.get("end")), source=r.get("source"),
                 observatory=r.get("observatory"), bands=sorted(bands))
        for k in ("f_mhz", "sweep", "speed_km_s", "region", "ongoing"):
            if r.get(k) is not None:
                m[k] = r[k]
        out.append(m)
    out.sort(key=lambda m: m["begin"])
    return out


# ------------------------------------------------------------------ lightning
def parse_strikes(payload):
    """[(unix, lat, lon)] from a lightning payload (see the module docstring)"""
    if isinstance(payload, dict):
        payload = payload.get("strikes", payload.get("data", []))
    out = []
    for s in payload if isinstance(payload, list) else []:
        try:
            if isinstance(s, (list, tuple)):
                t, lat, lon = parse_time(s[0]), float(s[1]), float(s[2])
            else:
                t = parse_time(s.get("time", s.get("t")))
                lat = float(s.get("lat", s.get("latitude")))
                lon = float(s.get("lon", s.get("longitude")))
        except (AttributeError, TypeError, ValueError, IndexError):
            continue
        if t is None or not (-90 <= lat <= 90 and -180 <= lon <= 360):
            continue
        out.append((t, lat, lon))
    return out


def match_lightning(strikes, ev, covered_until, window_s=600.0, radius_km=100.0):
    """
    Strikes within window_s of the event's arrivals: the number within
    radius_km of LWA-SV, the nearest distance, and whether the data reach the
    end of the window.
    """
    arr = arrivals(ev)
    if arr:
        t0, t1 = min(a[3] for a in arr) - window_s, max(a[4] for a in arr) + window_s
    else:
        t0, t1 = ev["t_inf"] - window_s, ev["t_inf"] + window_s
    d = [distance_km(LWASV_LAT, LWASV_LON, lat, lon) for t, lat, lon in strikes
         if t0 <= t <= t1]
    return dict(strikes_100km=sum(1 for x in d if x <= radius_km),
                nearest_km=round(min(d), 1) if d else None,
                window=[iso(t0), iso(t1)], complete=covered_until is not None
                and covered_until >= t1)


# ------------------------------------------------------------------ providers
class Provider:
    """
    One external source. fetch() returns new data or raises; the service
    keeps the last good data and calls match() for each event.
    """
    name = "provider"

    def __init__(self, interval_s=900.0, timeout_s=20.0):
        self.interval_s = interval_s
        self.timeout_s = timeout_s

    def _read(self, url):
        if "://" not in url or url.startswith("file://"):
            with open(url[7:] if url.startswith("file://") else url) as f:
                return json.load(f)
        req = urllib.request.Request(url, headers={"User-Agent": "skymon-combiner (LWA-SV)"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def fetch(self, previous):
        raise NotImplementedError

    def match(self, data, ev, fetched):
        raise NotImplementedError


class SwpcProvider(Provider):
    """NOAA SWPC: edited events (radio sweeps, X-ray flares) and the GOES flare lists"""
    name = "swpc"

    def __init__(self, urls=None, window_s=900.0, **kw):
        super().__init__(**kw)
        self.urls = dict(SWPC_URLS if urls is None else urls)
        self.window_s = window_s

    def fetch(self, previous):
        """Data per list; a list that fails keeps its last good copy. Fails if none answers."""
        data = dict(previous or {})
        errors = []
        for key, url in self.urls.items():
            try:
                payload = self._read(url)
                data[key] = parse_goes_flares(payload) if key.startswith("goes") \
                    else parse_edited_events(payload)
            except Exception as e:
                errors.append(f"{key}: {type(e).__name__}")
        if len(errors) == len(self.urls):
            raise ConnectionError("; ".join(errors))
        return data

    def match(self, data, ev, fetched):
        recs = [r for v in data.values() for r in v]
        return match_solar(recs, ev, self.window_s)


class LightningProvider(Provider):
    """Lightning strikes from a URL or a local file (format in the module docstring)"""
    name = "lightning"

    def __init__(self, url, keep_km=500.0, keep_s=48 * 3600.0, window_s=600.0,
                 radius_km=100.0, **kw):
        super().__init__(**kw)
        self.url, self.keep_km, self.keep_s = url, keep_km, keep_s
        self.window_s, self.radius_km = window_s, radius_km

    def fetch(self, previous):
        new = parse_strikes(self._read(self.url))
        now = time.time()
        keep = {tuple(s) for s in (previous or {}).get("strikes", []) if s[0] > now - self.keep_s}
        for t, lat, lon in new:
            if t > now - self.keep_s and \
                    distance_km(LWASV_LAT, LWASV_LON, lat, lon) <= self.keep_km:
                keep.add((t, lat, lon))
        return dict(strikes=sorted(keep), until=now)

    def match(self, data, ev, fetched):
        return match_lightning(data.get("strikes", []), ev, data.get("until", fetched),
                               self.window_s, self.radius_km)


# ------------------------------------------------------------------ service
class ContextService:
    """Polls the providers in the background and builds the context of an event"""

    def __init__(self, providers, cache_dir=None):
        self.providers = {p.name: p for p in providers}
        self.cache_dir = cache_dir
        self.lock = threading.Lock()
        self.state = {n: dict(data=None, fetched=None, error=None, attempts=0)
                      for n in self.providers}
        self.generation = 0  # increases whenever any provider's data change
        self.threads = []
        for n in self.providers:
            self._load_cache(n)

    # -------------------------------------------------- cache
    def _cache_path(self, name):
        return os.path.join(self.cache_dir, f"{name}.json") if self.cache_dir else None

    def _load_cache(self, name):
        p = self._cache_path(name)
        if not p or not os.path.exists(p):
            return
        try:
            with open(p) as f:
                c = json.load(f)
            self.state[name].update(data=c["data"], fetched=c["fetched"])
            if name == "lightning":
                c["data"]["strikes"] = [tuple(s) for s in c["data"].get("strikes", [])]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass

    def _save_cache(self, name, data, fetched):
        p = self._cache_path(name)
        if not p:
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(p + ".tmp", "w") as f:
                json.dump(dict(fetched=fetched, data=data), f)
            os.replace(p + ".tmp", p)
        except (OSError, TypeError, ValueError):
            pass

    # -------------------------------------------------- polling
    def set_data(self, name, data, fetched=None):
        """Install a provider's data (a successful fetch, or injected by a test)"""
        fetched = time.time() if fetched is None else fetched
        with self.lock:
            self.state[name].update(data=data, fetched=fetched, error=None)
            self.generation += 1
        self._save_cache(name, data, fetched)

    def refresh(self, name):
        """One fetch; a failure keeps the last good data and is logged once per outage"""
        prov, st = self.providers[name], self.state[name]
        st["attempts"] += 1
        try:
            data = prov.fetch(st["data"])
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
            if st["error"] is None:
                print(f"context: {name} unavailable ({err}); keeping the last data", flush=True)
            st["error"] = err
            return False
        if st["error"] is not None:
            print(f"context: {name} answering again", flush=True)
        self.set_data(name, data)
        return True

    def _loop(self, name, stop):
        while not stop.is_set():
            self.refresh(name)
            stop.wait(self.providers[name].interval_s)

    def start(self, stop: threading.Event):
        for n in self.providers:
            t = threading.Thread(target=self._loop, args=(n, stop), daemon=True,
                                 name=f"context-{n}")
            t.start()
            self.threads.append(t)

    # -------------------------------------------------- per event
    def for_event(self, ev):
        """The context of an event; never raises"""
        ctx = dict(sun_el_deg=None, sun_sep_deg=None, solar=None, lightning=None,
                   sources=[], unavailable=[])
        try:
            el, sep = sun_position(first_arrival(ev), ev.get("ra"), ev.get("dec"))
            ctx.update(sun_el_deg=round(el, 2), sun_sep_deg=None if sep is None else round(sep, 2))
        except Exception as e:
            ctx["sun_error"] = type(e).__name__
        for name, prov in self.providers.items():
            with self.lock:
                data, fetched = self.state[name]["data"], self.state[name]["fetched"]
            if data is None:
                ctx["unavailable"].append(name)
                continue
            try:
                ctx["solar" if name == "swpc" else name] = prov.match(data, ev, fetched)
                ctx["sources"].append(name)
            except Exception as e:
                ctx["unavailable"].append(name)
                ctx.setdefault("errors", {})[name] = type(e).__name__
        return ctx

    def status(self):
        with self.lock:
            return {n: dict(fetched=s["fetched"], error=s["error"], attempts=s["attempts"],
                            has_data=s["data"] is not None) for n, s in self.state.items()}


def make_service(swpc=True, lightning_url=None, cache_dir=None, interval_s=900.0,
                 timeout_s=20.0):
    providers = []
    if swpc:
        providers.append(SwpcProvider(interval_s=interval_s, timeout_s=timeout_s))
    if lightning_url:
        providers.append(LightningProvider(lightning_url, interval_s=interval_s,
                                           timeout_s=timeout_s))
    return ContextService(providers, cache_dir)
