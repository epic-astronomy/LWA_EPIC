"""
HTTP interface of a running monitor (standard library only).

  GET /api/status                      state, band, tiers, real-time metrics, injection stats
  GET /api/pixel?ra=&dec=              spectrogram of the tracked pixel nearest to a position
      /api/pixel?hpix=  /api/pixel?x=&y=  (HEALPix index, or 1-based image pixel)
      &since=<unix>                    only accumulation rows from this time
      &fine=1                          the last tier-0 window (83.6 ms, normalised) instead
      &format=npz                      binary npz instead of JSON
  GET /api/candidates?since=&cls=&limit=   recent candidates (newest last)
  GET /api/candidates/<id>             one candidate
  GET /api/candidates/<id>/npz         its data products
  GET /api/sky                         tracked pixels with their age and noise
  GET /api/injections                  live injection summary
  GET /api/watches?event=              watches requested by the combiner, and their results
  GET /api/conditions?since=<unix>     data-quality samples (conditions.py), every 10 s of
                                       data over the last 6 h, with the latest summary
  POST /api/watch    {ra, dec, dm, t_inf, dm_err, t_tol, id, event_id}
                                       measure this position at the burst's predicted
                                       arrival in this band (watch.py)
  POST /api/inject   {id, ra, dec, dm, t_inf, snr, width_s}
                                       inject this burst (coordinated across bands)

POST requests need the header X-Skymon-Token when the environment variable
SKYMON_TOKEN is set.

Each monitor serves only its own band. The website queries every worker's
monitor and shows whichever answer; one being down does not affect the others.
"""
from __future__ import annotations

import hmac
import io
import json
import math
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def _jsonable(o):
    if isinstance(o, np.ndarray):
        if o.dtype.kind == "f":
            return [_jsonable(v) for v in o] if o.ndim > 1 else \
                [None if not math.isfinite(v) else round(float(v), 6) for v in o]
        return o.tolist()
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not math.isfinite(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def make_handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        server_version = "skymon/1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(_jsonable(obj), allow_nan=False).encode())

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}
            parts = [p for p in u.path.split("/") if p]
            try:
                if parts[:2] == ["api", "status"]:
                    return self._json(monitor.status())
                if parts[:2] == ["api", "pixel"]:
                    f = lambda k: float(q[k]) if k in q else None
                    res = monitor.pixel(ra=f("ra"), dec=f("dec"),
                                        hpix=int(q["hpix"]) if "hpix" in q else None,
                                        x=f("x"), y=f("y"), since=f("since"),
                                        fine=q.get("fine") in ("1", "true"))
                    if res is None:
                        return self._json(dict(error="not running"), 503)
                    if q.get("format") == "npz":
                        buf = io.BytesIO()
                        np.savez(buf, times=np.asarray(res["times"], float),
                                 data=np.asarray(res["data"], np.float32),
                                 freqs_mhz=np.asarray(res["freqs_mhz"]))
                        return self._send(200, buf.getvalue(), "application/octet-stream")
                    return self._json(res)
                if parts[:2] == ["api", "candidates"]:
                    if len(parts) == 2:
                        since = float(q.get("since", 0))
                        lst = monitor.store.list(since, q.get("cls"), int(q.get("limit", 500)))
                        return self._json([c.to_dict() for c in lst])
                    c = monitor.store.get(parts[2])
                    if c is None:
                        return self._json(dict(error="unknown candidate"), 404)
                    if len(parts) > 3 and parts[3] == "npz":
                        p = c.files.get("npz")
                        if not p:
                            return self._json(dict(error="no data products"), 404)
                        with open(p, "rb") as fh:
                            return self._send(200, fh.read(), "application/octet-stream")
                    return self._json(c.to_dict())
                if parts[:2] == ["api", "sky"]:
                    return self._json(monitor.sky())
                if parts[:2] == ["api", "injections"]:
                    return self._json(monitor.injector.summary())
                if parts[:2] == ["api", "watches"]:
                    return self._json(monitor.watch_list(q.get("event")))
                if parts[:2] == ["api", "conditions"]:
                    since = float(q["since"]) if "since" in q else None
                    return self._json(monitor.conditions(since))
                return self._json(dict(error="not found"), 404)
            except (ValueError, KeyError) as e:
                return self._json(dict(error=str(e)), 400)
            except Exception as e:  # report, keep serving
                return self._json(dict(error=repr(e)), 500)

        def do_POST(self):
            token = os.environ.get("SKYMON_TOKEN")
            if token and not hmac.compare_digest(self.headers.get("X-Skymon-Token", ""), token):
                return self._json(dict(error="forbidden"), 403)
            u = urllib.parse.urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 65536:
                    return self._json(dict(error="request too large"), 413)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("a JSON object is expected")
                if parts[:2] == ["api", "watch"]:
                    w = monitor.watches.add(body)
                    return self._json({k: v for k, v in w.items() if not k.startswith("_")})
                if parts[:2] == ["api", "inject"]:
                    b = monitor.injector.add_external(body)
                    return self._json(dict(id=b.id, accepted=True))
                return self._json(dict(error="not found"), 404)
            except (ValueError, KeyError, TypeError) as e:
                return self._json(dict(error=str(e)), 400)
            except Exception as e:
                return self._json(dict(error=repr(e)), 500)

    return Handler


def serve(monitor, host="0.0.0.0", port=8095):
    httpd = ThreadingHTTPServer((host, port), make_handler(monitor))
    httpd.daemon_threads = True
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd
