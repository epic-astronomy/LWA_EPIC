"""
The catalogue of the all-sky monitor, in the head PostgreSQL, for the website.

The combiner writes it (combine --catalogue DSN): every FRB event (kind
"sky") with its state and reviews, one detection row per band of an event
(what the band detected or measured, with the candidate's record when the
combiner has read it), and every giant pulse of a pulsar known for them (the
Crab), which belongs to no event. Ordinary pulsar pulses, RFI and test
injections are not catalogued.

Each detection names its place in the archive, <node>/<YYYYMMDD>/<candidate>,
as the archiver lays it out (archiver.py), whose .npz and .json the website
serves once the data are there.

If the database cannot be reached, rows go to a spool directory and are
written when it is back, newest state last; a spooled row is never lost.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone

SCHEMA = """
CREATE SCHEMA IF NOT EXISTS skymon;
CREATE TABLE IF NOT EXISTS skymon.events (
    id           text PRIMARY KEY,
    status       text NOT NULL,
    review       text,
    t_inf        timestamptz NOT NULL,
    ra           double precision,
    dec          double precision,
    dm           double precision,
    dm_err       double precision,
    snr          double precision,
    n_seen       integer,
    n_could_see  integer,
    version      integer NOT NULL DEFAULT 0,
    record       jsonb NOT NULL,
    updated      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS events_t_inf ON skymon.events (t_inf DESC);
CREATE TABLE IF NOT EXISTS skymon.detections (
    node          text NOT NULL,
    cand_id       text NOT NULL,
    kind          text NOT NULL,          -- frb | giant_pulse
    event_id      text REFERENCES skymon.events (id) ON DELETE SET NULL,
    state         text,                   -- detected | followup (frb); detected (giant_pulse)
    pulsar        text,
    t_inf         timestamptz NOT NULL,
    f_lo_mhz      double precision,
    f_hi_mhz      double precision,
    snr           double precision,
    dm            double precision,
    width_s       double precision,
    fluence_jy_ms double precision,
    ra            double precision,
    dec           double precision,
    archive_path  text NOT NULL,          -- <node>/<YYYYMMDD>/<candidate>, without extension
    event_version integer NOT NULL DEFAULT 0,  -- the event's version it was written from
    record        jsonb NOT NULL,
    updated       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (node, cand_id)
);
CREATE INDEX IF NOT EXISTS detections_t_inf ON skymon.detections (t_inf DESC);
CREATE INDEX IF NOT EXISTS detections_event ON skymon.detections (event_id);
CREATE INDEX IF NOT EXISTS detections_kind ON skymon.detections (kind, t_inf DESC);
"""


def archive_path(node: str, cand_id: str, t_inf: float) -> str:
    """The candidate's place in the archive, as archiver.py lays it out (no extension)"""
    day = datetime.fromtimestamp(t_inf, timezone.utc).strftime("%Y%m%d")
    return f"{node}/{day}/{cand_id}"


def _ts(t):
    return None if t is None else datetime.fromtimestamp(float(t), timezone.utc)


def _finite(o):
    if isinstance(o, float):
        return o if o == o and abs(o) != float("inf") else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def is_giant_pulse(c: dict) -> bool:
    ks = c.get("known_source") or {}
    return c.get("cls") == "known" and ks.get("kind") == "pulsar" and bool(ks.get("giant_pulse"))


class Catalogue:
    def __init__(self, dsn: str | None, spool_dir: str, create: bool = True):
        self.dsn = dsn
        self.spool = spool_dir
        self.create = create
        os.makedirs(spool_dir, exist_ok=True)
        self.conn = None
        self.lock = threading.Lock()
        self.errors = 0
        self.last_error = None
        self.written = 0

    # ------------------------------------------------------------ connection
    def _conn(self):
        if not self.dsn:
            return None
        if self.conn is not None and not self.conn.closed:
            return self.conn
        try:
            from .localdb import connect
            self.conn = connect(self.dsn, timeout=10.0)
            if self.create:
                try:  # the deployment creates the schema; a writer without the right is fine
                    with self.conn.cursor() as cur:
                        cur.execute(SCHEMA)
                except Exception:
                    pass
            return self.conn
        except Exception as e:
            self._fail(e)
            return None

    def _fail(self, e):
        self.errors += 1
        self.last_error = f"{type(e).__name__}: {e}"[:300]
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass
        self.conn = None

    def _run(self, statements) -> bool:
        """Run statements in one transaction; False (and nothing written) if that fails"""
        with self.lock:
            conn = self._conn()
            if conn is None:
                return False
            try:
                conn.autocommit = False
                with conn.cursor() as cur:
                    for sql, args in statements:
                        cur.execute(sql, args)
                conn.commit()
                self.written += 1
                return True
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                self._fail(e)
                return False
            finally:
                if self.conn is not None:
                    self.conn.autocommit = True

    # ------------------------------------------------------------ rows
    @staticmethod
    def _event_sql(ev: dict):
        best = max([b.get("candidate", {}).get("snr") or 0 for b in (ev.get("bands") or {}).values()] or [0])
        return ("INSERT INTO skymon.events (id, status, review, t_inf, ra, dec, dm, dm_err, snr, n_seen, "
                "n_could_see, version, record, updated) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, now()) ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status, "
                "review = EXCLUDED.review, t_inf = EXCLUDED.t_inf, ra = EXCLUDED.ra, dec = EXCLUDED.dec, "
                "dm = EXCLUDED.dm, dm_err = EXCLUDED.dm_err, snr = EXCLUDED.snr, n_seen = EXCLUDED.n_seen, "
                "n_could_see = EXCLUDED.n_could_see, version = EXCLUDED.version, record = EXCLUDED.record, "
                "updated = now() WHERE skymon.events.version <= EXCLUDED.version",
                (ev["id"], ev["status"], ev.get("review"), _ts(ev["t_inf"]), ev.get("ra"), ev.get("dec"),
                 ev.get("dm"), ev.get("dm_err"), ev.get("combined_snr") or best or None,
                 len(ev.get("seen") or []), len(ev.get("could_see") or []), ev.get("version", 0),
                 json.dumps(_finite(ev))))

    @staticmethod
    def _detection_sql(node: str, kind: str, rec: dict, event_id: str | None, state: str | None,
                       pulsar: str | None, event_version: int = 0):
        flux = rec.get("flux") or {}
        t_inf = rec.get("t_inf")
        return ("INSERT INTO skymon.detections (node, cand_id, kind, event_id, state, pulsar, t_inf, f_lo_mhz, "
                "f_hi_mhz, snr, dm, width_s, fluence_jy_ms, ra, dec, archive_path, record, event_version, updated) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now()) "
                "ON CONFLICT (node, cand_id) DO UPDATE SET kind = EXCLUDED.kind, "
                "event_id = COALESCE(EXCLUDED.event_id, skymon.detections.event_id), state = EXCLUDED.state, "
                "pulsar = EXCLUDED.pulsar, t_inf = EXCLUDED.t_inf, f_lo_mhz = EXCLUDED.f_lo_mhz, "
                "f_hi_mhz = EXCLUDED.f_hi_mhz, snr = EXCLUDED.snr, dm = EXCLUDED.dm, width_s = EXCLUDED.width_s, "
                "fluence_jy_ms = EXCLUDED.fluence_jy_ms, ra = EXCLUDED.ra, dec = EXCLUDED.dec, "
                "record = EXCLUDED.record, event_version = EXCLUDED.event_version, updated = now() "
                # an older version of the event, arriving late, does not overwrite a newer one
                "WHERE skymon.detections.event_version <= EXCLUDED.event_version",
                (node, rec["id"], kind, event_id, state, pulsar, _ts(t_inf), rec.get("f_lo_mhz"),
                 rec.get("f_hi_mhz"), rec.get("snr"), rec.get("dm"), rec.get("width_s"),
                 flux.get("fluence_jy_ms") if isinstance(flux, dict) else None, rec.get("ra"), rec.get("dec"),
                 archive_path(node, rec["id"], t_inf), json.dumps(_finite(rec)), event_version))

    def _event_statements(self, ev: dict, records: dict):
        """The event's row and one row per band that detected or measured it; and the records used"""
        stmts, used = [self._event_sql(ev)], {}
        for node, b in (ev.get("bands") or {}).items():
            m = b.get("candidate") or b.get("followup")
            cid = (b.get("candidate") or {}).get("id") or b.get("data_id")
            if not m or not cid:
                continue
            rec = dict(records.get((node, cid)) or {})
            if rec:
                used[f"{node}|{cid}"] = rec
            rec.setdefault("id", cid)
            for k in ("snr", "dm", "t_inf", "ra", "dec", "width_s"):
                if rec.get(k) is None and m.get(k) is not None:
                    rec[k] = m[k]
            rec.setdefault("t_inf", ev["t_inf"])
            rec.setdefault("f_lo_mhz", b.get("f_lo"))
            rec.setdefault("f_hi_mhz", b.get("f_hi"))
            if rec.get("flux") is None and m.get("fluence_jy_ms") is not None:
                rec["flux"] = dict(fluence_jy_ms=m["fluence_jy_ms"])
            stmts.append(self._detection_sql(node, "frb", rec, ev["id"],
                                             "detected" if b.get("candidate") else "followup", None,
                                             ev.get("version", 0)))
        return stmts, used

    def put_event(self, ev: dict, records: dict | None = None) -> bool:
        """
        An FRB event and its detections. records maps (node, candidate id) to the
        candidate records the combiner has read; a band without one is written from
        what the event holds of it. Events of other kinds are not catalogued.
        """
        if ev.get("kind") != "sky":
            return True
        stmts, used = self._event_statements(ev, records or {})
        return self._write(dict(kind="event", ev=ev, records=used), stmts)

    def put_giant_pulse(self, node: str, rec: dict) -> bool:
        ks = rec.get("known_source") or {}
        return self._write(dict(kind="giant_pulse", node=node, rec=rec),
                           [self._detection_sql(node, "giant_pulse", rec, None, "detected", ks.get("name"))])

    # ------------------------------------------------------------ spool
    def _write(self, item, stmts) -> bool:
        if self.dsn and self._run(stmts):
            return True
        if self.dsn:
            self._spool_item(item)
        return False

    def _spool_item(self, item):
        p = os.path.join(self.spool, f"{time.time():.6f}-{uuid.uuid4().hex[:6]}.json")
        with open(p + ".tmp", "w") as f:
            json.dump(_finite(item), f)
        os.replace(p + ".tmp", p)

    def replay(self) -> int:
        """Write the spooled rows, oldest first; stops at the first failure"""
        if not self.dsn:
            return 0
        done = 0
        for n in sorted(x for x in os.listdir(self.spool) if x.endswith(".json")):
            p = os.path.join(self.spool, n)
            try:
                with open(p) as f:
                    item = json.load(f)
            except (OSError, ValueError):
                continue
            if item.get("kind") == "event":
                recs = {tuple(k.split("|", 1)): v for k, v in (item.get("records") or {}).items()}
                ok = self._run(self._event_statements(item["ev"], recs)[0])
            elif item.get("kind") == "giant_pulse":
                rec = item["rec"]
                ok = self._run([self._detection_sql(item["node"], "giant_pulse", rec, None, "detected",
                                                    (rec.get("known_source") or {}).get("name"))])
            else:
                ok = True
            if not ok:
                break
            os.remove(p)
            done += 1
        return done

    def spooled(self) -> int:
        return sum(1 for x in os.listdir(self.spool) if x.endswith(".json"))

    def status(self):
        return dict(enabled=bool(self.dsn), written=self.written, errors=self.errors,
                    last_error=self.last_error, spooled=self.spooled())
