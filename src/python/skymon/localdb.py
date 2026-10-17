"""
The local database of one imaging node, and the files beside it.

Each node keeps its own PostgreSQL instance on local disk (not the cluster's
Citus worker, whose data live on the shared file system) and a local data
directory. Nothing in the real-time path touches the shared file system,
which serves only as the archive (archiver.py).

The database is also the node's mailbox:

  candidates  every stored candidate (metadata as JSON, data products as a
              local file); the combiner reads new rows by id
  watches     measurements requested by the combiner, with their results
  injections  coordinated injections requested by the combiner
  events      the combiner's events, copied to every node it can reach, so
              that a combiner started on any node can continue
  status      the monitor's health counters
  conditions  the monitor's data-quality samples (conditions.py), one row
              per 10 s of data; samples wait in memory while the database
              is away
  calibration streamcal's calibration records (kind "record", one per
              solution) and the calibrators measured in the images (kind
              "measurement", one per minute), for flux and fluence (flux.py)

If the database cannot be reached, the monitor keeps searching: records go
to a spool directory on local disk and are inserted when the database is
back. A spooled record is never lost.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id          bigserial PRIMARY KEY,
    cid         text UNIQUE NOT NULL,
    cls         text NOT NULL,
    created     timestamptz NOT NULL DEFAULT now(),
    t_inf       double precision,
    snr         real,
    dm          real,
    event_id    text,
    record      jsonb NOT NULL,
    npz_path    text,
    archived    timestamptz
);
CREATE INDEX IF NOT EXISTS candidates_t_inf ON candidates (t_inf);
CREATE INDEX IF NOT EXISTS candidates_event ON candidates (event_id);
CREATE TABLE IF NOT EXISTS watches (
    id          text PRIMARY KEY,
    event_id    text,
    request     jsonb NOT NULL,
    state       text NOT NULL DEFAULT 'pending',
    result      jsonb,
    reason      text,
    context_id  text,
    created     timestamptz NOT NULL DEFAULT now(),
    updated     timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS injections (
    id          text PRIMARY KEY,
    spec        jsonb NOT NULL,
    state       text NOT NULL DEFAULT 'requested',
    created     timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS events (
    id          text PRIMARY KEY,
    version     bigint NOT NULL,
    status      text,
    review      text,
    t_inf       double precision,
    state       jsonb NOT NULL,
    updated     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS events_t_inf ON events (t_inf);
CREATE TABLE IF NOT EXISTS status (
    key         text PRIMARY KEY,
    value       jsonb NOT NULL,
    updated     timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS conditions (
    t           double precision PRIMARY KEY,
    value       jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS calibration (
    kind         text NOT NULL,
    t            double precision NOT NULL,
    gain_version bigint,
    value        jsonb NOT NULL,
    PRIMARY KEY (kind, t)
);
"""


def connect(dsn: str, timeout: float = 3.0):
    import psycopg2

    conn = psycopg2.connect(dsn, connect_timeout=max(1, int(timeout)),
                            options=f"-c statement_timeout={int(timeout * 1000)}")
    conn.autocommit = True
    return conn


class LocalDB:
    """
    The monitor's side: a connection that is reopened when lost, and a spool
    on local disk for records that could not be written.
    """

    def __init__(self, dsn: str | None, spool_dir: str):
        self.dsn = dsn
        self.spool = spool_dir
        os.makedirs(spool_dir, exist_ok=True)
        self.conn = None
        self.lock = threading.Lock()
        self.errors = 0
        self.last_error = None
        self.ready = False

    # ------------------------------------------------------------ connection
    def _conn(self):
        if self.dsn is None:
            return None
        if self.conn is not None and not self.conn.closed:
            return self.conn
        try:
            self.conn = connect(self.dsn)
            with self.conn.cursor() as cur:
                cur.execute(SCHEMA)
            self.ready = True
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

    def execute(self, sql, args=None, fetch=False):
        """Run one statement; returns rows (fetch) or True, or None if the database is away"""
        with self.lock:
            conn = self._conn()
            if conn is None:
                return None
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, args)
                    return cur.fetchall() if fetch else True
            except Exception as e:
                self._fail(e)
                return None

    # ------------------------------------------------------------ candidates
    def put_candidate(self, rec: dict, npz_path: str | None) -> bool:
        """Insert (or update) a candidate row; spool it on local disk if that fails"""
        ok = self.execute(
            "INSERT INTO candidates (cid, cls, t_inf, snr, dm, event_id, record, npz_path) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (cid) DO UPDATE SET "
            "cls = EXCLUDED.cls, record = EXCLUDED.record, "
            "npz_path = COALESCE(EXCLUDED.npz_path, candidates.npz_path), "
            "event_id = COALESCE(EXCLUDED.event_id, candidates.event_id)",
            (rec["id"], rec["cls"], rec.get("t_inf"), rec.get("snr"), rec.get("dm"),
             rec.get("event_id"), json.dumps(_finite(rec)), npz_path))
        if not ok:
            self._spool(dict(kind="candidate", rec=rec, npz_path=npz_path))
        return bool(ok)

    def _spool(self, item):
        p = os.path.join(self.spool, f"{time.time():.6f}-{uuid.uuid4().hex[:6]}.json")
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_finite(item), f)
        os.replace(tmp, p)

    def replay(self) -> int:
        """Insert spooled records, oldest first; returns how many went in"""
        names = sorted(n for n in os.listdir(self.spool) if n.endswith(".json"))
        done = 0
        for n in names:
            p = os.path.join(self.spool, n)
            try:
                with open(p) as f:
                    item = json.load(f)
            except (OSError, ValueError):
                continue
            if item.get("kind") == "candidate":
                ok = self.execute(
                    "INSERT INTO candidates (cid, cls, t_inf, snr, dm, event_id, record, npz_path) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (cid) DO UPDATE SET "
                    "record = EXCLUDED.record, cls = EXCLUDED.cls",
                    (item["rec"]["id"], item["rec"]["cls"], item["rec"].get("t_inf"),
                     item["rec"].get("snr"), item["rec"].get("dm"), item["rec"].get("event_id"),
                     json.dumps(item["rec"]), item.get("npz_path")))
            elif item.get("kind") == "watch":
                ok = self.update_watch(item["watch"], spool=False)
            else:
                ok = True
            if not ok:
                break
            os.remove(p)
            done += 1
        return done

    def spooled(self) -> int:
        return sum(1 for n in os.listdir(self.spool) if n.endswith(".json"))

    # ------------------------------------------------------------ mailbox
    def new_watches(self, known: set):
        """Pending watches not yet seen by this monitor"""
        rows = self.execute("SELECT id, request FROM watches WHERE state = 'pending'", fetch=True)
        if rows is None:
            return None
        return [dict(r[1], id=r[0]) for r in rows if r[0] not in known]

    def update_watch(self, w: dict, spool: bool = True) -> bool:
        ok = self.execute(
            "UPDATE watches SET state = %s, result = %s, reason = %s, context_id = %s, "
            "updated = now() WHERE id = %s",
            (w["state"], json.dumps(_finite(w.get("result"))) if w.get("result") else None,
             w.get("reason"), w.get("context_id"), w["id"]))
        if not ok and spool:
            self._spool(dict(kind="watch", watch={k: w.get(k) for k in
                                                  ("id", "state", "result", "reason",
                                                   "context_id")}))
        return bool(ok)

    def new_injections(self, known: set):
        rows = self.execute("SELECT id, spec FROM injections WHERE state = 'requested'",
                            fetch=True)
        if rows is None:
            return None
        return [dict(r[1], id=r[0]) for r in rows if r[0] not in known]

    def set_injection(self, iid: str, state: str):
        return self.execute("UPDATE injections SET state = %s WHERE id = %s", (state, iid))

    def put_status(self, key: str, value: dict):
        return self.execute(
            "INSERT INTO status (key, value, updated) VALUES (%s, %s, now()) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated = now()",
            (key, json.dumps(_finite(value))))

    def put_conditions(self, samples: list) -> bool:
        """Insert conditions samples (one statement); False if the database is away"""
        if not samples:
            return True
        sql = "INSERT INTO conditions (t, value) VALUES " + \
            ", ".join(["(%s, %s)"] * len(samples)) + \
            " ON CONFLICT (t) DO UPDATE SET value = EXCLUDED.value"
        args = []
        for s in samples:
            args += [s["t"], json.dumps(_finite(s))]
        return bool(self.execute(sql, args))


    def put_calibration(self, kind: str, t: float, gain_version, value: dict) -> bool:
        """One calibration record or measurement; False if the database is away"""
        return bool(self.execute(
            "INSERT INTO calibration (kind, t, gain_version, value) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (kind, t) DO UPDATE SET value = EXCLUDED.value, "
            "gain_version = EXCLUDED.gain_version",
            (kind, float(t), None if gain_version is None else int(gain_version),
             json.dumps(_finite(value)))))


def _finite(o):
    """JSON-safe copy: NaN and infinities become null"""
    import math
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return _finite(o.item())
    except ImportError:
        pass
    return o
