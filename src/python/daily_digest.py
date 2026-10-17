import collections
import os
import sys
import time

from pixel_extractor import get_pixel_indices2
from astropy.io import fits
from pixel_extract_utils import DynSources
from astropy.wcs import WCS
import numpy as np
from sqlalchemy import (
    create_engine,
    Table,
    Column,
    Integer,
    Float,
    String,
    MetaData,
    ARRAY,
    Text,
    TIMESTAMP,
    MetaData
)
from sqlalchemy.orm import Session, declarative_base
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

meta_obj = MetaData(schema="sevilleta")

# from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# Rows go to the node's own database: EPIC_NODE_DSN (a libpq connection
# string) if set, otherwise the libpq PG* environment variables. A database
# that is down or slow must not cost the imager its long-integration files:
# every write has short time limits, rows that could not be written wait in
# memory (up to PENDING_MAX) and go with the next write, and errors are logged
# at most once per LOG_EVERY_S. Nothing here raises.
PENDING_MAX = 20000
LOG_EVERY_S = 300.0
_state = {"engine": None, "pending": collections.deque(maxlen=PENDING_MAX),
          "log_at": {}, "dropped": 0}


def _log_limited(key, msg):
    now = time.monotonic()
    if now - _state["log_at"].get(key, -1e9) >= LOG_EVERY_S:
        _state["log_at"][key] = now
        print(f"daily_digest: {msg}", file=sys.stderr, flush=True)


def _engine():
    if _state["engine"] is None:
        import psycopg2

        dsn = os.environ.get("EPIC_NODE_DSN", "")  # "": libpq PG* variables

        def connect():
            return psycopg2.connect(dsn, connect_timeout=5,
                                    options="-c statement_timeout=10000",
                                    application_name="epic-imager-digest")

        _state["engine"] = create_engine(
            "postgresql+psycopg2://", creator=connect,
            pool_pre_ping=True, pool_size=1, max_overflow=0)
    return _state["engine"]


def write_digest_rows(rows):
    """
    Insert digest rows, with any left from earlier failures. Returns the
    number written; on failure keeps them (the oldest are dropped beyond
    PENDING_MAX) and returns 0.
    """
    pending = _state["pending"]
    for r in rows:
        if len(pending) == pending.maxlen:
            _state["dropped"] += 1
        pending.append(r)
    if not pending:
        return 0
    batch = list(pending)
    try:
        with _engine().connect() as conn:
            conn.execute(insert(EpicDailyDigestTable).values(batch))
            conn.commit()
    except Exception as e:
        _log_limited("write", f"{len(batch)} rows waiting for the database "
                     f"({_state['dropped']} dropped so far): "
                     f"{str(e).strip().splitlines()[0] if str(e) else type(e).__name__}")
        return 0
    for _ in batch:
        pending.popleft()
    return len(batch)


class EpicDailyDigestTable(Base):
    __tablename__ = "epic_daily_digest"
    metadata = meta_obj
    source_name = Column(Text, nullable=False, primary_key=True)
    img_time = Column(TIMESTAMP, nullable=False, primary_key=True)
    stokes_i = Column(ARRAY(Float), nullable=False)
    stokes_v = Column(ARRAY(Float), nullable=False)
    cfreq = Column(Float, nullable=False, primary_key=True)

def _digest_rows(ihdu, phdu, data, cfreq):
    """
    Extract pixel values for each source in the watch list, 
    create stokes I and V by adding all the values in the kernel,
    and ingest it into the DB
    """
    indices = get_pixel_indices2(ihdu, phdu, 0)
    if indices["nsrc"][0] <= 0:
        return []
    nsrc = indices["nsrc"][0]
    ncoords = indices["ncoords"][0]//nsrc
    

    rows = []
    # loop over each source and insert
    for i in range(nsrc):
        # XYc_re = data[
        #     2,  # Re(X*Y)
        #     :,  # all channels
        #     indices["pix_y"][i * ncoords : (i + 1) * ncoords].astype(int) - 1,
        #     indices["pix_x"][i * ncoords : (i + 1) * ncoords].astype(int) - 1,
        # ]/1e3
        Xyc_im = data[
            3,  # Im(X*Y)
            :,  # all channels
            indices["pix_y"][i * ncoords : (i + 1) * ncoords].astype(int) - 1,
            indices["pix_x"][i * ncoords : (i + 1) * ncoords].astype(int) - 1,
        ]

        stokes_V = np.abs(Xyc_im).sum(axis=0)
        rows.append(
            dict(
                source_name=indices["src_ids"][i * ncoords],
                img_time=ihdu["DATETIME"],
                stokes_i=data[
                    0:2,  # X*X, Y*Y
                    :,  # all channels
                    indices["pix_y"][i * ncoords : (i + 1) * ncoords].astype(
                        int
                    )
                    - 1,
                    indices["pix_x"][i * ncoords : (i + 1) * ncoords].astype(
                        int
                    )
                    - 1,
                ].sum(axis=(0, 2)).tolist(),
                stokes_v=stokes_V.tolist(),
                cfreq=cfreq
            )
        )

    return rows

def ingest_daily_digest(ihdu, phdu, data, cfreq):
    """
    Extract pixel values for each source in the watch list,
    create stokes I and V by adding all the values in the kernel,
    and ingest it into the DB. Never raises.
    """
    try:
        rows = _digest_rows(ihdu, phdu, data, cfreq)
    except Exception as e:
        _log_limited("extract", f"digest not computed: {e!r}")
        return 0
    return write_digest_rows(rows)


def test_extraction(
    file="accumulated_files/EPIC_1729064709.560_87.875MHz.fits",
):
    with fits.open(file) as hdul:
        phdu = hdul[0].header
        ihdu = hdul[1].header
        data = hdul[1].data

    ingest_daily_digest(ihdu,phdu, data, -1)


if __name__ == "__main__":
    test_extraction()
