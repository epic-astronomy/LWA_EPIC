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


class EpicDailyDigestTable(Base):
    __tablename__ = "epic_daily_digest"
    metadata = meta_obj
    source_name = Column(Text, nullable=False, primary_key=True)
    img_time = Column(TIMESTAMP, nullable=False, primary_key=True)
    stokes_i = Column(ARRAY(Float), nullable=False)
    stokes_v = Column(ARRAY(Float), nullable=False)
    cfreq = Column(Float, nullable=False, primary_key=True)

_engine = None


def _get_engine():
    """One engine per process, with time limits so a dead database cannot hang the writer"""
    global _engine
    if _engine is None:
        _engine = create_engine(
            "postgresql:///", pool_pre_ping=True,
            connect_args=dict(connect_timeout=10, keepalives=1, keepalives_idle=30,
                              keepalives_interval=10, keepalives_count=3,
                              options="-c statement_timeout=30000"))
    return _engine


def ingest_daily_digest(ihdu, phdu, data, cfreq, nimg=1):
    """
    Extract pixel values for each source in the watch list,
    create stokes I and V by adding all the values in the kernel,
    and ingest it into the DB.

    data is the sum of nimg images; the spectra are divided by nimg (means per
    image), so that partial and full accumulations, and accumulations of any
    length, are on the same scale.
    """
    scale = 1.0 / max(1, int(nimg))
    indices = get_pixel_indices2(ihdu, phdu, 0)
    if indices["nsrc"][0] <= 0:
        return
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

        stokes_V = np.abs(Xyc_im).sum(axis=0) * scale
        rows.append(
            dict(
                source_name=indices["src_ids"][i * ncoords],
                img_time=ihdu["DATETIME"],
                stokes_i=(data[
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
                ].sum(axis=(0, 2)) * scale).tolist(),
                stokes_v=stokes_V.tolist(),
                cfreq=cfreq
            )
        )

    insert_stmnt = insert(EpicDailyDigestTable).values(rows)
    with _get_engine().connect() as conn:
        conn.execute(insert_stmnt)
        conn.commit()

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
