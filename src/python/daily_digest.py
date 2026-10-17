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


def digest_spectra(data, pix_y, pix_x, nimg=1):
    """
    Stokes I and V spectra of one source's patch, as means per image.

    data [4][chan][y][x] holds X*X, Y*Y, Re(X*Y), Im(X*Y) summed over nimg
    images; pix_y, pix_x are the patch's pixels (0-based). Each spectrum sums
    the patch's pixels in every channel:
      I = XX + YY,
      V = 2 Im(XY*) (signed; the sign convention and the X-Y phase of the
          array are not calibrated, so V is a relative measure).
    """
    scale = 1.0 / max(1, int(nimg))
    patch = data[:, :, pix_y, pix_x]  # [4][chan][npix]
    stokes_i = (patch[0] + patch[1]).sum(axis=1) * scale
    stokes_v = 2.0 * patch[3].sum(axis=1) * scale
    return stokes_i, stokes_v


def ingest_daily_digest(ihdu, phdu, data, cfreq, nimg=1):
    """
    Extract pixel values for each source in the watch list,
    create stokes I and V spectra by adding the kernel's pixels in each
    channel (digest_spectra),
    and ingest it into the DB.

    data is the sum of nimg images; the spectra are divided by nimg (means per
    image), so that partial and full accumulations, and accumulations of any
    length, are on the same scale.
    """
    indices = get_pixel_indices2(ihdu, phdu, 0)
    if indices["nsrc"][0] <= 0:
        return
    nsrc = indices["nsrc"][0]
    ncoords = indices["ncoords"][0]//nsrc
    

    rows = []
    # loop over each source and insert
    for i in range(nsrc):
        sl = slice(i * ncoords, (i + 1) * ncoords)
        stokes_i, stokes_v = digest_spectra(
            data, indices["pix_y"][sl].astype(int) - 1, indices["pix_x"][sl].astype(int) - 1, nimg)
        rows.append(
            dict(
                source_name=indices["src_ids"][i * ncoords],
                img_time=ihdu["DATETIME"],
                stokes_i=stokes_i.tolist(),
                stokes_v=stokes_v.tolist(),
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
