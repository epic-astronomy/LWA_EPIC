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

def ingest_daily_digest(ihdu, phdu, data):
    indices = get_pixel_indices2(ihdu, phdu, 0)
    print(indices)
    #print(indices.keys())
    if indices["nsrc"][0] <= 0:
        return
    nsrc = indices["nsrc"][0]
    ncoords = indices["ncoords"][0]//nsrc
    

    rows = []
    # loop over each source and insert
    for i in range(nsrc):
        stokes_V_re = data[
            2,  # X*X, Y*Y
            :,  # all channels
            indices["pix_y"][i * ncoords : (i + 1) * ncoords].astype(int) - 1,
            indices["pix_x"][i * ncoords : (i + 1) * ncoords].astype(int) - 1,
        ].sum(axis=(1))
        stokes_V_im = data[
            2,  # X*X, Y*Y
            :,  # all channels
            indices["pix_y"][i * ncoords : (i + 1) * ncoords].astype(int) - 1,
            indices["pix_x"][i * ncoords : (i + 1) * ncoords].astype(int) - 1,
        ].sum(axis=(1))

        stokes_V = np.sqrt(stokes_V_re**2 + stokes_V_im**2)
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
                stokes_v=stokes_V.tolist()
            )
        )
    #print(rows)
    # print(data[
    #                 0,  # X*X, Y*Y
    #                 0,  # all channels
    #                 indices["pix_y"][0:25].astype(
    #                     int
    #                 )
    #                 - 1,
    #                 indices["pix_x"][0:25].astype(
    #                     int
    #                 )
    #                 - 1,
    #             ])
    engine = create_engine("postgresql:///")
    insert_stmnt = insert(EpicDailyDigestTable).values(rows)
    with engine.connect() as conn:
        conn.execute(insert_stmnt)
        conn.commit()

def test_extraction(
    file="accumulated_files/EPIC_1729035156.560_42.975MHz.fits",
):
    with fits.open(file) as hdul:
        phdu = hdul[0].header
        ihdu = hdul[1].header
        data = hdul[1].data

    ingest_daily_digest(ihdu,phdu, data)
    # print(phdu, ihdu)
    
    # connection = engine.connect()
    # session = Session(connection)
    # session.execute(insert_stmnt)
    # session.commit()
    # session.close()
    # print(data.shape, indices)
    # print(
    #     data[
    #         0:2,
    #         :,
    #         indices["pix_y"].astype(int) - 1,
    #         indices["pix_x"].astype(int) - 1,
    #     ].sum(axis=(0, 2))
    # )
    # wcs=WCS(ihdu, naxis=2)
    # print(indices)
    # radec=DynSources.get_lwasv_skypos('sun',ihdu['DATETIME'])
    # print(wcs.all_world2pix([radec[0]],[radec[1]],1, ra_dec_order=True))


if __name__ == "__main__":
    test_extraction()
