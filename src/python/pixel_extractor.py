# from datetime import datetime
# from datetime import timedelta
# from typing import Any
# from typing import Dict
# from typing import List
# from typing import Optional
from typing import Union

# from uuid import uuid4

import numpy as np
import pandas as pd

# from astropy.io.fits import Header
from astropy.wcs import WCS

# from sqlalchemy import insert
# from sqlalchemy import select
# from sqlalchemy import update
# from sqlalchemy.sql.expression import bindparam

from pixel_extract_utils import DynSources
from pixel_extract_utils import PatchMan
from pixel_extract_utils import get_lmn_grid
from epic_grpc import epic_image_pb2_grpc
from epic_grpc import epic_image_pb2

# from ..epic_orm.pg_pixel_storage import EpicWatchdogTable
from epic_types import NDArrayBool_t
from epic_types import NDArrayNum_t

# from epic_types import Patch_t
# from epic_types import WatchMode_t

# import numpy as np
from lsl.common.stations import lwasv as lwasv_station

# from astropy.constants import c as speed_of_light
# import matplotlib.image
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5

# from scipy.signal import correlate2d

from epic_utils import get_ADP_time_from_unix_epoch
import grpc
import json

# import datetime
# import time
# from MCS2 import Communicator
# import uuid
# from lsl.common.stations import lwasv

FS = 196.0e6
CHAN_BW = 25.0e3
# from .service_hub import ServiceHub


def get_watch_list(watchdog_endpoint):
    """
    Fetch the watchlist for EPIC at Sevilleta
    """
    # print("fetching watchlist", watchdog_endpoint)
    with grpc.insecure_channel(watchdog_endpoint) as channel:
        stub = epic_image_pb2_grpc.epic_post_processStub(channel)
        response = stub.fetch_watchlist(epic_image_pb2.empty())
        df_json = json.loads(response.pd_json)
        df = pd.read_json(df_json)
        df.rename(columns=dict(patch_type="kernel_dim"), inplace=True)
    # df = pd.DataFrame(
    #     dict(id=[0], source_name=["sun"], ra=[0.0], dec=[0.0], kernel_dim=[5])
    # )
        return df


def get_image_headers_sv(seq_start_id, grid_size, grid_res):
    """
    Generate image header for an observation taken at the specified time
    at Sevilleta
    """
    phdu = fits.PrimaryHDU()
    time_tag = get_ADP_time_from_unix_epoch() + seq_start_id / CHAN_BW
    phdu.header["DATE-OBS"] = Time(
        time_tag,
        format="unix",
        precision=6,
    ).isot

    sll = (2 * grid_size * np.sin(np.pi * grid_res / 360)) ** -1
    # cfreq = (metadata["chan0"] + metadata["nchan"] / 2 - 1) * CHAN_BW

    phdu.header["BUNITS"] = "UNCALIB"
    phdu.header["BSCALE"] = 1e0
    phdu.header["BZERO"] = 0e0
    phdu.header["EQUINOX"] = "J2000"
    phdu.header["EXTNAME"] = "PRIMARY"
    phdu.header["GRIDDIMX"] = grid_size
    phdu.header["GRIDDIMY"] = grid_size
    phdu.header["DGRIDX"] = sll
    phdu.header["DGRIDY"] = sll
    # phdu.header["INTTIM"] = metadata["img_len_ms"] * 1e-3
    phdu.header["INTTIMU"] = "SECONDS"
    # phdu.header["CFREQ"] = cfreq
    # phdu.header["CFREQU"] = "HZ"

    # dt = TimeDelta(1e-3 * metadata["img_len_ms"], format="sec")
    dtheta_x = 2 * np.arcsin(0.5 / (grid_size * sll))
    dtheta_y = 2 * np.arcsin(0.5 / (grid_size * sll))
    crit_pix_x = float(grid_size / 2 + 1)

    # Need to correct for shift in center pixel when we flipped dec dimension
    # when writing npz, Only applies for even dimension size
    crit_pix_y = float(grid_size / 2 + 1) - (grid_size + 1) % 2
    delta_x = -dtheta_x * 180.0 / np.pi
    delta_y = dtheta_y * 180.0 / np.pi
    # delta_f = CHAN_BW

    # crit_pix_f = (metadata["nchan"] - 1) * 0.5 + 1  # +1 for FITS numbering

    t0 = Time(
        time_tag,
        format="unix",
        precision=6,
        location=(
            lwasv_station.lon * 180.0 / np.pi,
            lwasv_station.lat * 180.0 / np.pi,
        ),
    )

    lsts = t0.sidereal_time("apparent")
    coords = SkyCoord(
        lsts.deg, lwasv_station.lat * 180.0 / np.pi, obstime=t0, unit="deg"
    ).transform_to(FK5(equinox=Time("J2000")))

    # img_data = np.transpose(output_arr[:, :, :, :], (0, 1, 3, 2))
    # img_data = np.transpose(output_arr[:, :, :, :], (3, 0, 2, 1))
    # img_data = np.fft.fftshift(img_data, axes=(2, 3))[:, :, ::-1, :]
    # img_data = img_data / img_data.max(axis=(2, 3), keepdims=True)
    ihdu = fits.ImageHDU(np.zeros((grid_size, grid_size)))

    ihdu.header["DATETIME"] = t0.isot
    ihdu.header["LST"] = lsts.hour
    ihdu.header["EQUINOX"] = 2000

    ihdu.header["CTYPE1"] = "RA---SIN"
    ihdu.header["CRPIX1"] = crit_pix_x
    ihdu.header["CDELT1"] = delta_x
    ihdu.header["CRVAL1"] = coords.ra.deg
    ihdu.header["CUNIT1"] = "deg"

    ihdu.header["CTYPE2"] = "DEC--SIN"
    ihdu.header["CRPIX2"] = crit_pix_y
    ihdu.header["CDELT2"] = delta_y
    ihdu.header["CRVAL2"] = coords.dec.deg
    ihdu.header["CUNIT2"] = "deg"
    # Coordinates - Freq
    # ihdu.header["CTYPE3"] = "FREQ"
    # ihdu.header["CRPIX3"] = crit_pix_f
    # ihdu.header["CDELT3"] = delta_f
    # ihdu.header["CRVAL3"] = cfreq
    # ihdu.header["CUNIT3"] = "Hz"
    # # # Coordinates - Stokes parameters
    # ihdu.header["CTYPE4"] = "STOKES"
    # ihdu.header["CRPIX4"] = 1
    # ihdu.header["CDELT4"] = -1
    # ihdu.header["CRVAL4"] = -5  # pol_nums[pol_order[0]]
    return phdu.header, ihdu.header


class EpicPixels:
    def __init__(
        self,
        img_hdr: str,
        primary_hdr: str,
        # img_array: NDArrayNum_t,
        watch_df: pd.DataFrame,
        # epic_ver: str = "0.0.1",
        # img_axes: List[int] = [1, 2],
        elevation_limit: float = 0.0,
    ) -> None:
        # self.img_array = img_array
        # self.header_str = header
        # self.epic_ver = epic_ver

        self._watch_df = watch_df
        self.img_hdr = img_hdr
        self.primary_hdr = primary_hdr

        self.ra0 = self.img_hdr["CRVAL1"]
        self.dec0 = self.img_hdr["CRVAL2"]

        self.x0 = self.img_hdr["CRPIX1"]
        self.y0 = self.img_hdr["CRPIX2"]

        self.dx = self.img_hdr["CDELT1"]
        self.dy = self.img_hdr["CDELT2"]

        self.delta_min = elevation_limit

        self.xdim = self.primary_hdr["GRIDDIMX"]
        self.ydim = self.primary_hdr["GRIDDIMY"]

        self.dgridx = self.primary_hdr["DGRIDX"]
        self.dgrixy = self.primary_hdr["DGRIDY"]

        # self.inttime = self.primary_hdr["INTTIM"]

        self.t_obs = self.img_hdr["DATETIME"]
        # print("t_obs", self.t_obs)

        self.wcs = WCS(self.img_hdr, naxis=2)

        self.max_rad = self.xdim * 0.5 * np.cos(np.deg2rad(elevation_limit))

        # self.filename = self.img_hdr["FILENAME"]

    def ra2x(
        self, ra: Union[float, NDArrayNum_t]
    ) -> Union[float, NDArrayNum_t]:
        """Return the X-pixel (0-based index) number given an RA"""
        pix = (ra - self.ra0) / self.dx + self.x0
        return self.nearest_pix(pix) - 1

    def nearest_pix(
        self, pix: Union[float, NDArrayNum_t]
    ) -> Union[float, NDArrayNum_t]:
        frac_dist = np.minimum(np.modf(pix)[0], 0.5)
        nearest_pix: Union[float, NDArrayNum_t] = np.floor(pix + frac_dist)
        return nearest_pix

    def dec2y(
        self, dec: Union[float, NDArrayNum_t]
    ) -> Union[float, NDArrayNum_t]:
        """Return the Y-pixel (0-based index) number given a DEC"""
        pix = (dec - self.dec0) / self.dy + self.y0
        return self.nearest_pix(pix) - 1

    def is_skycoord_fov(
        self,
        ra: Union[float, NDArrayNum_t, pd.Series],
        dec: Union[float, NDArrayNum_t, pd.Series],
    ) -> NDArrayBool_t:
        """Return a bool index indicating whether the
        specified sky coordinates lie inside the fov
        """
        is_fov: NDArrayBool_t = np.less_equal(
            np.linalg.norm(
                np.vstack(
                    [
                        self.ra2x(ra) - self.xdim / 2,
                        self.dec2y(dec) - self.ydim / 2,
                    ]
                ),
                axis=0,
            ),
            self.max_rad,
        )
        return is_fov

    def is_pix_fov(
        self,
        x: Union[float, NDArrayNum_t, pd.Series],
        y: Union[float, NDArrayNum_t, pd.Series],
    ) -> NDArrayBool_t:
        """Return a bool index indicating whether the
        specified pixel coordinates lie inside the fov
        """
        # print(np.vstack([x - self.xdim/2,y - self.ydim/2]))
        # print(np.linalg.norm(np.vstack([x - self.xdim/2,y - self.ydim/2])
        # ,axis=0))
        is_fov: NDArrayBool_t = np.less_equal(
            np.linalg.norm(
                np.vstack([x - self.xdim / 2, y - self.ydim / 2]), axis=0
            ),
            self.max_rad,
        )
        return is_fov

    # def header_to_metadict(self, source_names: List[str]) -> Dict[str, Any]:
    #     ihdr = self.img_hdr
    #     return dict(
    #         id=[str(uuid4())],
    #         img_time=[
    #             datetime.strptime(ihdr["DATETIME"], "%Y-%m-%dT%H:%M:%S.%f")
    #         ],
    #         n_chan=[int(ihdr["NAXIS3"])],
    #         n_pol=[int(ihdr["NAXIS4"])],
    #         chan0=[ihdr["CRVAL3"] - ihdr["CDELT3"] * ihdr["CRPIX3"]],
    #         chan_bw=[ihdr["CDELT3"]],
    #         epic_version=[self.epic_ver],
    #         img_size=[str((ihdr["NAXIS1"], ihdr["NAXIS2"]))],
    #         int_time=self.inttime,
    #         filename=self.filename,
    #         source_names=[source_names.tolist()],
    #     )

    # def store_pg(self, s_hub: ServiceHub) -> None:
    #     if self.pixel_idx_df is None or self.pixel_meta_df is None:
    #         # no sources in the fov to update
    #         return
    #     s_hub.insert_single_epoch_pgdb(self.pixel_idx_df, self.pixel_meta_df)

    def get_pix_indices(
        self,
    ) -> pd.DataFrame:
        self.idx_l = self._watch_df.index.to_numpy()
        self.src_l = self._watch_df["source_name"].to_numpy().astype(str)
        self.ra_l = self._watch_df["ra"].to_numpy()
        self.dec_l = self._watch_df["dec"].to_numpy()
        self.patch_size_l = self._watch_df["kernel_dim"].to_numpy()
        # (
        #     self._watch_df["kernel_dim"]
        #     .str.split("x")
        #     .str[0]
        #     .astype(float)
        #     .to_numpy()
        # )
        self.patch_npix_l = self.patch_size_l**2

        # print(self.ra_l, self.dec_l)
        self._update_src_skypos(self.t_obs)
        # print(self.ra_l, self.dec_l)

        self.x_l, self.y_l = self.wcs.all_world2pix(
            self.ra_l, self.dec_l, 1, ra_dec_order=True
        )  # 1-based index
        self.x_l, self.y_l = self.nearest_pix(self.x_l), self.nearest_pix(
            self.y_l
        )
        self.in_fov = np.logical_and((self.x_l >= 0), (self.y_l >= 0))

        # filter the indices or sources outside the sky
        self.watch_l = np.vstack(
            [
                self.idx_l,
                self.ra_l,
                self.dec_l,
                self.x_l,
                self.y_l,
                self.in_fov,
                self.patch_size_l,
            ]
        )

        if not self.in_fov.any():
            # no sources within the FOV
            self.pixel_idx_df = None
            self.pixel_meta_df = None
            return None
        self.watch_l = self.watch_l[:, self.in_fov]

        xpatch_pix_idx, ypatch_pix_idx = np.hstack(
            list(map(PatchMan.get_patch_idx, self.watch_l[-1, :]))
        )

        self.watch_l = np.repeat(
            self.watch_l, self.watch_l[-1, :].astype(int) ** 2, axis=1
        )

        # update the pixel indices
        self.watch_l[3, :] += xpatch_pix_idx
        self.watch_l[4, :] += ypatch_pix_idx

        # remove sources if they cross the fov
        self.watch_l[1, :], self.watch_l[2, :] = self.wcs.all_pix2world(
            self.watch_l[3, :], self.watch_l[4, :], 1
        )

        self.watch_l[-2, :] = np.logical_not(
            np.isnan(self.watch_l[1, :]) | np.isnan(self.watch_l[2, :])
        ) & self.is_pix_fov(self.watch_l[3, :], self.watch_l[4, :])

        # test fov crossing
        groups = np.split(
            self.watch_l[-2, :],
            np.unique(self.watch_l[0, :], return_index=True)[1][1:],
        )

        is_out_fov = [
            i if np.all(i) else np.logical_and(i, False) for i in groups
        ]

        # filter patches crossing fov
        is_out_fov = np.concatenate(is_out_fov).astype(bool).ravel()
        self.watch_l = self.watch_l[:, is_out_fov]
        xpatch_pix_idx = xpatch_pix_idx[is_out_fov]
        ypatch_pix_idx = ypatch_pix_idx[is_out_fov]
        src_ids = np.unique(self.watch_l[0, :]).astype(int)

        # extract the pixel values for each pixel
        # img_array indices [complex, npol, nchan, y, x]
        # pix_values = self.img_array[
        #     :,
        #     :,
        #     :,
        #     self.watch_l[4, :].astype(int)
        #     - 1,  # convert 1-based to 0-based for np indexing
        #     self.watch_l[3, :].astype(int) - 1,
        # ]
        # pix_values_l = [
        #     pix_values[:, :, :, i].ravel().tolist()
        #     for i in range(pix_values.shape[-1])
        # ]

        # skypos_pg_fmt = [
        #     f"SRID=4326;POINT({i} {j})"
        #     for i, j in zip(self.watch_l[1, :], self.watch_l[2, :])
        # ]

        # grab lm coords
        lmn_grid = get_lmn_grid(self.xdim, self.ydim)
        l_vals = lmn_grid[
            0, self.watch_l[3, :].astype(int), self.watch_l[4, :].astype(int)
        ]
        m_vals = lmn_grid[
            1, self.watch_l[3, :].astype(int), self.watch_l[4, :].astype(int)
        ]

        # lm_coord_fmt = [f"({i},{j})" for i, j in zip(l_vals, m_vals)]
        pix_x_l = self.watch_l[3, :].astype(int)
        pix_y_l = self.watch_l[4, :].astype(int)

        # pix_coord_fmt = [
        #     f"({i},{j})"
        #     for i, j in zip(
        #      self.watch_l[3, :].astype(int), self.watch_l[4, :].astype(int)
        #     )
        # ]
        # source_names = self.src_l[self.watch_l[0, :].astype(int)]
        return dict(
            nsrc=np.array([src_ids.size]).copy(),
            ncoords=np.array([l_vals.size]).copy(),
            kernel_dim=np.array([self.patch_size_l[0]]).copy(),
            src_ids=src_ids.ravel().astype(float).copy(),
            l=l_vals.ravel().astype(float).copy(),
            m=m_vals.ravel().astype(float).copy(),
            pix_x=pix_x_l.ravel().astype(float).copy(),
            pix_y=pix_y_l.ravel().astype(float).copy(),
            pix_ofst_x=xpatch_pix_idx.ravel().astype(float).copy(),
            pix_ofst_y=ypatch_pix_idx.ravel().astype(float).copy(),
        )

        # self.pixel_meta_df = pd.DataFrame.from_dict(
        #     self.header_to_metadict(source_names=np.unique(source_names))
        # )

        # self.pixel_idx_df = pd.DataFrame.from_dict(
        #     dict(
        #         id=[
        #             self.pixel_meta_df.iloc[0]["id"]
        #             for i in range(len(l_vals))
        #         ],
        #         pixel_coord=pix_coord_fmt,
        #         pixel_values=pix_values_l,
        #         pixel_skypos=skypos_pg_fmt,
        #         source_names=source_names,
        #         pixel_lm=lm_coord_fmt,
        #         pix_ofst_x=xpatch_pix_idx,
        #         pix_ofst_y=ypatch_pix_idx,
        #     )
        # )

    def _update_src_skypos(
        self,
        # source_list: pd.DataFrame,
        t_obs_str: str,
    ) -> None:
        for i, src in enumerate(self.src_l):
            if src in DynSources.bodies:
                self.ra_l[i], self.dec_l[i] = DynSources.get_lwasv_skypos(
                    src, t_obs_str
                )


def get_pixel_indices(
    seq_start_id, grid_size, grid_res, elev_limit, watchdog_endpoint
):
    watchlist = get_watch_list(watchdog_endpoint)
    phdu, ihdu = get_image_headers_sv(seq_start_id, grid_size, grid_res)

    pix_extractor = EpicPixels(ihdu, phdu, watchlist, elev_limit)
    indices = pix_extractor.get_pix_indices()
    if indices is None:  # no sources in the FOV
        return dict(
            nsrc=np.array([0]).copy(),
            ncoords=np.array([0]).copy(),
            kernel_dim=np.array([0]).copy(),
            src_ids=np.array([]).astype(float).copy(),
            l=np.array([]).astype(float).copy(),
            m=np.array([]).astype(float).copy(),
            pix_x=np.array([]).astype(float).copy(),
            pix_y=np.array([]).astype(float).copy(),
            pix_ofst_x=np.array([]).astype(float).copy(),
            pix_ofst_y=np.array([]).astype(float).copy(),
        )
    else:
        return indices


if __name__ == "__main__":
    for i in range(24):
        seq_start = i * 3600 * CHAN_BW
        grid_size = 128
        grid_res = 1
        elev_lim = 0
        kernel_dim = 5
        indices = get_pixel_indices(seq_start, grid_size, grid_res, elev_lim,"129.24.76.213:31342")
        print(indices["nsrc"], len(indices["pix_x"]))
