from typing import Union


import numpy as np
import pandas as pd

from astropy.wcs import WCS

from pixel_extract_utils import DynSources
from pixel_extract_utils import PatchMan
from pixel_extract_utils import get_lmn_grid
from epic_grpc import epic_image_pb2_grpc
from epic_grpc import epic_image_pb2

from epic_types import NDArrayBool_t
from epic_types import NDArrayNum_t

from adp import get_ADP_time_from_unix_epoch
from lsl.common.stations import lwasv as lwasv_station

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5
from epic_geometry import apply_wcs

import dns.resolver


import grpc
import json


FS = 196.0e6
SEQ_MULT_S = 8192  # FS samples per NDP spectrum
CHAN_BW = FS / SEQ_MULT_S  # NDP channel width


def _consul_resolvers():
    # The host resolver, then the local Consul agent, in case the host's DNS
    # does not forward the .consul domain
    yield dns.resolver.get_default_resolver()
    agent = dns.resolver.Resolver(configure=False)
    agent.nameservers = ["127.0.0.1"]
    agent.port = 8600
    yield agent


def get_endpoint(service="epic-watchdog.service.consul"):
    errors = []
    for resolver in _consul_resolvers():
        try:
            rdata = resolver.resolve(service, "SRV")[0]
            target = rdata.target.to_text()
            ip_address = resolver.resolve(target, "A")[0].to_text()
            return f"{ip_address}:{rdata.port}"
        except Exception as e:
            errors.append(f"{resolver.nameservers}: {e}")
    raise RuntimeError(f"Cannot resolve {service}: {'; '.join(errors)}")


def get_watch_list(watchdog_endpoint=None):
    """
    Fetch the watchlist for EPIC at Sevilleta
    """
    endpoint = (
        watchdog_endpoint
        if watchdog_endpoint is not None
        else get_endpoint("epic-watchdog.service.consul")
    )
    with grpc.insecure_channel(endpoint) as channel:
        stub = epic_image_pb2_grpc.epic_post_processStub(channel)
        response = stub.fetch_watchlist(epic_image_pb2.empty(), timeout=5)
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
    # NDP sequences count from the Unix epoch
    time_tag = seq_start_id * SEQ_MULT_S / FS
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

    ihdu = fits.ImageHDU(np.zeros((grid_size, grid_size)))

    ihdu.header["DATETIME"] = t0.isot
    ihdu.header["LST"] = lsts.hour
    # Same celestial WCS as the saved images (phase centre via AltAz, PC rotation)
    apply_wcs(ihdu.header, t0, grid_size, grid_res)

    return phdu.header, ihdu.header


class EpicPixels:
    def __init__(
        self,
        img_hdr: str,
        primary_hdr: str,
        watch_df: pd.DataFrame,
        elevation_limit: float = 0.0,
    ) -> None:
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

        self.t_obs = self.img_hdr["DATETIME"]

        self.wcs = WCS(self.img_hdr, naxis=2)

        self.max_rad = self.xdim * 0.5 * np.cos(np.deg2rad(elevation_limit))

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
        is_fov: NDArrayBool_t = np.less_equal(
            np.linalg.norm(
                np.vstack([x - self.xdim / 2, y - self.ydim / 2]), axis=0
            ),
            self.max_rad,
        )
        return is_fov

    def get_pix_indices(
        self,
    ) -> pd.DataFrame:
        self.idx_l = (
            self._watch_df.reset_index().index.to_numpy()
        )  # the fetched indices correspond to orginal ids in the watchlist
        self.src_l = self._watch_df["source_name"].to_numpy().astype(str)
        self.ra_l = self._watch_df["ra"].to_numpy()
        self.dec_l = self._watch_df["dec"].to_numpy()
        self.patch_size_l = self._watch_df["kernel_dim"].to_numpy()
        self.patch_npix_l = self.patch_size_l**2

        self._update_src_skypos(self.t_obs)

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
        src_names_arr = self._watch_df.iloc[src_ids]["source_name"].tolist()
        src_names_arr = '{"' + '","'.join(src_names_arr) + '"}'

        # grab lm coords
        lmn_grid = get_lmn_grid(self.xdim, self.ydim)
        l_vals = lmn_grid[
            0, self.watch_l[3, :].astype(int), self.watch_l[4, :].astype(int)
        ]
        m_vals = lmn_grid[
            1, self.watch_l[3, :].astype(int), self.watch_l[4, :].astype(int)
        ]

        pix_x_l = self.watch_l[3, :].astype(int)
        pix_y_l = self.watch_l[4, :].astype(int)

        source_names = self.src_l[self.watch_l[0, :].astype(int)]
        return dict(
            nsrc=np.array([src_ids.size]).copy(),
            ncoords=np.array([l_vals.size]).copy(),
            kernel_dim=np.array([self.patch_size_l[0]]).copy(),
            # src_ids=src_ids.ravel().astype(float).copy(),
            src_ids=source_names.ravel().astype(str).copy().tolist().copy(),
            l=l_vals.ravel().astype(float).copy(),
            m=m_vals.ravel().astype(float).copy(),
            pix_x=pix_x_l.ravel().astype(float).copy(),
            pix_y=pix_y_l.ravel().astype(float).copy(),
            pix_ofst_x=xpatch_pix_idx.ravel().astype(float).copy(),
            pix_ofst_y=ypatch_pix_idx.ravel().astype(float).copy(),
            src_names_arr=src_names_arr,
        )

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
            src_ids=np.array([]).astype(str).copy(),
            l=np.array([]).astype(float).copy(),
            m=np.array([]).astype(float).copy(),
            pix_x=np.array([]).astype(float).copy(),
            pix_y=np.array([]).astype(float).copy(),
            pix_ofst_x=np.array([]).astype(float).copy(),
            pix_ofst_y=np.array([]).astype(float).copy(),
            src_names=[],
        )
    else:
        return indices


def get_pixel_indices2(ihdu, phdu, elev_limit=10):
    watchlist = get_watch_list()
    pix_extractor = EpicPixels(ihdu, phdu, watchlist, elev_limit)
    indices = pix_extractor.get_pix_indices()
    if indices is None:  # no sources in the FOV
        return dict(
            nsrc=np.array([0]).copy(),
            ncoords=np.array([0]).copy(),
            kernel_dim=np.array([0]).copy(),
            src_ids=np.array([]).astype(str).copy(),
            l=np.array([]).astype(float).copy(),
            m=np.array([]).astype(float).copy(),
            pix_x=np.array([]).astype(float).copy(),
            pix_y=np.array([]).astype(float).copy(),
            pix_ofst_x=np.array([]).astype(float).copy(),
            pix_ofst_y=np.array([]).astype(float).copy(),
            src_names=[],
        )
    else:
        return indices


if __name__ == "__main__":
    for i in range(100):
        seq_start = i * 3600 * CHAN_BW
        grid_size = 128
        grid_res = 1
        elev_lim = 0
        kernel_dim = 5
        indices = get_pixel_indices(
            seq_start, grid_size, grid_res, elev_lim, "169.254.128.1:1729"
        )
        print(
            i,
            indices["nsrc"],
            len(indices["pix_x"]),
            indices["src_ids"],
            indices["src_names_arr"],
        )
        #
