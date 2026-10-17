from typing import Union


import numpy as np
import pandas as pd

from astropy.wcs import WCS

from pixel_extract_utils import DynSources
from pixel_extract_utils import PatchMan
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
import io
import json
import os
import sys
import time


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


WATCH_COLUMNS = ["id", "source_name", "ra", "dec", "kernel_dim"]

# The watch list is global: it lives in the head node's database and is served
# by the epic-watchdog service. The imager must go on when either is down, so
#   1. it asks the watchdog (gRPC);
#   2. failing that, it reads the table directly if EPIC_WATCH_DSN (a libpq
#      connection string for the head node's database) is set;
#   3. failing that, it uses the last list it obtained, kept in memory and in
#      the file EPIC_WATCHLIST_CACHE (if set) so that it survives a restart;
#   4. with none of these, it extracts no pixels until a list is available.
# After a failure the watchdog is not asked again for WATCH_RETRY_S, so that a
# dead service does not cost a time-out at every refresh.
WATCH_RETRY_S = 60.0
WATCH_SQL = """
SELECT id, source AS source_name, ra_deg AS ra, dec_deg AS dec,
       patch_type AS kernel_dim
FROM public.epic_watchdog
WHERE watch_status = 'watching'
  AND t_start < (now() AT TIME ZONE 'UTC')
  AND (t_end IS NULL OR t_end > (now() AT TIME ZONE 'UTC'))
ORDER BY id
"""

_watch = {"df": None, "json": None, "source": None, "time": 0.0,
          "grpc_retry_at": 0.0, "log_at": {}}


def _log_limited(key, msg, every=300.0):
    """Print msg (to the imager's log) at most once per `every` seconds per key."""
    now = time.monotonic()
    if now - _watch["log_at"].get(key, -1e9) >= every:
        _watch["log_at"][key] = now
        print(f"pixel_extractor: {msg}", file=sys.stderr, flush=True)


def _empty_watch_list():
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        WATCH_COLUMNS, ["int64", "str", "float64", "float64", "int64"])})


def _normalise(df):
    df = df.rename(columns=dict(patch_type="kernel_dim"))
    missing = [c for c in WATCH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"watch list without the columns {missing}")
    return df[WATCH_COLUMNS].reset_index(drop=True)


def _fetch_watchdog(watchdog_endpoint):
    endpoint = (
        watchdog_endpoint
        if watchdog_endpoint is not None
        else get_endpoint("epic-watchdog.service.consul")
    )
    with grpc.insecure_channel(endpoint) as channel:
        stub = epic_image_pb2_grpc.epic_post_processStub(channel)
        response = stub.fetch_watchlist(epic_image_pb2.empty(), timeout=5)
        df_json = json.loads(response.pd_json)
        return _normalise(pd.read_json(io.StringIO(df_json)))


def _fetch_database(dsn):
    import psycopg2

    conn = psycopg2.connect(dsn, connect_timeout=5,
                            options="-c statement_timeout=5000",
                            application_name="epic-imager-watchlist")
    try:
        with conn.cursor() as cur:
            cur.execute(WATCH_SQL)
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=WATCH_COLUMNS)
    return _normalise(df.astype({"ra": float, "dec": float}))


def _cache_path():
    return os.environ.get("EPIC_WATCHLIST_CACHE", "")


def _remember(df, source):
    _watch.update(df=df, source=source, time=time.time())
    path = _cache_path()
    if not path:
        return
    js = df.to_json(orient="records")
    if js == _watch["json"]:
        return
    try:
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"saved": time.time(), "source": source,
                       "rows": json.loads(js)}, f)
        os.replace(tmp, path)
        _watch["json"] = js
    except OSError as e:
        _log_limited("cache-write", f"cannot save the watch list to {path}: {e}")


def _cached():
    if _watch["df"] is not None:
        return _watch["df"], _watch["time"]
    path = _cache_path()
    if not path or not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            saved = json.load(f)
        df = _normalise(pd.DataFrame(saved["rows"], columns=WATCH_COLUMNS))
        _watch.update(df=df, time=saved["saved"], source="cache file")
        return df, saved["saved"]
    except (OSError, ValueError, KeyError) as e:
        _log_limited("cache-read", f"cannot read the cached watch list {path}: {e}")
        return None, None


def get_watch_list(watchdog_endpoint=None):
    """
    Fetch the watchlist for EPIC at Sevilleta. Never raises: see the notes
    above WATCH_SQL for the fallbacks. Columns: WATCH_COLUMNS.
    """
    errors = []
    if time.monotonic() >= _watch["grpc_retry_at"]:
        try:
            df = _fetch_watchdog(watchdog_endpoint)
            _remember(df, "watchdog")
            return df
        except Exception as e:  # grpc, DNS, decoding: all mean "not now"
            _watch["grpc_retry_at"] = time.monotonic() + WATCH_RETRY_S
            errors.append(f"watchdog: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
    else:
        errors.append("watchdog: skipped after a recent failure")

    dsn = os.environ.get("EPIC_WATCH_DSN", "")
    if dsn:
        try:
            df = _fetch_database(dsn)
            _remember(df, "head database")
            _log_limited("fallback-db", "watch list read from the head node's database ("
                         + "; ".join(errors) + ")")
            return df
        except Exception as e:
            errors.append(f"head database: {str(e).strip().splitlines()[0] if str(e) else type(e).__name__}")

    df, saved = _cached()
    if df is not None:
        age_min = (time.time() - saved) / 60.0
        _log_limited("fallback-cache", f"using the watch list from {age_min:.0f} min ago "
                     f"({len(df)} sources; " + "; ".join(errors) + ")")
        return df

    _log_limited("no-list", "no watch list available, extracting no pixels ("
                 + "; ".join(errors) + ")")
    return _empty_watch_list()


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
        # exact (fractional) source positions, kept so that pixel values can be
        # interpolated at the source instead of read from the nearest pixel
        self.xf_l, self.yf_l = self.x_l.copy(), self.y_l.copy()
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
        src_xf, src_yf = self.xf_l[self.in_fov], self.yf_l[self.in_fov]

        xpatch_pix_idx, ypatch_pix_idx = np.hstack(
            list(map(PatchMan.get_patch_idx, self.watch_l[-1, :]))
        )

        npix = self.watch_l[-1, :].astype(int) ** 2
        self.watch_l = np.repeat(self.watch_l, npix, axis=1)
        src_xf, src_yf = np.repeat(src_xf, npix), np.repeat(src_yf, npix)

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
        src_xf, src_yf = src_xf[is_out_fov], src_yf[is_out_fov]
        src_ids = np.unique(self.watch_l[0, :]).astype(int)
        src_names_arr = self._watch_df.iloc[src_ids]["source_name"].tolist()
        src_names_arr = '{"' + '","'.join(src_names_arr) + '"}'

        # direction cosines of each pixel in the image frame (east is -x).
        # A pixel spans 1 / (N * grid spacing) = 2 sin(res / 2) in l and m.
        l_vals = -(self.watch_l[3, :] - self.x0) / (self.xdim * self.dgridx)
        m_vals = (self.watch_l[4, :] - self.y0) / (self.ydim * self.dgrixy)

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
            src_x=src_xf.ravel().astype(float).copy(),
            src_y=src_yf.ravel().astype(float).copy(),
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


def _no_sources():
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


def get_pixel_indices(
    seq_start_id, grid_size, grid_res, elev_limit, watchdog_endpoint
):
    watchlist = get_watch_list(watchdog_endpoint)
    if len(watchlist.index) == 0:
        return _no_sources()
    phdu, ihdu = get_image_headers_sv(seq_start_id, grid_size, grid_res)

    pix_extractor = EpicPixels(ihdu, phdu, watchlist, elev_limit)
    indices = pix_extractor.get_pix_indices()
    if indices is None:  # no sources in the FOV
        return _no_sources()
    else:
        return indices


def get_pixel_indices2(ihdu, phdu, elev_limit=10):
    watchlist = get_watch_list()
    if len(watchlist.index) == 0:
        return _no_sources()
    pix_extractor = EpicPixels(ihdu, phdu, watchlist, elev_limit)
    indices = pix_extractor.get_pix_indices()
    if indices is None:  # no sources in the FOV
        return _no_sources()
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
