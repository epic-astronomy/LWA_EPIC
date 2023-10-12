import socket
from functools import lru_cache
from typing import List
from typing import Optional
from typing import Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.coordinates import get_body
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time
from numpy.typing import NDArray

from epic_types import NDArrayNum_t
from epic_types import Numeric_t
from epic_types import Patch_t
from epic_types import PixCoord2d_t


class PatchMan:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_patch_indices(
        patch_type: Patch_t = "3x3",
    ) -> List[NDArray[np.int64]]:
        patch: int = PatchMan.get_patch_size(patch_type)

        return np.meshgrid(
            np.arange(patch) - int(patch / 2),
            np.arange(patch) - int(patch / 2),
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def get_patch_idx(patch_size: int = 3) -> NDArrayNum_t:
        x, y = np.meshgrid(
            np.arange(patch_size) - int(patch_size / 2),
            np.arange(patch_size) - int(patch_size / 2),
        )
        return np.vstack([x.ravel(), y.ravel()])

    @staticmethod
    @lru_cache(maxsize=None)
    def get_patch_size(patch_type: Patch_t = "3x3") -> int:
        return int(str(patch_type).split("x")[0])

    @staticmethod
    def get_patch_pixels(
        pixel: Optional[PixCoord2d_t] = None,
        x: Optional[Numeric_t] = None,
        y: Optional[Numeric_t] = None,
        patch_type: Patch_t = "3x3",
    ) -> Tuple[NDArrayNum_t, NDArrayNum_t]:
        xgrid, ygrid = PatchMan.get_patch_indices(patch_type)

        if pixel is not None:
            return (xgrid + pixel[0]).flatten(), (ygrid + pixel[1]).flatten()
        elif x is not None and y is not None:
            return (xgrid + x).flatten(), (ygrid + y).flatten()
        else:
            raise Exception("Either pixel or x and y must be specified")


@lru_cache
def get_lmn_grid(xsize: int, ysize: int) -> NDArrayNum_t:
    lm_matrix = np.zeros(shape=(xsize, ysize, 3))
    m_step = 2.0 / ysize
    l_step = 2.0 / xsize
    i, j = np.meshgrid(np.arange(xsize), np.arange(ysize))
    # this builds a 3 x 64 x 64 matrix, need to transpose axes to [2, 1, 0]
    #  to getcorrect 64 x 64 x 3 shape
    lm_matrix = np.asarray(
        [i * l_step - 1.0, j * m_step - 1.0, np.zeros_like(j)]
    )

    return lm_matrix


class DynSources:
    lwasv_loc = EarthLocation(lat=34.348361 * u.deg, lon=-106.885778 * u.deg)
    # EarthLocation(lat=34.348333 * u.deg, lon=-105.114422 * u.deg)
    bodies = solar_system_ephemeris.bodies

    @staticmethod
    def get_lwasv_skypos(body: str, t_obs_str: str) -> List[float]:
        time = Time(t_obs_str, format="isot", scale="utc")
        loc = get_body(body, time, DynSources.lwasv_loc)
        return [loc.ra.value, loc.dec.value]


# for body in solar_system_ephemeris.bodies:
#     get_skypos_func = partial(DynSources._get_lwasv_skypos, body)
#     setattr(
#         DynSources,
#         f"get_skypos_{body}",
#         staticmethod(get_skypos_func),
#     )


def get_epic_stpro_uds_id() -> str:
    """
    Query the central registry to get the UDS ID
    """
    # querying logic
    # return "localhost:8005"
    return f"unix-abstract:{socket.gethostname()}_epic_processor"


def get_thread_UDS_addr() -> str:
    return f"\0{socket.gethostname()}_epic_processor"


# class DotDict(dict):
#     """dot.notation access to dictionary attributes."""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__
