"""Station geometry shared by the streamcal modules (wraps src/python/epic_geometry.py)."""
import os
import sys

_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PY not in sys.path:
    sys.path.insert(0, _PY)
import epic_geometry as eg  # noqa: E402,F401

SOL = 299792458.0
# NDP F-engine channel width (FS / 8192), as constants.h BANDWIDTH
CHAN_BW = 196.0e6 / 8192
