"""Replay recorded GPU-tap tables (epic++ --caltap_record_dir) through the streamcal daemon.

    python replay_tables.py TABLE_DIR WORKDIR [--n 20] [daemon options...]

Each vis_<version>.bin is written in place to WORKDIR/vis, the file the daemon polls, and one
daemon step is run on it; gains go to WORKDIR/gains (never the imager's table). Logs the daemon
summary per table.
"""
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from streamcal import daemon as D  # noqa: E402


def main():
    tables, work = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    n = 20
    if "--n" in rest:
        i = rest.index("--n"); n = int(rest[i + 1]); rest = rest[:i] + rest[i + 2:]
    os.makedirs(work, exist_ok=True)
    vis, gains = os.path.join(work, "vis"), os.path.join(work, "gains")
    for p in (vis, gains):
        if os.path.exists(p):
            os.remove(p)
    files = sorted(glob.glob(os.path.join(tables, "vis_*.bin")))[:n]
    with open(files[0], "rb") as f, open(vis, "wb") as o:
        o.write(f.read())
    a = D.build_parser().parse_args(["--vis", vis, "--gains", gains] + rest) if hasattr(D, "build_parser") else None
    if a is None:
        raise SystemExit("daemon has no build_parser()")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sc = D.StreamCal(a)
    for k, fn in enumerate(files):
        if k > 0:
            with open(fn, "rb") as f, open(vis, "r+b") as o:
                o.write(f.read())
        sc.step()


if __name__ == "__main__":
    main()
