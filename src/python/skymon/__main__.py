"""
EPIC all-sky monitor.

  python -m skymon run --shm epic_sky0 --out /path/to/output --port 8095
  python -m skymon simulate --shm skysim --duration 1800            (images only)
  python -m skymon simtest --duration 1800 --realtime --out DIR     (simulator + monitor)
  python -m skymon evaluate --truth DIR/truth.jsonl --out DIR
  python -m skymon build                                           (compile the kernels)

Run with src/python on PYTHONPATH (the imager's geometry module lives there).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time

from .monitor import Config, Monitor


def add_monitor_args(p):
    d = Config()
    p.add_argument("--out", default=d.out_dir, help="output directory (candidates, injections)")
    p.add_argument("--port", type=int, default=8095, help="HTTP port, 0 to disable")
    p.add_argument("--nside", type=int, default=0, help="HEALPix nside (0: from the beam)")
    p.add_argument("--el-min", type=float, default=d.el_min)
    p.add_argument("--dm-max", type=float, default=d.dm_max)
    p.add_argument("--threshold", type=float, default=d.threshold)
    p.add_argument("--threads", type=int, default=d.nthreads)
    p.add_argument("--block", type=int, default=d.block0, help="tier-0 samples per search")
    p.add_argument("--accum-interval", type=float, default=d.accum_interval_s)
    p.add_argument("--accum-hours", type=float, default=d.accum_hours)
    p.add_argument("--inject-every", type=float, default=d.inject_interval_s / 60,
                   help="mean minutes between live test injections (0 = off)")
    p.add_argument("--inject-snr", type=float, nargs=2, default=d.inject_snr)
    p.add_argument("--inject-dm", type=float, nargs=2, default=d.inject_dm)
    p.add_argument("--no-rfi-records", action="store_true", help="do not store RFI candidates")
    p.add_argument("--seed", type=int, default=None)


def make_config(a) -> Config:
    return Config(out_dir=a.out, nside=a.nside, el_min=a.el_min, dm_max=a.dm_max,
                  threshold=a.threshold, nthreads=a.threads, block0=a.block,
                  accum_interval_s=a.accum_interval, accum_hours=a.accum_hours,
                  inject_interval_s=a.inject_every * 60, inject_snr=tuple(a.inject_snr),
                  inject_dm=tuple(a.inject_dm), store_rfi=not a.no_rfi_records, seed=a.seed)


def run_ring(monitor: Monitor, ring: str, stop: threading.Event, max_queue: int = 512,
             status_every: float = 60.0, until=None):
    """Follow a shared-memory ring until stop is set (or until() is true)"""
    from .shm import RingReader

    rd = RingReader(ring)
    q: queue.Queue = queue.Queue()
    dropped = [0]
    done = threading.Event()  # stops the reader before the ring is closed

    def reader():
        idle = 0.0
        while not (stop.is_set() or done.is_set()):
            try:
                item = rd.read()
            except ValueError as e:
                print(f"skymon: {e}", flush=True)
                time.sleep(5)
                continue
            if item is None:
                time.sleep(0.005)
                idle += 0.005
                continue
            idle = 0.0
            if q.qsize() >= max_queue:  # falling behind: drop the oldest, keep running
                try:
                    q.get_nowait()
                    dropped[0] += 1
                except queue.Empty:
                    pass
            q.put(item)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    last = time.time()
    max_q = 0
    while not stop.is_set():
        try:
            meta, img = q.get(timeout=0.5)
        except queue.Empty:
            if until and until():
                break
            continue
        max_q = max(max_q, q.qsize())
        monitor.process(meta, img)
        monitor.metrics["ring_dropped"] = rd.dropped
        monitor.metrics["queue_dropped"] = dropped[0]
        monitor.metrics["queue_max"] = max_q
        monitor.metrics["queue"] = q.qsize()
        if time.time() - last > status_every:
            last = time.time()
            s = monitor.status()
            print(f"skymon: {s['images']} images, lag {s['lag_s'] or 0:.2f} s, "
                  f"cpu {100 * (s['cpu_fraction'] or 0):.0f}% of real time, queue {q.qsize()} "
                  f"(max {max_q}), ring drops {rd.dropped}, queue drops {dropped[0]}, "
                  f"tracked {s['tracked']}, stored {s['candidates']['stored']}", flush=True)
    done.set()
    th.join(5)
    rd.close()
    return dict(ring_dropped=rd.dropped, queue_dropped=dropped[0], queue_max=max_q)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="skymon", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="monitor the images of the local imager")
    p.add_argument("--shm", default="epic_sky0", help="shared-memory ring name (/dev/shm)")
    add_monitor_args(p)

    p = sub.add_parser("simulate", help="write simulated images to a ring")
    p.add_argument("--shm", default="skysim")
    p.add_argument("--duration", type=float, default=3600)
    p.add_argument("--fast", action="store_true", help="do not pace at the image rate")
    add_sim_args(p)

    p = sub.add_parser("simtest", help="simulator and monitor together, then evaluate")
    p.add_argument("--duration", type=float, default=1800)
    p.add_argument("--realtime", action="store_true",
                   help="run the simulator as a separate process at the image rate, "
                        "through a shared-memory ring")
    add_monitor_args(p)
    add_sim_args(p)

    p = sub.add_parser("evaluate", help="match candidates with a truth file")
    p.add_argument("--truth", required=True)
    p.add_argument("--out", required=True)

    sub.add_parser("build", help="compile the kernels")
    a = ap.parse_args(argv)

    if a.cmd == "build":
        from . import kernels
        print(kernels.build(force=True))
        return 0
    if a.cmd == "evaluate":
        from .evaluate import report
        print(report(a.truth, a.out)["text"])
        return 0
    if a.cmd == "simulate":
        from .simulate import run_shm
        sim = make_sim(a)
        print(json.dumps(run_shm(sim, a.shm, a.duration, realtime=not a.fast)))
        return 0
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "simtest":
        return cmd_simtest(a)


def add_sim_args(p):
    p.add_argument("--chan0", type=int, default=2400 - 64,
                   help="first NDP channel (2336: 55.87-58.94 MHz)")
    p.add_argument("--nchan", type=int, default=128)
    p.add_argument("--frb-per-hour", type=float, default=30)
    p.add_argument("--frb-snr", type=float, nargs=2, default=(8, 30))
    p.add_argument("--frb-dm", type=float, nargs=2, default=(20, 3000))
    p.add_argument("--rfi-per-hour", type=float, default=6)
    p.add_argument("--start", type=float, default=None, help="simulated start (unix)")
    p.add_argument("--sim-seed", type=int, default=1)
    p.add_argument("--truth", default=None)


def make_sim(a, truth=None):
    from .geometry import Band
    from .simulate import SkySimulator, tier_dt_function
    band = Band(a.chan0, a.nchan)
    dt = 2000 * 8192 / 196e6
    return SkySimulator(chan0=a.chan0, nchan=a.nchan, start_unix=a.start,
                        frb_per_hour=a.frb_per_hour, snr=tuple(a.frb_snr), dm=tuple(a.frb_dm),
                        rfi_per_hour=a.rfi_per_hour, truth_path=truth or a.truth,
                        seed=a.sim_seed,
                        tier_dt=tier_dt_function(band.edges_mhz(), dt, a.nchan))


def _serve(monitor, port):
    if port:
        from .server import serve
        serve(monitor, port=port)
        print(f"skymon: HTTP on port {port}", flush=True)


def cmd_run(a):
    monitor = Monitor(make_config(a))
    _serve(monitor, a.port)
    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.set())
    print(f"skymon: following /dev/shm/{a.shm}", flush=True)
    run_ring(monitor, a.shm, stop)
    monitor.store.flush()
    return 0


def cmd_simtest(a):
    import subprocess

    os.makedirs(a.out, exist_ok=True)
    truth = os.path.join(a.out, "truth.jsonl")
    monitor = Monitor(make_config(a))
    _serve(monitor, a.port)
    t_start = time.time()
    if a.realtime:
        ring = f"skysim_{os.getpid()}"
        start = a.start or time.time()
        cmd = [sys.executable, "-m", "skymon", "simulate", "--shm", ring,
               "--duration", str(a.duration), "--chan0", str(a.chan0), "--nchan", str(a.nchan),
               "--frb-per-hour", str(a.frb_per_hour), "--frb-snr", *map(str, a.frb_snr),
               "--frb-dm", *map(str, a.frb_dm), "--rfi-per-hour", str(a.rfi_per_hour),
               "--start", str(start), "--sim-seed", str(a.sim_seed), "--truth", truth]
        env = dict(os.environ)
        sim = subprocess.Popen(cmd, env=env)
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        info = run_ring(monitor, ring, stop, until=lambda: sim.poll() is not None)
        sim.wait()
        try:
            os.unlink(f"/dev/shm/{ring}")
        except FileNotFoundError:
            pass
        t_end_data = start + a.duration
        t_start_data = start
    else:
        sim = make_sim(a, truth)
        n = int(a.duration / sim.dt)
        info = {}
        for i in range(n):
            meta, img = sim.next_image()
            monitor.process(meta, img)
            if i % int(300 / sim.dt) == 0:
                s = monitor.status()
                print(f"simtest: {i}/{n} images ({i * sim.dt / 60:.0f} min), "
                      f"cpu {100 * (s['cpu_fraction'] or 0):.0f}% of real time, "
                      f"{len(sim.truth)} FRBs, stored {s['candidates']['stored']}", flush=True)
        t_end_data = sim._t(n)
        t_start_data = sim._t(0)
    monitor.store.flush()
    wall = time.time() - t_start
    st = monitor.status()
    from .evaluate import ready_after, report
    tiers = monitor.searcher.describe()
    rep = report(truth, a.out, t_end=t_end_data,
                 latency_s=max(t.block * t.dt for t in monitor.searcher.tiers),
                 ready=ready_after(tiers, t_start_data))
    summary = dict(wall_s=wall, data_s=a.duration, start_unix=t_start_data,
                   end_unix=t_end_data, tiers=tiers, images=st["images"],
                   missing_images=st["missing_images"],
                   cpu_fraction_of_realtime=st["cpu_fraction"], **info,
                   search_s=monitor.searcher.stats["search_s"],
                   ingest_s=monitor.searcher.stats["ingest_s"],
                   tracked=st["tracked"], memory_gib=st.get("memory_gib"),
                   candidates=st["candidates"])
    print(json.dumps(summary, indent=1))
    print(rep["text"])
    with open(os.path.join(a.out, "simtest_summary.json"), "w") as f:
        json.dump(dict(summary=summary, report=rep["text"]), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
