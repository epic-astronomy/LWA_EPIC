"""
EPIC all-sky monitor.

  python -m skymon run --shm epic_sky0 --out /path/to/output --port 8095
  python -m skymon simulate --shm skysim --duration 1800            (images only)
  python -m skymon simtest --duration 1800 --realtime --out DIR     (simulator + monitor)
  python -m skymon evaluate --truth DIR/truth.jsonl --out DIR
  python -m skymon build                                           (compile the kernels)
  python -m skymon combine --nodes epic1=http://epic1:8095,... --out DIR   (consensus across bands)
  python -m skymon frblist --out frbs.json --start T --duration S    (FRBs shared by several bands)

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
    p.add_argument("--out", default=d.out_dir,
                   help="output directory on local disk (candidates, spool, injections)")
    p.add_argument("--db", default=os.environ.get("SKYMON_DB"),
                   help="the node's local PostgreSQL, e.g. 'host=/var/run/skymon port=5433 "
                        "dbname=skymon' (default $SKYMON_DB; none: files only)")
    p.add_argument("--port", type=int, default=8095, help="HTTP port, 0 to disable")
    p.add_argument("--nside", type=int, default=0, help="HEALPix nside (0: from the beam)")
    p.add_argument("--el-min", type=float, default=d.el_min)
    p.add_argument("--dm-max", type=float, default=d.dm_max)
    p.add_argument("--threshold", type=float, default=d.threshold,
                   help="S/N threshold of the reference search (boxcars 1, 2, 4, 8); wider "
                        "boxcars raise the threshold applied for their extra trials, at the same "
                        "Gaussian false-alarm rate, unless --no-far-match")
    p.add_argument("--no-far-match", action="store_true",
                   help="apply --threshold unchanged whatever the boxcar widths")
    p.add_argument("--widths", default=None,
                   help="boxcar widths in tier samples: '1,2,4,8' for every tier, or per tier "
                        "range, e.g. '0-4:1,2,4,8,16,32,64;5-:1,2,4,8' (default "
                        f"{d.widths if isinstance(d.widths, str) else ','.join(map(str, d.widths))})")
    p.add_argument("--hp-tau", default=None,
                   help=f"tier high-pass time constant in tier samples, a number or a per-tier "
                        f"spec (default {d.hp_tau}); at least --hp-factor times the widest boxcar")
    p.add_argument("--hp-factor", type=float, default=d.hp_factor)
    p.add_argument("--detrend-len", default=None,
                   help="tier history (span of the trend fit) of at least this many tier samples, "
                        "a number or a per-tier spec (default: the transform's window)")
    p.add_argument("--detrend-factor", type=float, default=d.detrend_factor,
                   help="tier history of at least this many times the widest boxcar")
    p.add_argument("--detrend", choices=("linear", "mean"), default=d.detrend)
    p.add_argument("--boxcar-stride", type=int, default=d.boxcar_stride_div,
                   help="evaluate a boxcar of w samples every w / N samples (a power of two; "
                        "0: every sample); the S/N loss is 1/(4N) on average")
    p.add_argument("--noise-every", type=int, default=d.noise_every,
                   help="measure the search noise in one block of this many per tier")
    p.add_argument("--history-dtype", choices=("float16", "float32"), default=d.history_dtype,
                   help="precision of the tiers' histories (float16: half the memory)")
    p.add_argument("--threads", type=int, default=d.nthreads)
    p.add_argument("--block", type=int, default=d.block0, help="tier-0 samples per search")
    p.add_argument("--max-delay-factor", type=float, default=d.delay_factor,
                   help="FDMT delay rows per tier as a multiple of the channels (2: each tier "
                        "reaches twice the DM, so bursts are searched at a finer time resolution "
                        "against their smearing, +18-33%% S/N above tier 0, about 1.45x the "
                        "search time with --block 512)")
    p.add_argument("--accum-interval", type=float, default=d.accum_interval_s)
    p.add_argument("--accum-hours", type=float, default=d.accum_hours)
    p.add_argument("--inject-every", type=float, default=d.inject_interval_s / 60,
                   help="mean minutes between live test injections (0 = off)")
    p.add_argument("--inject-snr", type=float, nargs=2, default=d.inject_snr)
    p.add_argument("--inject-dm", type=float, nargs=2, default=d.inject_dm)
    p.add_argument("--inject-scatter", action=argparse.BooleanOptionalAction,
                   default=d.inject_scatter,
                   help="scatter live injections: tau at 600 MHz log-uniform over "
                        "--inject-tau600-ms, scaled as nu^-4")
    p.add_argument("--inject-tau600-ms", type=float, nargs=2, default=d.inject_tau600_ms)
    p.add_argument("--inject-unscattered", type=float, default=d.inject_unscattered_fraction,
                   help="fraction of live injections left unscattered (short-pulse check)")
    p.add_argument("--clip", type=float, default=d.clip,
                   help="normalised cells beyond this are set to zero")
    p.add_argument("--chan-flag-sigma", type=float, default=d.chan_flag_sigma,
                   help="spread over the sky above which a channel is flagged in an image "
                        "(noise gives about 1; inf disables)")
    p.add_argument("--no-rfi-records", action="store_true", help="do not store RFI candidates")
    p.add_argument("--rfi-records-max", type=int, default=20,
                   help="records kept per search block for certain RFI, brightest first; the rest are counted")
    p.add_argument("--fixed-dir-repeats", type=int, default=2,
                   help="earlier detections at a candidate's azimuth and elevation, with the sky elsewhere, "
                        "that make it RFI (data kept)")
    p.add_argument("--fixed-dir-hours", type=float, default=6.0,
                   help="how long detections are remembered by their direction above the horizon")
    p.add_argument("--sidelobe-max", type=float, default=0.03,
                   help="a burst at most this fraction of the amplitude of one elsewhere at the same time "
                        "is its sidelobe (RFI, data kept)")
    p.add_argument("--narrowband-min-channels", type=float, default=d.narrowband_min_channels,
                   help="fewer effective channels than this is narrowband RFI")
    p.add_argument("--known-pulsars", default=None,
                   help="CSV name,ra_deg,dec_deg,dm,period_s of pulsars added to the built-in list")
    p.add_argument("--disturbed-rfi-rate", type=float, default=d.cond_rfi_rate,
                   help="RFI candidates per minute above which a period is marked disturbed")
    p.add_argument("--calinfo", default=None,
                   help="streamcal's calibration record, e.g. /dev/shm/epic_calinfo_gpu0.json: "
                        "fluence and flux density of candidates (none: no flux)")
    p.add_argument("--beam-model", default="empirical",
                   help="lsl dipole model of the primary beam for fluences ('record': the beam "
                        "table of the calibration record, as the simulator writes)")
    p.add_argument("--seed", type=int, default=None)


def make_config(a) -> Config:
    return Config(out_dir=a.out, db_dsn=a.db, nside=a.nside, el_min=a.el_min, dm_max=a.dm_max,
                  threshold=a.threshold, nthreads=a.threads, block0=a.block,
                  delay_factor=a.max_delay_factor,
                  accum_interval_s=a.accum_interval, accum_hours=a.accum_hours,
                  inject_interval_s=a.inject_every * 60, inject_snr=tuple(a.inject_snr),
                  inject_dm=tuple(a.inject_dm), store_rfi=not a.no_rfi_records, rfi_records_max=a.rfi_records_max,
                  sidelobe_max=a.sidelobe_max,
                  fixed_dir_repeats=a.fixed_dir_repeats, fixed_dir_hours=a.fixed_dir_hours, seed=a.seed,
                  narrowband_min_channels=a.narrowband_min_channels,
                  known_pulsars_csv=a.known_pulsars,
                  clip=a.clip, chan_flag_sigma=a.chan_flag_sigma,
                  cond_rfi_rate=a.disturbed_rfi_rate, calinfo_path=a.calinfo,
                  beam_model=a.beam_model, far_matched=not a.no_far_match,
                  inject_scatter=a.inject_scatter, inject_tau600_ms=tuple(a.inject_tau600_ms),
                  inject_unscattered_fraction=a.inject_unscattered,
                  hp_factor=a.hp_factor, boxcar_stride_div=a.boxcar_stride,
                  noise_every=a.noise_every, detrend_factor=a.detrend_factor, detrend=a.detrend,
                  history_dtype=a.history_dtype,
                  **{k: v for k, v in (("widths", a.widths), ("hp_tau", a.hp_tau),
                                       ("detrend_len", a.detrend_len)) if v is not None})


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
    p.add_argument("--slots", type=int, default=64, help="ring slots (128 for 41.8 ms images)")
    add_sim_args(p)

    p = sub.add_parser("simtest", help="simulator and monitor together, then evaluate")
    p.add_argument("--duration", type=float, default=1800)
    p.add_argument("--realtime", action="store_true",
                   help="run the simulator as a separate process at the image rate, "
                        "through a shared-memory ring")
    add_monitor_args(p)
    add_sim_args(p)

    p = sub.add_parser("evaluate", help="match candidates (or combiner events) with the truth")
    p.add_argument("--truth", help="a simulator's truth file (with --out)")
    p.add_argument("--out", help="a monitor's output directory")
    p.add_argument("--combiner", help="a combiner's output directory (with --frb-file)")
    p.add_argument("--frb-file", help="the FRB list given to the simulators")
    p.add_argument("--monitors", help="name=dir,... of the bands' simtest outputs (truth files), "
                                      "to check that no detectable FRB was lost")

    sub.add_parser("build", help="compile the kernels")

    p = sub.add_parser("combine", help="consensus of the monitors of all bands")
    p.add_argument("--nodes", required=True, help="name=url,name=url,... of the monitors")
    p.add_argument("--db-port", type=int, default=5433,
                   help="port of the nodes' local databases, on each monitor's host (0: HTTP only)")
    p.add_argument("--db-name", default="skymon")
    p.add_argument("--db-user", default="skymon", help="password from $SKYMON_DB_PASSWORD")
    p.add_argument("--out", default="skymon_combiner",
                   help="local directory for a copy of the events (the state is in the nodes' databases)")
    p.add_argument("--port", type=int, default=8096)
    p.add_argument("--poll", type=float, default=10.0)
    p.add_argument("--confirm-snr", type=float, default=6.0)
    p.add_argument("--inject-every", type=float, default=0.0,
                   help="mean hours between coordinated injections (0 = off)")
    p.add_argument("--inject-dm", type=float, nargs=2, default=(50.0, 1000.0))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--context-swpc", action=argparse.BooleanOptionalAction, default=True,
                   help="attach NOAA SWPC solar radio sweeps and X-ray flares to events")
    p.add_argument("--lightning-url", default=None,
                   help="URL or local file of lightning strikes (JSON; see context.py), off by default")
    p.add_argument("--context-every", type=float, default=15.0,
                   help="minutes between polls of each external context source")

    p = sub.add_parser("archive", help="copy what is worth keeping to the archive; free local disk")
    p.add_argument("--db", default=os.environ.get("SKYMON_DB"), required=False)
    p.add_argument("--data", required=True, help="the monitor's local output directory")
    p.add_argument("--archive", required=True, help="the archive root (on the shared file system)")
    p.add_argument("--node", default=None, help="this node's name in the combiner's events")
    p.add_argument("--rfi-days", type=float, default=3.0)
    p.add_argument("--review-days", type=float, default=14.0)
    p.add_argument("--keep-days", type=float, default=30.0)
    p.add_argument("--max-disk", type=float, default=0.8)
    p.add_argument("--every", type=float, default=60.0)

    p = sub.add_parser("frblist", help="FRBs for the simulators of several bands")
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--duration", type=float, default=3600)
    p.add_argument("--rate", type=float, default=20, help="per hour")
    p.add_argument("--dm", type=float, nargs=2, default=(50, 600))
    p.add_argument("--snr", type=float, nargs=2, default=(10, 30))
    p.add_argument("--f-low", type=float, default=33.0)
    p.add_argument("--f-high", type=float, default=77.3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--spectra", action="store_true",
                   help="give each FRB a spectrum (power law or band-limited) and scattering")
    p.add_argument("--tau600-ms", type=float, nargs=2, default=None,
                   help="scattering at 600 MHz log-uniform over this range (ms), scaled as "
                        "nu^-4 (0.31 10 is the base distribution of the sensitivity study)")
    p.add_argument("--unscattered", type=float, default=0.0,
                   help="fraction left unscattered with --tau600-ms")
    a = ap.parse_args(argv)

    if a.cmd == "combine":
        return cmd_combine(a)
    if a.cmd == "archive":
        from .archiver import Archiver
        arc = Archiver(a.db, a.data, a.archive, a.node, a.rfi_days, a.review_days, a.keep_days,
                       a.max_disk)
        stop = threading.Event()
        for sg in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sg, lambda *_: stop.set())
        print(f"archiver: {a.data} -> {a.archive} as {arc.node}", flush=True)
        arc.run(stop, a.every)
        return 0
    if a.cmd == "frblist":
        from .simulate import make_frb_list
        frbs = make_frb_list(a.start or time.time(), a.duration, a.rate, tuple(a.dm),
                             tuple(a.snr), a.f_low, a.f_high, seed=a.seed, spectra=a.spectra,
                             tau600_ms=a.tau600_ms, unscattered=a.unscattered)
        with open(a.out, "w") as f:
            json.dump(frbs, f, indent=1)
        print(f"{len(frbs)} FRBs written to {a.out}")
        return 0

    if a.cmd == "build":
        from . import kernels
        print(kernels.build(force=True))
        return 0
    if a.cmd == "evaluate":
        from .evaluate import report, report_combiner
        if a.combiner and a.monitors:
            from .evaluate import report_no_loss
            dirs = dict(item.split("=", 1) for item in a.monitors.split(","))
            print(report_no_loss(a.frb_file, a.combiner, dirs)["text"])
        elif a.combiner:
            print(report_combiner(a.frb_file, a.combiner)["text"])
        else:
            print(report(a.truth, a.out)["text"])
        return 0
    if a.cmd == "simulate":
        from .simulate import run_shm
        sim = make_sim(a)
        print(json.dumps(run_shm(sim, a.shm, a.duration, nslots=a.slots, realtime=not a.fast)))
        return 0
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "simtest":
        return cmd_simtest(a)


def add_sim_args(p):
    p.add_argument("--chan0", type=int, default=2400 - 64,
                   help="first NDP channel (2336: 55.87-58.94 MHz)")
    p.add_argument("--nchan", type=int, default=128)
    p.add_argument("--nseqs", type=int, default=2000,
                   help="sequences per image: 2000 for 83.6 ms images, 1000 for 41.8 ms")
    p.add_argument("--frb-per-hour", type=float, default=30)
    p.add_argument("--frb-snr", type=float, nargs=2, default=(8, 30))
    p.add_argument("--frb-dm", type=float, nargs=2, default=(20, 3000))
    p.add_argument("--rfi-per-hour", type=float, default=6)
    p.add_argument("--start", type=float, default=None, help="simulated start (unix)")
    p.add_argument("--sim-seed", type=int, default=1)
    p.add_argument("--truth", default=None)
    p.add_argument("--frb-file", default=None,
                   help="FRBs from this list (skymon frblist) instead of at random")
    p.add_argument("--sim-calinfo", default=None,
                   help="write a calibration record of the simulated sky here (flux.py)")


def make_sim(a, truth=None):
    from .geometry import Band
    from .simulate import SkySimulator, tier_dt_function
    band = Band(a.chan0, a.nchan)
    nseqs = getattr(a, "nseqs", 2000)
    dt = nseqs * 8192 / 196e6
    return SkySimulator(chan0=a.chan0, nchan=a.nchan, nseqs=nseqs, start_unix=a.start,
                        frb_per_hour=a.frb_per_hour, snr=tuple(a.frb_snr), dm=tuple(a.frb_dm),
                        rfi_per_hour=a.rfi_per_hour, truth_path=truth or a.truth,
                        seed=a.sim_seed, calinfo_path=getattr(a, "sim_calinfo", None),
                        tier_dt=tier_dt_function(band.edges_mhz(), dt, a.nchan),
                        frb_list=json.load(open(a.frb_file)) if a.frb_file else None)


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
    monitor.start_db_sync(stop)
    print(f"skymon: following /dev/shm/{a.shm}", flush=True)
    run_ring(monitor, a.shm, stop)
    monitor.store.flush()
    return 0


def cmd_simtest(a):
    import subprocess

    os.makedirs(a.out, exist_ok=True)
    truth = os.path.join(a.out, "truth.jsonl")
    # the simulator's calibration record, so that candidates get fluences in its units
    if not a.calinfo:
        a.calinfo = os.path.join(os.path.abspath(a.out), "calinfo.json")
    a.sim_calinfo = a.calinfo
    a.beam_model = "record"  # the simulated sky has its own primary beam
    monitor = Monitor(make_config(a))
    _serve(monitor, a.port)
    sync_stop = threading.Event()
    monitor.start_db_sync(sync_stop)
    t_start = time.time()
    if a.realtime:
        ring = f"skysim_{os.getpid()}"
        start = a.start or time.time()
        cmd = [sys.executable, "-m", "skymon", "simulate", "--shm", ring,
               "--duration", str(a.duration), "--chan0", str(a.chan0), "--nchan", str(a.nchan),
               "--nseqs", str(getattr(a, "nseqs", 2000)),
               "--slots", str(128 if getattr(a, "nseqs", 2000) < 2000 else 64),
               "--frb-per-hour", str(a.frb_per_hour), "--frb-snr", *map(str, a.frb_snr),
               "--frb-dm", *map(str, a.frb_dm), "--rfi-per-hour", str(a.rfi_per_hour),
               "--start", str(start), "--sim-seed", str(a.sim_seed), "--truth", truth,
               "--sim-calinfo", a.sim_calinfo] + \
            (["--frb-file", a.frb_file] if a.frb_file else [])
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


def cmd_combine(a):
    from .combiner import Combiner, CombinerConfig, serve as serve_combiner
    nodes = dict(item.split("=", 1) for item in a.nodes.split(",") if "=" in item)
    dsns = {}
    if a.db_port:
        import urllib.parse as up
        pw = os.environ.get("SKYMON_DB_PASSWORD", "")
        for n, u in nodes.items():
            host = up.urlparse(u).hostname
            dsns[n] = f"host={host} port={a.db_port} dbname={a.db_name} user={a.db_user}" + \
                (f" password={pw}" if pw else "")
    comb = Combiner(CombinerConfig(nodes=nodes, dsns=dsns, out_dir=a.out, poll_s=a.poll,
                                   token=os.environ.get("SKYMON_TOKEN"),
                                   confirm_snr=a.confirm_snr,
                                   inject_interval_s=a.inject_every * 3600,
                                   inject_dm=tuple(a.inject_dm), seed=a.seed,
                                   context_swpc=a.context_swpc, lightning_url=a.lightning_url,
                                   context_interval_s=a.context_every * 60))
    if a.port:
        serve_combiner(comb, port=a.port)
        print(f"combiner: HTTP on port {a.port}", flush=True)
    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.set())
    print(f"combiner: following {', '.join(nodes)}", flush=True)
    comb.run(stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
