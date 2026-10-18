import numpy as np
from lsl.common.stations import lwasv
from astropy.constants import c as speed_of_light
import matplotlib.image
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5
from scipy.signal import correlate2d

import datetime
import os
import sys
import time
from MCS2 import Communicator
import uuid
from daily_digest import ingest_daily_digest
from epic_geometry import apply_wcs, array_frame

DATE_FORMAT = "%Y_%m_%dT%H_%M_%S"
FS = 196.0e6
CHAN_BW = FS / 8192  # NDP channel width
ADP_EPOCH = datetime.datetime(1970, 1, 1)

lwasv_station = lwasv


def gen_loc_lwasv(grid_size, grid_resolution):
    """
    Generate grid centered locations of LWASV stands compatible with DFT code.
    The indices must be divided by the wavelength to transform the coordinates
    to pixel units

    grid_size: Dimension of the grid.
    grid_resolution: Grid resolution in degrees
    """

    lsl_locsf = np.array(
        [(ant.stand.x, ant.stand.y, ant.stand.z) for ant in lwasv[::2]]
    )
    # Express the stands in the grid frame of the phase centre (rows: grid x,
    # grid y, phase centre). For EPIC_PHASE_CENTER=zenith this is the identity.
    _, frame, _ = array_frame()
    lsl_locsf = lsl_locsf @ frame.T
    lsl_locsf[
        [i for i, a in enumerate(lwasv[::2]) if a.stand.id == 256], :
    ] = 0.0

    # adjusted to ensure the pixel size <= half the wavelength
    delta = (2 * grid_size * np.sin(np.pi * grid_resolution / 360)) ** -1
    lsl_locsf = lsl_locsf / delta
    lsl_locsf -= np.min(lsl_locsf, axis=0, keepdims=True)

    # Centre locations slightly
    # divide by wavelength and add grid_size/2 to get the correct grid
    # positions
    lsl_locsf -= np.max(lsl_locsf, axis=0, keepdims=True) / 2.0

    # add ntime axis
    # final dimensions = 3, ntime, nchan, nant, npol
    return dict(
        delta=delta, locations=lsl_locsf.astype(np.double).ravel().copy()
    )


def gen_phases_lwasv(nchan, chan0):
    """
    Generate complex phases for the LWA-SV antennas

    nchan: Number of channels
    chan0: Channel number of the first channel
    """
    nstand = int(len(lwasv.antennas) / 2)
    npol = 2
    phases = np.zeros((nchan, nstand, npol), dtype=np.complex64)
    bandwidth = CHAN_BW
    freq = np.arange(chan0, chan0 + nchan) * bandwidth
    # Geometric delay towards the phase centre (zenith: p = (0, 0, 1), i.e. z/c)
    pc, _, _ = array_frame()

    for i in range(nstand):
        # X
        a = lwasv.antennas[2 * i + 0]
        delay = a.cable.delay(freq) - np.dot(pc, (a.stand.x, a.stand.y, a.stand.z)) / speed_of_light.value
        phases[:, i, 0] = np.exp(2j * np.pi * freq * delay)
        phases[:, i, 0] /= np.sqrt(a.cable.gain(freq))
        # phases[:, i, 0] = np.exp(1j * np.pi)

        # Y
        a = lwasv.antennas[2 * i + 1]
        delay = a.cable.delay(freq) - np.dot(pc, (a.stand.x, a.stand.y, a.stand.z)) / speed_of_light.value
        phases[:, i, 1] = np.exp(2j * np.pi * freq * delay)
        phases[:, i, 1] /= np.sqrt(a.cable.gain(freq))
        # Explicit bad and suspect antenna masking - this will
        # mask an entire stand if either pol is bad
        if (
            lwasv.antennas[2 * i + 0].combined_status < 33
            or lwasv.antennas[2 * i + 1].combined_status < 33
        ):
            phases[:, i, :] = 0.0
        # Explicit outrigger masking - we probably want to do
        # away with this at some point
        if a.stand.id in [256]+[i for i in range(122,129)]+[185,186]:
            phases[:, i, :] = 0.0

    return phases.ravel().copy()


_log_last = {}


def _log_limited(key, msg, every_s=300.0):
    """Print msg at most once per every_s for each key, with the number held back"""
    now = time.time()
    last, held = _log_last.get(key, (0.0, 0))
    if now - last >= every_s:
        extra = f" ({held} more since the last report)" if held else ""
        print(f"save_output: {msg}{extra}", file=sys.stderr, flush=True)
        _log_last[key] = (now, 0)
    else:
        _log_last[key] = (last, held + 1)


def quick_look(img_data, grid_size, sll, scale=4):
    """
    RGBA quick look of an accumulation: Stokes I (XX + YY summed over all
    channels), blank beyond the horizon, on an asinh stretch scaled by the
    sky's robust noise so that one bright source (the Sun, Cas A) does not
    flatten the rest, enlarged scale times. img_data is
    [pol][chan][y][x] as written to the FITS file; the horizon is at
    grid_size * sll pixels from the phase centre (grid/2, grid/2 - 1 for an
    even grid; apply_wcs).
    """
    import matplotlib

    stokes_i = np.asarray(img_data[0], np.float64).sum(0) + np.asarray(img_data[1], np.float64).sum(0)
    n = stokes_i.shape[-1]
    y, x = np.mgrid[:n, :n]
    sky = np.hypot(x - n // 2, y - (n // 2 - (n + 1) % 2)) <= grid_size * sll
    vals = stokes_i[sky & np.isfinite(stokes_i)]
    if vals.size:
        med = np.median(vals)
        noise = 1.4826 * np.median(np.abs(vals - med)) or 1.0
        lo, hi = np.percentile(vals, 1.0), vals.max()
    else:
        med, noise, lo, hi = 0.0, 1.0, 0.0, 1.0
    # asinh stretch: linear within a few noise levels of the background,
    # logarithmic above, so a bright source and the sky around it both show
    stretch = lambda v: np.arcsinh((v - lo) / (3.0 * noise))  # noqa: E731
    norm = np.clip(stretch(stokes_i) / (stretch(hi) or 1.0), 0, 1)
    rgba = matplotlib.colormaps["inferno"](norm)
    rgba[~sky] = (0, 0, 0, 1)
    return np.kron(rgba, np.ones((scale, scale, 1)))


def save_output(output_arr, grid_size, nchan, outdir, metadata):
    """
    Save an accumulated image to <outdir>/YYYYMMDD/ (UTC day of its middle)
    and return the path of the FITS file, or "" when it could not be written.

    output_arr: Output image array, the sum of metadata["nimg"] images
    grid_size: X or Y Size of the grid in pixels
    nchan: Number of output channlels in the image
    outdir: Output directory (--out_dir)
    metadata: Dict with the metatadata to be written to the fits file

    It never raises: a missing or read-only directory, a full disk or a stale
    NFS handle is logged (rate-limited) and that accumulation is dropped. The
    caller runs it off the imaging pipeline, so a hung mount only stalls the
    writer.
    """
    try:
        return _save_output(output_arr, grid_size, nchan, outdir, metadata)
    except Exception as e:  # noqa: BLE001 - the imager must never stop here
        _log_limited("write", f"accumulation not written to {outdir}: {e!r}")
        return ""


def _save_output(output_arr, grid_size, nchan, outdir, metadata):
    # start = time.time()
    output_arr = output_arr.reshape((nchan, grid_size, grid_size, 4))
    # output_arr_sft = np.fft.fftshift(output_arr, axes=(2, 3))

    # print(metadata)

    # print(output_arr_sft.min(), output_arr_sft.max())
    phdu = fits.PrimaryHDU()
    for k, v in metadata.items():
        # print("py", k,v)
        phdu.header[k] = v
    # An accumulation of --nimg_accum images spans img_len_ms; the sky drifts by
    # 0.25 deg per minute meanwhile. Times, the LST and the celestial WCS refer
    # to the middle of the span, so that a source's pixel (and the daily digest
    # read from it) is centred on its track through the accumulation.
    # img_len_ms is the span actually summed: nimg images, fewer than
    # --nimg_accum for a partial sum written at a gap in the images
    t_beg = metadata["time_tag"] / FS
    t_end = t_beg + 1e-3 * metadata["img_len_ms"]
    t_mid = 0.5 * (t_beg + t_end)
    nimg = max(1, int(metadata.get("nimg", 1)))
    phdu.header["DATE-OBS"] = Time(t_mid, format="unix", precision=6).isot
    phdu.header["DATE-BEG"] = Time(t_beg, format="unix", precision=6).isot
    phdu.header["DATE-END"] = Time(t_end, format="unix", precision=6).isot
    phdu.header["NIMGACC"] = (nimg, "images summed in this accumulation")

    sll = (
        2 * metadata["grid_size"] * np.sin(np.pi * metadata["grid_res"] / 360)
    ) ** -1
    cfreq = (metadata["chan0"] + metadata["nchan"] / 2 - 1) * CHAN_BW

    phdu.header["BUNITS"] = "UNCALIB"
    phdu.header["BSCALE"] = 1e0
    phdu.header["BZERO"] = 0e0
    phdu.header["EQUINOX"] = "J2000"
    phdu.header["EXTNAME"] = "PRIMARY"
    phdu.header["GRIDDIMX"] = metadata["grid_size"]
    phdu.header["GRIDDIMY"] = metadata["grid_size"]
    phdu.header["DGRIDX"] = sll
    phdu.header["DGRIDY"] = sll
    phdu.header["INTTIM"] = metadata["img_len_ms"] * 1e-3
    phdu.header["INTTIMU"] = "SECONDS"
    phdu.header["CFREQ"] = cfreq
    phdu.header["CFREQU"] = "HZ"

    delta_f = CHAN_BW

    crit_pix_f = (metadata["nchan"] - 1) * 0.5 + 1  # +1 for FITS numbering

    t0 = Time(
        t_mid,
        format="unix",
        precision=6,
        location=(
            lwasv_station.lon * 180.0 / np.pi,
            lwasv_station.lat * 180.0 / np.pi,
        ),
    )

    lsts = t0.sidereal_time("apparent")

    # img_data = np.transpose(output_arr[:, :, :, :], (0, 1, 3, 2))
    img_data = np.transpose(output_arr[:, :, :, :], (3, 0, 2, 1))
    img_data = np.fft.fftshift(img_data, axes=(2, 3))[:, :, ::-1, :]
    # img_data = img_data / img_data.max(axis=(2, 3), keepdims=True)
    ihdu = fits.ImageHDU(img_data)

    ihdu.header["DATETIME"] = t0.isot
    ihdu.header["LST"] = lsts.hour

    # Celestial axes: CRVAL at the phase centre via AltAz (precession included)
    # and the grid-axis rotation in a PC matrix. See epic_geometry.apply_wcs.
    apply_wcs(ihdu.header, t0, metadata["grid_size"], metadata["grid_res"])
    # Coordinates - Freq
    ihdu.header["CTYPE3"] = "FREQ"
    ihdu.header["CRPIX3"] = crit_pix_f
    ihdu.header["CDELT3"] = delta_f
    ihdu.header["CRVAL3"] = cfreq
    ihdu.header["CUNIT3"] = "Hz"
    # # Coordinates - Stokes parameters
    ihdu.header["CTYPE4"] = "STOKES"
    ihdu.header["CRPIX4"] = 1
    ihdu.header["CDELT4"] = -1
    ihdu.header["CRVAL4"] = -5  # pol_nums[pol_order[0]]

    hdulist = fits.HDUList([phdu, ihdu])
    filename = f"EPIC_{t_mid:.3f}_{cfreq/1e6:.3f}MHz"
    daydir = os.path.join(outdir, time.strftime("%Y%m%d", time.gmtime(t_mid)))
    os.makedirs(daydir, exist_ok=True)
    path = os.path.join(daydir, filename + ".fits")
    # written under a temporary name, so that a failed write leaves no partial file
    try:
        hdulist.writeto(path + ".part", overwrite=True)
        os.replace(path + ".part", path)
    except BaseException:
        try:
            os.remove(path + ".part")
        except OSError:
            pass
        raise

    try:
        matplotlib.image.imsave(os.path.join(daydir, filename + "_QL.png"),
                                quick_look(img_data, metadata["grid_size"], sll),
                                origin="lower")
    except Exception as e:  # noqa: BLE001 - the quick look is optional
        _log_limited("png", f"quick look not written: {e!r}")

    # the digest: per-image means over each watched source's patch, so that
    # a partial accumulation is on the same scale as a full one
    try:
        ingest_daily_digest(ihdu.header, phdu.header, img_data, cfreq, nimg=nimg)
    except Exception as e:  # noqa: BLE001 - database or watch list unavailable
        _log_limited("digest", f"daily digest not written: {e!r}")

    return path


def get_ADP_time_from_unix_epoch():
    """
    Get time in seconds from the unix epoch using the UTC start timestamp from
    the NDP service (which replaced ADP)
    """
    got_utc_start = False
    exception_count=0
    while not got_utc_start:
        try:
            with Communicator("MCS") as adp_control:
                utc_start = adp_control.report("UTC_START")
                # NDP reports a time tag. Raises while it reads NULL.
                utc_start_tt = int(utc_start, 10)
            got_utc_start = True
        except Exception as ex:
            exception_count += 1
            if exception_count > 10:
                raise ex
            time.sleep(0.1)
    return utc_start_tt / FS


def get_time_tag0_adp():
    timestamp0 = get_ADP_time_from_unix_epoch()
    time_tag0 = timestamp0 * FS
    return time_tag0


def get_time_from_unix_epoch(utcstart):
    """
    Get time difference in seconds from unix epoch

    utcstart: Timestamp string
    """
    utc_start_dt = datetime.datetime.strptime(utcstart, DATE_FORMAT)
    return (utc_start_dt - ADP_EPOCH).total_seconds()


def get_40ms_gulp():
    """
    Generate a 40 ms gulp from an npz file
    """
    # npfile=np.load("data/40ms_128chan_gulp_c64.npz")
    # npfile=np.load("data/40ms_128chan_gulp_c64_virtransit.npz")
    npfile = np.load("data/40ms_128chan_600offset_gulp_c64_virtransit.npz")
    # npfile = np.load("data/40ms_1chan_sim_50MHz.npz")
    # npfile = np.load("data/40ms_1chan_sim_50MHz_orig_beam.npz")
    # npfile = np.load("data/40ms_1chan_sim_50MHz_L2.npz")
    # npfile = np.load("data/40ms_1chan_sim_50MHz_asymm.npz")
    meta = np.concatenate(
        (npfile["meta"].ravel(), [npfile["data"].size], npfile["data"].shape)
    )
    # _data = npfile["data"].copy()
    # for i in range(20):
    #     print(i,np.min(_data[i,1,:,:]))
    return dict(
        meta=meta.astype(np.double).ravel().copy(),
        data=npfile["data"].ravel().copy(),
    )


def gaussian(x, mu=0, sigma=4):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def integrate_pixel(fun, dx, dy, d_per_pixel=1, nsteps=5):
    sum = 0
    delta = 1 / float(nsteps)
    offset = 1 / float(2 * nsteps)

    # pragma unroll
    # xinit = offset
    # yinit = offset
    for x in range(nsteps):
        for y in range(nsteps):
            xx = x * delta + offset
            yy = y * delta + offset
            sum += fun((dx + xx) * d_per_pixel) * fun((dy + yy) * d_per_pixel)

    return sum


def get_gaussian_2D(support=3):
    g = np.zeros((support, support))
    for xx in range(support):
        for yy in range(support):
            x = xx - support / 2
            y = yy - support / 2

            g[xx, yy] = integrate_pixel(gaussian, x, y)

    g /= g.sum()
    return g


def get_correction_grid(corr_ker_arr, grid_size, support, nchan, oversample=1):
    """
    Generates a correction grid based on the specified kernels

    corr_ker_arr: Correction kernel array for each frequency
    grid_size: 1D size of the correction grid
    support: Support size
    nchan: Number of channels in the correction grid
    """
    grid_size_orig = grid_size
    grid_size = int(grid_size * oversample)
    corr_ker_arr = corr_ker_arr.reshape((nchan, support, support))
    corr_grid_arr = np.zeros((nchan, grid_size, grid_size))

    offset = (grid_size - grid_size_orig) // 2

    kernel_offset = int(grid_size / 2)
    for i in range(nchan):
        corr_grid_arr[
            i,
            kernel_offset : kernel_offset + support,
            kernel_offset : kernel_offset + support,
        ] = corr_ker_arr[
            i, :, :
        ]  # * g2d
    matplotlib.image.imsave(
        "gpu_corr_kernel_raw.png", (corr_grid_arr[0, :, :])
    )
    X, Y = np.mgrid[0:grid_size, 0:grid_size]
    phase_correction = 1  # np.exp(-1j * 2 * np.pi / grid_size * (Y))

    corr_grid_arr = (
        np.absolute(
            np.fft.fftshift(
                np.transpose(  # cuFFTDx generates the transpose of the image
                    np.fft.fftshift(
                        np.fft.ifft2(corr_grid_arr * phase_correction)
                    )[
                        :,
                        offset : offset + grid_size_orig,
                        offset : offset + grid_size_orig,
                    ][
                        :, :, :
                    ],
                    axes=(0, 2, 1),
                )
            )
        )
        ** 2
    )
    matplotlib.image.imsave(
        "gpu_corr.png", (np.fft.fftshift(corr_grid_arr[0, :, :]))
    )
    matplotlib.image.imsave("gpu_corr_kernel.png", (corr_ker_arr[0, :, :]))
    ac = corr_ker_arr[0, :, :] / corr_ker_arr[0, :, :].sum()

    corr_grid_arr = np.reciprocal(corr_grid_arr)

    np.savez('corr_grid_arr.npz',corr_grid_arr)
    return (corr_grid_arr).copy().ravel()


def get_random_uuid():
    return str(uuid.uuid1())


def meta2pgtime(time_tag0, img_len_ms):
    return str(
        Time(
            time_tag0 / FS + 1e-3 * img_len_ms / 2.0, # time tag marks the end of the sequence
            format="unix",
            precision=6,
        ).isot
    )


if __name__ == "__main__":
    freq = 1260 * 25000

    res = gen_loc_lwasv(64, 2)
    print(freq)
    print(
        32 + (res["locations"][3 * 256 : 6 * 256].astype(float) * freq / 3e8)
    )
