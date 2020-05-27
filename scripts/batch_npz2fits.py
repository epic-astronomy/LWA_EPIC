#!/usr/bin/env python
import numpy
import argparse
from astropy.io import fits
from astropy.time import Time
from astropy.time import TimeDelta
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5


# Borrowed from original EPIC repository.
def epic2fits(filename, data, hdr, image_nums):
    """Dump EPIC images into FITs file.

    Parameters
    -----------
    filename : str
        file to save
    data : numpy.ndarray
        shape (Ntimes, Nfreq, Npol, Ny, Nx)
    hdr : dict
        header dictionary with necessary metadata
    image_nums : list
        List of image numbers from start of sequence. Length Ntimes.

    """
    # Set up primary header and create file
    hdu = fits.PrimaryHDU()
    hdu.header["TELESCOP"] = hdr["telescope"]
    t0 = Time(
        hdr["time_tag"] / hdr["FS"] + 1e-3 * hdr["accumulation_time"] / 2.,
        format="unix",
        precision=6,
        location=(hdr["longitude"], hdr["latitude"]))
    hdu.header["DATE-OBS"] = t0.isot
    hdu.header["BUNIT"] = hdr["data_units"]
    hdu.header["BSCALE"] = 1e0
    hdu.header["BZERO"] = 0e0
    hdu.header["EQUINOX"] = "J2000"
    hdu.header["EXTNAME"] = "PRIMARY"
    hdu.header["GRIDDIMX"] = hdr["grid_size_x"]
    hdu.header["GRIDDIMY"] = hdr["grid_size_y"]
    hdu.header["DGRIDX"] = hdr["sampling_length_x"]
    hdu.header["DGRIDY"] = hdr["sampling_length_y"]
    hdu.header["INTTIM"] = hdr["accumulation_time"] * 1e-3
    hdu.header["INTTIMU"] = "SECONDS"
    hdu.header["CFREQ"] = hdr["cfreq"]
    hdu.header["CFREQU"] = "HZ"

    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)

    # Restructure data in preparation to stuff into fits
    data = data.transpose(0, 2, 1, 3, 4)  # Now (Ntimes, Npol, Nfreq, y, x)
    # Reorder pol for fits convention
    pol_dict = {"xx": -5, "yy": -6, "xy": -7, "yx": -8}
    pol_nums = [pol_dict[p] for p in hdr["pols"]]
    pol_order = numpy.argsort(pol_nums)[::-1]
    data = data[:, pol_order, :, :, :]
    # Break up real/imaginary
    data = data[:, numpy.newaxis, :, :, :, :]  # Now (Ntimes, 2 (complex), Npol, Nfreq, y, x)
    data = numpy.concatenate([data.real, data.imag], axis=1)

    if not isinstance(image_nums, (list, tuple, numpy.ndarray)):
        image_nums = [image_nums]

    dt = TimeDelta(1e-3 * hdr["accumulation_time"], format="sec")

    for im_num, d in zip(image_nums, data):
        hdu = fits.PrimaryHDU()
        # Time
        t = t0 + im_num * dt
        lst = t.sidereal_time("apparent")
        hdu.header["DATETIME"] = t.isot
        hdu.header["LST"] = lst.hour
        # Coordinates - sky
        ra0 = lst.deg
        dec0 = hdr["latitude"]
        coord = SkyCoord(ra0, dec0, frame="fk5", equinox="J" + str(t.jyear), unit="deg")
        coord.transform_to(FK5(equinox="J2000"))
        hdu.header["EQUINOX"] = "J2000"
        hdu.header["CTYPE1"] = "RA---SIN"
        hdu.header["CRPIX1"] = float(hdr["grid_size_x"] / 2 + 1)
        dtheta = 2 * numpy.arcsin(.5 / (hdr["grid_size_x"] * hdr["sampling_length_x"]))
        hdu.header["CDELT1"] = -dtheta * 180. / numpy.pi
        hdu.header["CRVAL1"] = coord.ra.deg
        hdu.header["CUNIT1"] = "deg"
        hdu.header["CTYPE2"] = "DEC--SIN"
        # Need to correct for shift in center pixel when we flipped dec dimension when writing npz
        # Only applies for even dimension size
        hdu.header["CRPIX2"] = float(hdr["grid_size_y"] / 2 + 1) - (hdr["grid_size_x"] + 1) % 2
        dtheta = 2 * numpy.arcsin(.5 / (hdr["grid_size_y"] * hdr["sampling_length_y"]))
        hdu.header["CDELT2"] = dtheta * 180. / numpy.pi
        hdu.header["CRVAL2"] = coord.dec.deg
        hdu.header["CUNIT2"] = "deg"
        # Coordinates - Freq
        hdu.header["CTYPE3"] = "FREQ"
        hdu.header["CRPIX3"] = (hdr["nchan"] - 1) * 0.5 + 1  # +1 for FITS numbering
        hdu.header["CDELT3"] = hdr["bw"] / hdr["nchan"]
        hdu.header["CRVAL3"] = hdr["cfreq"]
        hdu.header["CUNIT3"] = "Hz"
        # Coordinates - Stokes parameters
        hdu.header["CTYPE4"] = "STOKES"
        hdu.header["CRPIX4"] = 1
        hdu.header["CDELT4"] = -1
        hdu.header["CRVAL4"] = pol_nums[pol_order[0]]
        # Coordinates - Complex
        hdu.header["CTYPE5"] = "COMPLEX"
        hdu.header["CRVAL5"] = 1.0
        hdu.header["CRPIX5"] = 1.0
        hdu.header["CDELT5"] = 1.0

        fits.append(filename, d, hdu.header)


a = argparse.ArgumentParser(description="Convert batch of npz files to fits")
a.add_argument(
    "files",
    metavar="files",
    type=str,
    nargs="*",
    default=[],
    help="*.npz files to convert to fits.",
)
args = a.parse_args()

for f in args.files:
    d = numpy.load(f, allow_pickle=True)
    of = f[:-3] + "fits"
    epic2fits(of, d["image"], numpy.ravel(d["hdr"])[0], d["image_nums"])
