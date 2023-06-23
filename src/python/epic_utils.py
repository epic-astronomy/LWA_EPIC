import numpy as np
from lsl.common.stations import lwasv
from astropy.constants import c as speed_of_light
import matplotlib.image
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, FK5

import datetime
import time
from  MCS2 import Communicator
DATE_FORMAT = "%Y_%m_%dT%H_%M_%S"
FS = 196.0e6
CHAN_BW = 25.0e3
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
    lsl_locsf[[i for i, a in enumerate(lwasv[::2]) if a.stand.id == 256], :] = 0.0

    # adjusted to ensure the pixel size <= half the wavelength
    delta = (2 * grid_size * np.sin(np.pi * grid_resolution / 360)) ** -1
    lsl_locsf = lsl_locsf / delta
    lsl_locsf -= np.min(lsl_locsf, axis=0, keepdims=True)

    # Centre locations slightly
    # divide by wavelength and add grid_size/2 to get the correct grid positions
    lsl_locsf -= np.max(lsl_locsf, axis=0, keepdims=True) / 2.0

    # add ntime axis
    # final dimensions = 3, ntime, nchan, nant, npol
    return dict(delta=delta, locations=lsl_locsf.astype(np.double).ravel().copy())


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

    for i in range(nstand):
        # X
        a = lwasv.antennas[2 * i + 0]
        delay = a.cable.delay(freq) - a.stand.z / speed_of_light.value
        phases[:, i, 0] = np.exp(2j * np.pi * freq * delay)
        phases[:, i, 0] /= np.sqrt(a.cable.gain(freq))

        # Y
        a = lwasv.antennas[2 * i + 1]
        delay = a.cable.delay(freq) - a.stand.z / speed_of_light.value
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
        if a.stand.id == 256:
            phases[:, i, :] = 0.0

    return phases.ravel().copy()

def save_output(output_arr, grid_size, nchan, filename, metadata):
    """
    Save image to disk

    output_arr: Output image array
    grid_size: X or Y Size of the grid in pixels
    nchan: Number of output channlels in the image
    filename: Name of the file to save
    metadata: Dict with the metatadata to be written to the fits file
    """
    output_arr = output_arr.reshape((nchan, grid_size, grid_size))
    output_arr_sft = np.fft.fftshift(output_arr, axes=(1,2))

    print(metadata)

    print(output_arr_sft.min(), output_arr_sft.max())
    phdu = fits.PrimaryHDU()
    # for k,v in metadata.items():
    #     phdu.header[k] = v

    phdu.header['DATE-OBS']=Time(
        metadata["time_tag"]/FS + 1e-3 * metadata['img_len_ms']/2.0,
        format="unix",
        precision=6,
    ).isot

    sll = (2 * metadata["grid_size"] * np.sin(np.pi * metadata["grid_res"] / 360)) ** -1
    cfreq = (metadata["chan0"] + metadata["nchan"]/2-1)*CHAN_BW

    phdu.header["BUNITS"]="UNCALIB"
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

    dt = TimeDelta(1e-3 * metadata["img_len_ms"], format="sec")
    dtheta_x = 2 * np.arcsin(0.5 / (metadata["grid_size"] * sll))
    dtheta_y = 2 * np.arcsin(0.5 / (metadata["grid_size"] * sll))
    crit_pix_x = float(metadata["grid_size"] / 2 + 1)

    # Need to correct for shift in center pixel when we flipped dec dimension
    # when writing npz, Only applies for even dimension size
    crit_pix_y = float(metadata["grid_size"] / 2 + 1) - (metadata["grid_size"] + 1) % 2
    delta_x = -dtheta_x * 180.0 / np.pi
    delta_y = dtheta_y * 180.0 / np.pi
    delta_f = CHAN_BW

    crit_pix_f = (metadata["nchan"] - 1) * 0.5 + 1  # +1 for FITS numbering

    t0 = Time(
        metadata["time_tag"] / FS,
        format="unix",
        precision=6,
        location=(lwasv_station.lon * 180. / np.pi, lwasv_station.lat * 180. / np.pi)
    )

    lsts = t0.sidereal_time("apparent")
    coords = SkyCoord(
        lsts.deg, lwasv_station.lat * 180. / np.pi, obstime=t0, unit="deg"
    ).transform_to(FK5(equinox="J2000"))



    img_data = np.transpose(output_arr_sft[:,::-1,::-1],(0,2,1))
    img_data = img_data/img_data.max(axis=(1,2),keepdims=True)
    ihdu = fits.ImageHDU(img_data)

    ihdu.header["DATETIME"]=t0.isot
    ihdu.header["LST"]=lsts.hour
    ihdu.header["EQUINOX"] = "J2000"

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
    ihdu.header["CTYPE3"] = "FREQ"
    ihdu.header["CRPIX3"] = crit_pix_f
    ihdu.header["CDELT3"] = delta_f
    ihdu.header["CRVAL3"] = cfreq
    ihdu.header["CUNIT3"] = "Hz"
    # # Coordinates - Stokes parameters
    # ihdu.header["CTYPE4"] = "STOKES"
    # ihdu.header["CRPIX4"] = 1
    # ihdu.header["CDELT4"] = -1
    # ihdu.header["CRVAL4"] = pol_nums[pol_order[0]]
    # # Coordinates - Complex
    # ihdu.header["CTYPE5"] = "COMPLEX"
    # ihdu.header["CRVAL5"] = 1.0
    # ihdu.header["CRPIX5"] = 1.0
    # ihdu.header["CDELT5"] = 1.0

    hdulist = fits.HDUList([phdu, ihdu])
    hdulist.writeto(f"{filename}.fits", overwrite=True)

    temp_im = output_arr_sft[50,::-1,:].T
    # temp_im[:,0:4]=0
    # temp_im[0:4,:]=0
    # temp_im[:,-4:]=0
    # temp_im[-4:,:]=0
    matplotlib.image.imsave(f"{filename}.png", temp_im)
    matplotlib.image.imsave("original_test_out.png",(output_arr[50,:,:]))



def get_ADP_time_from_unix_epoch():
    """
    Get time in seconds from the unix epoch using the UTC start timestamp from the 
    ADP service
    """
    got_utc_start = False
    while not got_utc_start:
        try:
            with Communicator() as adp_control:
                utc_start = adp_control.report('UTC_START')
                # Check for valid timestamp
                utc_start_dt = datetime.datetime.strptime(utc_start, DATE_FORMAT)
            got_utc_start = True
        except Exception as ex:
            print(ex)
            time.sleep(0.1)
    return (utc_start_dt-ADP_EPOCH).total_seconds()

def get_time_from_unix_epoch(utcstart):
    """
    Get time difference in seconds from unix epoch

    utcstart: Timestamp string
    """
    utc_start_dt = datetime.datetime.strptime(utcstart, DATE_FORMAT)
    return (utc_start_dt-ADP_EPOCH).total_seconds()



def get_40ms_gulp():
    """
    Generate a 40 ms gulp from an npz file
    """
    # npfile=np.load("data/40ms_128chan_gulp_c64.npz")
    # npfile=np.load("data/40ms_128chan_gulp_c64_virtransit.npz")
    npfile=np.load("data/40ms_128chan_600offset_gulp_c64_virtransit.npz")
    meta=np.concatenate((npfile['meta'].ravel() , [npfile['data'].size], npfile['data'].shape))
    _data=npfile["data"].copy()
    # for i in range(20):
    #     print(i,np.min(_data[i,1,:,:]))
    return dict(meta=meta.astype(np.double).ravel().copy(),\
            data=npfile['data'].ravel().copy())

def gaussian(x, mu=0, sigma=4):
    return np.exp(-(x-mu)**2/(2*sigma**2))

def integrate_pixel(fun,dx, dy,d_per_pixel=1, nsteps=5):
    sum=0
    delta = 1/float(nsteps)
    offset = 1/float(2 * nsteps); 

    #pragma unroll
    xinit=offset
    yinit=offset
    for x in range(nsteps):
        for y in range(nsteps):
            xx=x*delta+offset
            yy=y*delta+offset
            sum+=fun((dx+xx)*d_per_pixel)*fun((dy+yy)*d_per_pixel)

    return sum
    
    
def get_gaussian_2D(support=3):
    g = np.zeros((support, support))
    for xx in range(support):
        for yy in range(support):
            x = xx - support/2
            y = yy - support/2

            g[xx, yy] =  integrate_pixel(gaussian, x, y)

    g/=g.sum()
    return g

def get_correction_grid(corr_ker_arr, grid_size, support, nchan, oversample=4):
    """
    Generates a correction grid based on the specified kernels

    corr_ker_arr: Correction kernel array for each frequency
    grid_size: 1D size of the correction grid
    support: Support size
    nchan: Number of channels in the correction grid
    """
    grid_size_orig = grid_size
    grid_size = grid_size * oversample
    corr_ker_arr = corr_ker_arr.reshape((nchan, support, support))
    corr_grid_arr = np.zeros((nchan, grid_size, grid_size))

    offset = (grid_size - grid_size_orig)//2

    g2d = get_gaussian_2D(support)

    # corr_grid_arr[:,:,:]=5

    kernel_offset=4
    for i in range(nchan):
        corr_grid_arr[i,5:5+support, 5:5+support] = corr_ker_arr[i,:,:] #* g2d

    corr_grid_arr = np.absolute(
        np.fft.fftshift(
            np.transpose(# cuFFTDx generates the transpose of the image
                np.fft.fftshift(
                np.fft.ifft2(corr_grid_arr)
                )[:,offset:offset+grid_size_orig, offset:offset+grid_size_orig], axes=(0,2,1)
            )
            )
        )**2
    matplotlib.image.imsave("gpu_corr.png",(corr_grid_arr[9,:,:]))
    matplotlib.image.imsave("gpu_corr_kernel.png",(corr_ker_arr[9,:,:]))
    corr_grid_arr = np.reciprocal(corr_grid_arr)
    corr_grid_arr = corr_grid_arr/corr_grid_arr.sum(axis=(1,2),keepdims=True)

    return (corr_grid_arr).copy().ravel()

if __name__=="__main__":
    # a = gen_phases_lwasv(132, 800)
    # print(a[:4])

    freq = 1260 * 25000

    # b = lwasv.antennas[0]
    # delay = b.cable.delay(freq) - b.stand.z / speed_of_light.value
    # cphase = np.exp(2j * np.pi * freq * delay) / np.sqrt(b.cable.gain(freq))
    # print(a[0], cphase)
    # # assert(a[0]==cphase)

    # b = lwasv.antennas[1]
    # delay = b.cable.delay(freq) - b.stand.z / speed_of_light.value
    # print(a[1],cphase)

    res=gen_loc_lwasv(64, 2)
    print(freq)
    # print(res['locations'][:3*256].astype(float))
    print(32+(res['locations'][3*256:6*256].astype(float)*freq/3e8))


    # assert(a[1]==cphase)


