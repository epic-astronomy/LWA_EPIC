import numpy as np
from lsl.common.stations import lwasv
from astropy.constants import c as speed_of_light
import matplotlib.image

import datetime
import time
from  MCS2 import Communicator
DATE_FORMAT = "%Y_%m_%dT%H_%M_%S"
FS = 196.0e6
CHAN_BW = 25.0e3
ADP_EPOCH = datetime.datetime(1970, 1, 1)


def gen_loc_lwasv(grid_size, grid_resolution):
    """
    Generate grid centered locations of LWASV stands compatible with DFT code.
    grid_size: Dimension of the grid.
    grid_resolution: Grid resolution in degrees
    """

    lsl_locsf = np.array(
        [(ant.stand.x, ant.stand.y, ant.stand.z) for ant in lwasv[::2]]
    )
    # print(lsl_locsf[0,:])

    lsl_locsf[[i for i, a in enumerate(lwasv[::2]) if a.stand.id == 256], :] = 0.0

    delta = (2 * grid_size * np.sin(np.pi * grid_resolution / 360)) ** -1
    # chan_wavelengths = speed_of_light.value / frequencies
    # sample_grid = chan_wavelengths * delta
    # sll = sample_grid[0] / chan_wavelengths[0]
    # lsl_locs = lsl_locs.T

    lsl_locsf = lsl_locsf / delta
    # print(lsl_locsf[0,:])

    # sample_grid[np.newaxis, np.newaxis, :, np.newaxis]
    # print(np.min(lsl_locsf, axis=0, keepdims=True))
    lsl_locsf -= np.min(lsl_locsf, axis=0, keepdims=True)
    # print(lsl_locsf[0,:])

    # Centre locations slightly
    # divide by wavelength and add grid_size/2 to get the correct grid positions
    lsl_locsf -= np.max(lsl_locsf, axis=0, keepdims=True) / 2.0
    # print(lsl_locsf[0,:])

    # add ntime axis
    # final dimensions = 3, ntime, nchan, nant, npol
    # locc = np.broadcast_to(lsl_locsf, (ntime, 3, npol, nchan, lsl_locs.shape[1])).transpose(1, 0, 3, 4, 2).copy()
    return dict(delta=delta, locations=lsl_locsf.astype(np.double).ravel().copy())


def gen_phases_lwasv(nchan, chan0):
    print("ok0")
    nstand = int(len(lwasv.antennas) / 2)
    npol = 2
    phases = np.zeros((nchan, nstand, npol), dtype=np.complex64)
    bandwidth = 25000
    freq = np.arange(chan0, chan0 + nchan) * bandwidth
    print("ok1")
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
    print("ok")
    return phases.conj().ravel().copy()

def save_output(output_arr, grid_size, nchan, filename):
    output_arr = output_arr.reshape((nchan, grid_size, grid_size))
    output_arr_sft = np.fft.fftshift(output_arr, axes=(1,2))

    # print(output_arr[0,:,:10])
    # output_arr = output_arr.sum(axis=0)
    # matplotlib.image.imsave(filename, output_arr[:,:].T/1000)
    matplotlib.image.imsave(filename, np.log10(output_arr_sft[0,:,:].T/1000))
    matplotlib.image.imsave("original_test_out.png",output_arr[0,:,:])



def get_ADP_time_from_unix_epoch():
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


def get_40ms_gulp():
    npfile=np.load("data/40ms_128chan_gulp_c64.npz")
    meta=np.concatenate((npfile['meta'].ravel() , [npfile['data'].size], npfile['data'].shape))
    _data=npfile["data"].copy()
    for i in range(20):
        print(i,np.min(_data[i,1,:,:]))
    return dict(meta=meta.astype(np.double).ravel().copy(),\
            data=npfile['data'].ravel().copy())


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


