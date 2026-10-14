import datetime
import time
from MCS2 import Communicator


DATE_FORMAT = "%Y_%m_%dT%H_%M_%S"
FS = 196.0e6
CHAN_BW = FS / 8192  # NDP channel width
ADP_EPOCH = datetime.datetime(1970, 1, 1)


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