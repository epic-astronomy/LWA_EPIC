import datetime
import time
from MCS2 import Communicator


DATE_FORMAT = "%Y_%m_%dT%H_%M_%S"
FS = 196.0e6
CHAN_BW = 25.0e3
ADP_EPOCH = datetime.datetime(1970, 1, 1)


def get_ADP_time_from_unix_epoch():
    """
    Get time in seconds from the unix epoch using the UTC start timestamp from
    the ADP service
    """
    got_utc_start = False
    exception_count=0
    while not got_utc_start:
        try:
            with Communicator("MCS") as adp_control:
                utc_start = adp_control.report("UTC_START")
                # Check for valid timestamp
                utc_start_dt = datetime.datetime.strptime(
                    utc_start, DATE_FORMAT
                )
            got_utc_start = True
        except Exception as ex:
            exception_count += 1
            if exception_count > 10:
                raise ex
            time.sleep(0.1)
    # print((utc_start_dt - ADP_EPOCH).total_seconds())
    return (utc_start_dt - ADP_EPOCH).total_seconds()