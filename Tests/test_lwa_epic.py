#!/usr/bin/env python


from LWA_EPIC import LWA_EPIC
import glob

def test_default_run():
    args, parser = LWA_EPIC.gen_args(return_parser=True)  # This will give you the defaults
    args.offline = True  # manually set the offline argument
    args.tbnfile = "/data5/LWA_SV_data/data_raw/TBN/Jupiter/058161_000086727"
    args.imagesize = 64
    args.imageres = 1.79057
    args.nts = 512
    args.channels = 4
    args.accumulate = 50
    args.ints_per_file = 40
    LWA_EPIC.main(args, parser)

    fileList = glob.glob('*.npz')

    assert len(fileList) == 24
