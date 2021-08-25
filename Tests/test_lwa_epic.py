#!/usr/bin/env python


import LWA_EPIC
import glob
import os

def test_default_run(tmpdir):
    args, parser = LWA_EPIC.gen_args(return_parser=True)  # This will give you the defaults
    args.offline = True  # manually set the offline argument
    args.tbnfile = "/data5/LWA_SV_data/data_raw/TBN/Jupiter/058161_000086727"
    args.imagesize = 64
    args.imageres = 1.79057
    args.nts = 512
    args.channels = 4
    args.accumulate = 50
    args.ints_per_file = 40
    args.out_dir = tmpdir
    LWA_EPIC.main(args, parser)

    fileList = glob.glob(os.path.join(tmpdir, '*.fits'))

    assert len(fileList) == 24
