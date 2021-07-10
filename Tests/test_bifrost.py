#!/usr/bin/env python


from LWA import LWA_bifrost
import os, os.path

def test_args():
    args = LWA_bifrost.args_maker()  # This will give you the defaults
    args.offline = True  # manually set the offline argument
    args.tbnfile = "/data5/LWA_SV_data/data_raw/TBN/Jupiter/058161_000086727"
    args.imagesize = 64
    args.imageres = 1.79057
    args.nts = 512
    args.channels = 4
    args.accumulate = 50
    args.ints_per_file = 40
    LWA_bifrost.main(args)
    path = "/LWA_EPIC/Tests/*.npz"
    assert len([name for name in os.listdir(path) if os.path.isfile(name)]) == 24

	


