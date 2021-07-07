#!/usr/bin/env python

import LWA_bifrost

def test_args():
    args = LWA_bifrost.args_maker()  # This will give you the defaults
    args.offline = True  # manually set the offline argument
    LWA_bifrost.main(args)

	


