#!/usr/bin/env python
from epic import data_interface as DI
import argparse
import numpy as np

a = argparse.ArgumentParser(description='Convert batch of npz files to fits')
a.add_argument('files', metavar='files', type=str, nargs='*', default=[],
               help='*.npz files to convert to fits.')
args = a.parse_args()

for f in args.files:
    d = np.load(f, allow_pickle=True)
    of = f[:-3] + 'fits'
    DI.epic2fits(of, d['image'], np.ravel(d['hdr'])[0], d['image_nums'])
