#!/bin/bash
# One worker node (run by tbf_tcc_dispatch.sh): correlate the given TBF captures on this node's GPU and send the results to epic1.
#   tbf_tcc_node.sh WORKDIR FILE [FILE ...]   (env PY: python with lsl, numpy, astropy)
# Raw files are streamed from epic-sv over the 100G network (12x faster than the 1 GbE NFS
# mount), kept on local disk only while being correlated, then deleted.
W=$1; shift
mkdir -p $W/raw $W/out
O="-o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com"
cd "$(dirname "$0")/../../../.."
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
for f in "$@"; do
  b=$(basename $f)
  ssh $O -o HostKeyAlias=epic0 192.168.40.20 "cat $f" > $W/raw/$b
  nice ${PY:-python} src/python/streamcal/research/tbf_tcc.py $W/raw/$b $W/out
  rm -f $W/raw/$b
  scp -q $O -o HostKeyAlias=epic1 $W/out/$b.npz 192.168.40.21:/data/streamcal/tbf_full/$b.npz.part && \
    ssh $O -o HostKeyAlias=epic1 192.168.40.21 "mv /data/streamcal/tbf_full/$b.npz.part /data/streamcal/tbf_full/$b.npz" && \
    rm -f $W/out/$b.npz
  echo "$(hostname) done $b"
done
