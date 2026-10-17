#!/bin/bash
# Run on epic1: correlate the Memo 12 TBF captures with the GPUs of epic2-6 (tbf_tcc_node.sh, two
# captures at a time per node), results to epic1:/data/streamcal/tbf_full. Captures already done, or
# being correlated on epic1, are skipped; ALLDONE is appended to the log when all 32 are in.
#   PY=<python with lsl, numpy, astropy> tbf_tcc_dispatch.sh
# Nodes are addressed on the 100G network (epic1's /etc/hosts lacks epic5).
L=/data/streamcal; S=$(cd "$(dirname "$0")" && pwd)
inprog=$(ps -eo args= | awk '/research\/tbf_tcc.py/ && !/awk/ {print $4}' | xargs -n1 basename 2>/dev/null)
todo=()
for f in /FileStore/epic0/memo12_Data/rawfiles/*_DeMux; do
  b=$(basename $f)
  [ -f $L/tbf_full/$b.npz ] && continue
  echo "$inprog" | grep -qx "$b" && continue
  todo+=($f)
done
echo "in progress on epic1: $(echo $inprog | wc -w); to dispatch: ${#todo[@]}" | tee -a $L/tbf_full.log
nodes=(epic2 epic3 epic4 epic5 epic6)
declare -A addr=([epic2]=192.168.40.22 [epic3]=192.168.40.23 [epic4]=192.168.40.24 [epic5]=192.168.40.25 [epic6]=192.168.40.26)
declare -A work=([epic2]=/data/streamcal [epic3]=/data/streamcal [epic4]=/data/streamcal [epic5]=/var/tmp/streamcal [epic6]=/data/streamcal)
# two streams per node: slot k of node i takes every (2*5)th file
for i in "${!nodes[@]}"; do
  for k in 0 1; do
    mine=(); for j in "${!todo[@]}"; do [ $(( j % 10 )) -eq $(( i * 2 + k )) ] && mine+=(${todo[$j]}); done
    [ ${#mine[@]} -gt 0 ] && ssh -o BatchMode=yes -o HostKeyAlias=${nodes[$i]} ${addr[${nodes[$i]}]} "PY=${PY:-python} bash $S/tbf_tcc_node.sh ${work[${nodes[$i]}]} ${mine[*]}" >> $L/tbf_full.log 2>&1 &
  done
done
wait
while [ $(ls $L/tbf_full/*_DeMux.npz | wc -l) -lt 32 ]; do sleep 30; done
echo ALLDONE >> $L/tbf_full.log
