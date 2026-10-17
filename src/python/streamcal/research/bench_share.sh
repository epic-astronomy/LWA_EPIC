#!/bin/bash
# GPU sharing test: epic++ --bench and the Tensor-Core Correlator load generator on one GPU.
#   bench_share.sh NAME PACE(realtime|unpaced) TCC_PERIOD_MS(0 = back to back, - = no TCC)
NAME=$1; PACE=$2; PER=$3
ROOT=/FileStore/epicT/epic/LWA_EPIC_streamcal
OUT=/dev/shm/streamcal_bench/share_$NAME; rm -rf $OUT; mkdir -p $OUT
eval "$(/home/epic/anaconda/bin/conda shell.bash hook)"; conda activate epic310
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 PYTHONWARNINGS=ignore PYTHONPATH=$ROOT/src/python
RT=""; [ "$PACE" = realtime ] && RT="--bench_realtime"
cd $ROOT/build-streamcal
./epic++ --bench --bench_ngulps 2400 --bench_data /dev/shm/streamcal_bench/gulps.bin $RT --bench_report 240 \
  --imagesize 128 --imageres 1.056 --nts 1000 --seq_accum 80 --channels 128 --support 5 --aeff 25 \
  --accum_16bit --nstreams 8 --kernel_oversample 4 --wproj --remove_autocorr --ngpus 1 --gpu_ids 0 \
  --disable_metrics > $OUT/epic.log 2>&1 &
EPIC=$!
until grep -q "Using w-projected" $OUT/epic.log; do sleep 1; done
sleep 12  # past the first statistics block (startup)
if [ "$PER" != "-" ]; then
  $ROOT/tests/streamcal/tcc/tcc_load 40 $PER 128 256 1024 ${COPY:-0} > $OUT/tcc.log 2>&1
fi
wait $EPIC
echo "== $NAME ($PACE, TCC period $PER ms)"
grep -E "bench gen: [0-9]+ gulps" $OUT/epic.log | tail -n 8 | sed -E 's/.*bench gen: //' | cut -c1-150
grep -E "bench img:" $OUT/epic.log | tail -n 4 | sed -E 's/.*bench img: //' | cut -c1-120
cat $OUT/tcc.log 2>/dev/null
