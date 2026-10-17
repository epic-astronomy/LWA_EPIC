#!/bin/bash
# Imager + calibrator benchmark on one GPU node (no F-engine, database, disk or video).
#   bench_run.sh NAME NGULPS [extra epic++ options...]
# Environment: CAL=1 runs the calibration tap, streamcal and the watcher; GPUTAP=1 makes that tap
# the tensor-core correlator on the GPU (--caltap_gpu) instead of the CPU slice; DATA=path of
# simulated gulps (empty: zero gulps); ORACLE=1 publishes the exact gain table instead of running
# streamcal.
set -u
NAME=$1; NGULPS=$2; shift 2
ROOT=/FileStore/epicT/epic/LWA_EPIC_streamcal
BENCH=/FileStore/epicT/epic/streamcal_bench
MODEL=${MODEL:-$BENCH/model75}
OUT=/dev/shm/streamcal_bench/$NAME
mkdir -p $OUT
rm -f /dev/shm/epic_vis_gpu0 /dev/shm/epic_gains_gpu0
eval "$(/home/epic/anaconda/bin/conda shell.bash hook)"; conda activate epic310
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 PYTHONWARNINGS=ignore PYTHONPATH=$ROOT/src/python
SEQ0=$(python -c "import numpy as np; print(int(np.load('$MODEL/truth.npz')['seq0']))")
TIME=$(python -c "import numpy as np; print(str(np.load('$MODEL/truth.npz')['time']))")
CALOPT=""
if [ "${CAL:-0}" = 1 ] || [ "${ORACLE:-0}" = 1 ]; then
  CALOPT="--gain_table /dev/shm/epic_gains_gpu{gpu}"
fi
if [ "${CAL:-0}" = 1 ]; then
  if [ "${GPUTAP:-0}" = 1 ]; then
    CALOPT="$CALOPT --caltap_gpu --caltap_interval 10"
  else
    CALOPT="$CALOPT --caltap_nseq ${TAPNSEQ:-50} --caltap_interval 10 --caltap_threads 2"
  fi
fi
if [ "${ORACLE:-0}" = 1 ]; then
  $BENCH/venv/bin/python $ROOT/src/python/streamcal/research/bench_watch.py $MODEL /dev/shm/epic_gains_gpu0 --oracle
fi
cd $ROOT/build-streamcal
./epic++ --bench --bench_ngulps $NGULPS --bench_seq0 $SEQ0 --bench_data "${DATA:-}" \
  --imagesize 128 --imageres 1.056 --nts 1000 --seq_accum 80 --channels 128 --support 5 --aeff 25 \
  --accum_16bit --nstreams 8 --kernel_oversample 4 --wproj --remove_autocorr --ngpus 1 --gpu_ids 0 \
  --disable_metrics $CALOPT "$@" > $OUT/epic.log 2>&1 &
EPIC=$!
if [ "${CAL:-0}" = 1 ]; then
  ( cd $ROOT/src/python && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 12-23 \
    $BENCH/venv/bin/python -m streamcal.daemon --vis /dev/shm/epic_vis_gpu0 --gains /dev/shm/epic_gains_gpu0 \
      --fixed-time $TIME --save $OUT/history.npy ${DAEMON_OPTS:-} > $OUT/daemon.log 2>&1 ) &
  DAEMON=$!
  $BENCH/venv/bin/python $ROOT/src/python/streamcal/research/bench_watch.py $MODEL /dev/shm/epic_gains_gpu0 > $OUT/watch.log 2>&1 &
  WATCH=$!
fi
wait $EPIC
echo "epic++ exit $?" >> $OUT/epic.log
if [ "${CAL:-0}" = 1 ]; then
  sleep 3
  kill $WATCH 2>/dev/null; pkill -f "streamcal.daemon --vis /dev/shm/epic_vis_gpu0" 2>/dev/null
fi
grep -E "bench |bench gen|CalTap|TccTap|Applied calibration|rror|FATAL" $OUT/epic.log | tail -40
