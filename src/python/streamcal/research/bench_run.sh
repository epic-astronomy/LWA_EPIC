#!/bin/bash
# Imager + calibrator benchmark on one GPU node (no F-engine, database, disk or video).
#   bench_run.sh NAME NGULPS [extra epic++ options...]
# Environment: CAL=1 runs the calibration tap, streamcal and the watcher; GPUTAP=1 makes that tap
# the tensor-core correlator on the GPU (--caltap_gpu) instead of the CPU slice; DATA=path of
# simulated gulps (empty: zero gulps); ORACLE=1 publishes the exact gain table instead of running
# streamcal. PROJ=1 has streamcal publish the interference projector and epic++ apply it
# (--proj_table); IMPULSE=sigma enables the impulse detector; DAEMON_CPUS and WORKERS place the
# daemon (default 12-23 and 8); WATCH=0 skips the gain watcher (datasets without true gains, e.g.
# make_table_gulps.py).
set -u
NAME=$1; NGULPS=$2; shift 2
ROOT=/FileStore/epicT/epic/LWA_EPIC_streamcal
BENCH=/FileStore/epicT/epic/streamcal_bench
MODEL=${MODEL:-$BENCH/model75}
OUT=/dev/shm/streamcal_bench/$NAME
mkdir -p $OUT
rm -f /dev/shm/epic_vis_gpu0 /dev/shm/epic_gains_gpu0 /dev/shm/epic_proj_gpu0
eval "$(/home/epic/anaconda/bin/conda shell.bash hook)"; conda activate epic310
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 PYTHONWARNINGS=ignore PYTHONPATH=$ROOT/src/python
SEQ0=$(python -c "import numpy as np; print(int(np.load('$MODEL/truth.npz')['seq0']))")
TIME=$(python -c "import numpy as np; print(str(np.load('$MODEL/truth.npz')['time']))")
CHAN0=$(python -c "import numpy as np; t = np.load('$MODEL/truth.npz'); print(int(t['chan0']) if 'chan0' in t else 3104)")
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
DOPTS="${DAEMON_OPTS:-}"
if [ "${PROJ:-0}" = 1 ]; then
  CALOPT="$CALOPT --proj_table /dev/shm/epic_proj_gpu{gpu}"
  DOPTS="$DOPTS --proj /dev/shm/epic_proj_gpu0"
fi
if [ -n "${IMPULSE:-}" ]; then
  CALOPT="$CALOPT --impulse_sigma $IMPULSE"
fi
if [ "${ORACLE:-0}" = 1 ]; then
  $BENCH/venv/bin/python $ROOT/src/python/streamcal/research/bench_watch.py $MODEL /dev/shm/epic_gains_gpu0 --oracle
fi
cd $ROOT/build-streamcal
./epic++ --bench --bench_ngulps $NGULPS --bench_seq0 $SEQ0 --bench_chan0 $CHAN0 --bench_data "${DATA:-}" \
  --imagesize 128 --imageres 1.056 --nts 1000 --seq_accum 80 --channels 128 --support 5 --aeff 25 \
  --accum_16bit --nstreams 8 --kernel_oversample 4 --wproj --remove_autocorr --ngpus 1 --gpu_ids 0 \
  --disable_metrics $CALOPT "$@" > $OUT/epic.log 2>&1 &
EPIC=$!
if [ "${CAL:-0}" = 1 ]; then
  ( cd $ROOT/src/python && OMP_NUM_THREADS=${DAEMON_THREADS:-1} MKL_NUM_THREADS=${DAEMON_THREADS:-1} \
    OPENBLAS_NUM_THREADS=${DAEMON_THREADS:-1} taskset -c ${DAEMON_CPUS:-12-23} \
    $BENCH/venv/bin/python -m streamcal.daemon --vis /dev/shm/epic_vis_gpu0 --gains /dev/shm/epic_gains_gpu0 \
      --workers ${WORKERS:-8} --fixed-time $TIME --save $OUT/history.npy $DOPTS > $OUT/daemon.log 2>&1 ) &
  DAEMON=$!
  WPID=""
  if [ "${WATCH:-1}" != 0 ]; then
    $BENCH/venv/bin/python $ROOT/src/python/streamcal/research/bench_watch.py $MODEL /dev/shm/epic_gains_gpu0 > $OUT/watch.log 2>&1 &
    WPID=$!
  fi
fi
wait $EPIC
echo "epic++ exit $?" >> $OUT/epic.log
if [ "${CAL:-0}" = 1 ]; then
  sleep 3
  [ -n "$WPID" ] && kill $WPID 2>/dev/null
  # the daemon by PID (pkill -f would match this shell's own command line)
  for pid in $(ps -eo pid,args | awk '$2 ~ /python/ && /streamcal.daemon --vis \/dev\/shm\/epic_vis_gpu0/ {print $1}'); do
    kill $pid 2>/dev/null
  done
fi
grep -E "bench |bench gen|CalTap|TccTap|RfiStage|Applied calibration|projector|rror|FATAL" $OUT/epic.log | tail -40
