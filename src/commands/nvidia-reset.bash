#!/bin/bash

if [[ "$(id -u)" != "0" ]]; then
  echo "Insufficient permisssions. Are you root?"
  exit -1
fi

killall Xorg
rm /tmp/.X0-lock
X :0 &

# see https://unix.stackexchange.com/a/636628
# and http://bailiwick.io/2019/09/21/controlling-nvidia-gpu-fans-on-a-headless-ubuntu-system/
NUM_GPUS=$(DISPLAY=:0 nvidia-settings -q gpus | grep -c 'gpu:')
NUM_FANS=$(DISPLAY=:0 nvidia-settings -q fans | grep -c 'fan:')

echo "Found ${NUM_GPUS} GPUs and ${NUM_FANS} Fans"

# For each GPU, disable fan control, reset power limit and power mizer mode.
for i in $(seq 0 1 $(echo "$NUM_GPUS-1" | bc -l)); do
  DISPLAY=:0 nvidia-settings --verbose=all -a "[gpu:$i]/GPUFanControlState=0"

  DISPLAY=:0 nvidia-settings --verbose=all -a "[gpu:$i]/GPUPowerMizerMode=0"

  DISPLAY=:0 nvidia-smi -i $i -pl 450

done

# Clock the GPU clocks
nvidia-smi -rgc
nvidia-smi -rmc

echo "Waiting for GPU spin down..."
sleep 4
# Print the status
nvidia-smi --query-gpu=index,gpu_name,persistence_mode,clocks.sm,clocks.mem,temperature.gpu,fan.speed,power.draw,pstate --format=csv
killall -9 Xorg