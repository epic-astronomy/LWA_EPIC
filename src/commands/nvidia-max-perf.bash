#!/bin/bash

if [[ "$(id -u)" != "0" ]]; then
    echo "Insufficient permisssions. Are you root?"
    exit -1
fi

killall -9 Xorg
rm /tmp/.X0-lock
X :0 &

# see https://unix.stackexchange.com/a/636628
# and http://bailiwick.io/2019/09/21/controlling-nvidia-gpu-fans-on-a-headless-ubuntu-system/
NUM_GPUS=$(DISPLAY=:0 nvidia-settings -q gpus | grep -c 'gpu:')
NUM_FANS=$(DISPLAY=:0 nvidia-settings -q fans | grep -c 'fan:')

echo "Found ${NUM_GPUS} GPUs and ${NUM_FANS} Fans"

# For each GPU, enable fan control.
for i in $(seq 0 1 $(echo "$NUM_GPUS-1" | bc -l)); do
    DISPLAY=:0 nvidia-settings --verbose=all -a "[gpu:$i]/GPUFanControlState=1"

    DISPLAY=:0 nvidia-settings -a "[gpu:$i]/GPUPowerMizerMode=1" # The "Prefer Maximum Performance" mode

    DISPLAY=:0 nvidia-smi -i $i -pl 600

done

# For each fan, set fan speed to 100%.
for i in $(seq 0 1 $(echo "$NUM_FANS-1" | bc -l)); do
    DISPLAY=:0 nvidia-settings --verbose=all -a "[fan:$i]/GPUTargetFanSpeed=100"

done

# Give some time for the fans to spin up
echo "Waiting for the GPU fans to spin up..."
sleep 5

# Clock the GPUs to maximum
nvidia-smi -pm 1
nvidia-smi -lgc 3105
nvidia-smi -lmc 10501

# try to clock it more
for i in $(seq 0 1 $(echo "$NUM_GPUS-1" | bc -l)); do
    DISPLAY=:0 nvidia-settings -a "[gpu:$i]/GPUGraphicsClockOffsetAllPerformanceLevels=75"
done

# Print the status
nvidia-smi --query-gpu=index,gpu_name,persistence_mode,clocks.sm,clocks.mem,temperature.gpu,fan.speed,power.draw,pstate --format=csv

# Note: Although the performance state will be set to P0 after the initial clock locking, it quickly goes to P2 once EPIC starts to run. The reason is as follows:
#(see https://babeltechreviews.com/nvidia-cuda-force-p2-state/)

# […] Basically, we added this p-state because running at max memory clocks for some CUDA applications can cause memory errors when running HUGE datasets. Think DL apps, oil exploration use cases, etc where you are crunching large numbers and it would error out with full memory clocks. These are the types of apps you really shouldn’t be running on GeForce anyway but since there are a lot of folks who do and were running into this issue we created this new mode for them.

killall -9 Xorg