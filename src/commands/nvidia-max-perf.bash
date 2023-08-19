#!/bin/bash

if [[ "$(id -u)" != "0" ]]
  then echo "Insufficient permisssions. Are you root?"
  exit
fi

#if the fan control commands fail as "Error assigning value XX to attribute 'GPUTargetFanSpeed'", check if the file /etc/X11/Xwrapper.config has the following
# lines
#-----
# allowed_users=anybody
# needs_root_rights=yes
#----
# If not create it and restart the X-server before running this script
# see https://www.systutorials.com/docs/linux/man/5-Xwrapper.config/
# for a description of these lines
if [ ! -f "/etc/X11/Xwrapper.config" ];then
    echo "The file /etc/X11/Xwrapper.config does not exist."
    echo "Create this file and add the following lines before re-running this script"
    echo "allowed_users=anybody"
    echo "needs_root_rights=yes"
    exit 1
fi

uid="$(id -u gdm)"
# see https://unix.stackexchange.com/a/636628
# and http://bailiwick.io/2019/09/21/controlling-nvidia-gpu-fans-on-a-headless-ubuntu-system/
NUM_GPUS=$(DISPLAY=:0 XAUTHORITY=/run/user/${uid}/gdm/Xauthority nvidia-settings -q gpus | grep -c 'gpu:')
NUM_FANS=$(DISPLAY=:0 XAUTHORITY=/run/user/${uid}/gdm/Xauthority nvidia-settings -q fans | grep -c 'fan:')

echo "Found ${NUM_GPUS} GPUs and ${NUM_FANS} Fans"

# For each GPU, enable fan control.
for i in $(seq 0 1 $(echo "$NUM_GPUS-1" | bc -l));
do
    DISPLAY=:0 XAUTHORITY=/run/user/${uid}/gdm/Xauthority nvidia-settings --verbose=all -a "[gpu:$i]/GPUFanControlState=1"
done

# For each fan, set fan speed to 100%.
for i in $(seq 0 1 $(echo "$NUM_FANS-1" | bc -l));
do
    DISPLAY=:0 XAUTHORITY=/run/user/${uid}/gdm/Xauthority nvidia-settings --verbose=all -a "[fan:$i]/GPUTargetFanSpeed=30"
done

# Give some time for the fans to spin up
echo "Waiting for the GPU fans to spin up..."
sleep 5

# Clock the GPUs to maximum
nvidia-smi -pm 1
nvidia-smi -lgc 3105
nvidia-smi -lmc 10501

# Print the status
nvidia-smi --query-gpu=index,gpu_name,persistence_mode,clocks.sm,clocks.mem,temperature.gpu,fan.speed,power.draw,pstate --format=csv