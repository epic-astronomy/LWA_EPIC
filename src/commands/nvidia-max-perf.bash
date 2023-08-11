#!/bin/bash

nvidia-smi -pm 1
nvidia-smi -lgc 3105
nvidia-smi -lmc 10501