#!/bin/bash
set -e

cd 1 && CUDA_VISIBLE_DEVICES=0 bash start_envoy.sh &
cd 2 && CUDA_VISIBLE_DEVICES=1 bash start_envoy.sh &
cd 3 && CUDA_VISIBLE_DEVICES=2 bash start_envoy.sh &
cd 4 && CUDA_VISIBLE_DEVICES=3 bash start_envoy.sh &
cd 5 && CUDA_VISIBLE_DEVICES=0 bash start_envoy.sh &
cd 6 && CUDA_VISIBLE_DEVICES=1 bash start_envoy.sh &
cd 7 && CUDA_VISIBLE_DEVICES=2 bash start_envoy.sh &
cd 8 && CUDA_VISIBLE_DEVICES=3 bash start_envoy.sh &
