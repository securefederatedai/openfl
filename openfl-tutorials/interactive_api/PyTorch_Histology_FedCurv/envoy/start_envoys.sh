#!/bin/bash
set -e

cd 1 && bash start_envoy.sh &
cd 2 && bash start_envoy.sh &
cd 3 && bash start_envoy.sh &
cd 4 && bash start_envoy.sh &
cd 5 && bash start_envoy.sh &
cd 6 && bash start_envoy.sh &
cd 7 && bash start_envoy.sh &
cd 8 && bash start_envoy.sh 
