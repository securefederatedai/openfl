#!/bin/bash
set -e

cd director
bash start_director.sh &
PID=$!

sleep 3
if ! ps -p $PID > /dev/null
then
  echo 'Error: failed to create director'
  exit 1
fi


cd ../envoy
pip install -r sd_requirements.txt
bash start_envoy.sh &
PID=$!
sleep 3
if ! ps -p $PID > /dev/null
then
  echo 'Error: failed to create envoy'
  exit 1
else
  echo "Found $PID in $(ps -p $PID)"
fi


cd ../../../../../..
python -m tests.github.interactive_api_director.experiments.tensorflow_mnist.experiment