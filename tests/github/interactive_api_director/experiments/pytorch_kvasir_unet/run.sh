#!/bin/bash
set -e

cd director
fx director start --disable-tls -c director_config.yaml &
PID=$!

sleep 3
if ! ps -p $PID > /dev/null
then
  echo 'Error: failed to create director'
  exit 1
fi


cd ../envoy
pip install -r requirements.txt
fx envoy start -n env_one --disable-tls --shard-config-path shard_config_one.yaml -dh localhost -dp 50051 &
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
python -m tests.github.interactive_api_director.experiments.pytorch_kvasir_unet.experiment
