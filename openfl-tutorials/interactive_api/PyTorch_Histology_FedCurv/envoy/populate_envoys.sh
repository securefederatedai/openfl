#!/bin/bash
DIRECTOR_HOST=${1:-'localhost'}
DIRECTOR_PORT=${2:-'50051'}
PYTHON=${3:-'python3.8'}

for i in {1..8}
do
    mkdir $i
    cd $i
    echo "shard_descriptor:
    template: histology_shard_descriptor.HistologyShardDescriptor
    params:
        data_folder: histology_data
        rank_worldsize: $i,8
" > envoy_config.yaml

    eval ${PYTHON} '-m venv venv'
    echo "source venv/bin/activate
    pip install ../../../../.. # install OpenFL
    pip install -r requirements.txt
    fx envoy start -n env_$i --disable-tls --envoy-config-path envoy_config.yaml -dh ${DIRECTOR_HOST} -dp ${DIRECTOR_PORT}
    " > start_envoy.sh
    cp ../requirements.txt .
    cp ../histology_shard_descriptor.py .
    cd ..
done
