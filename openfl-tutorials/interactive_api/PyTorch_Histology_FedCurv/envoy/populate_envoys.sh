#!/bin/bash

for i in {1..8}
do
    mkdir $i
    echo "shard_descriptor:
    template: histology_shard_descriptor.HistologyShardDescriptor
    params:
        data_folder: histology_data
        rank_worldsize: $i,8
" > $i/envoy_config.yaml
    echo "fx envoy start -n env_$i --disable-tls --envoy-config-path envoy_config.yaml -dh nnlicv431.inn.intel.com -dp 50053" > $i/start_envoy.sh
    cp requirements.txt $i
    cp histology_shard_descriptor.py $i
done
