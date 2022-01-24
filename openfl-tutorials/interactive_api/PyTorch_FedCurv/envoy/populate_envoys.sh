#!/bin/bash

for i in {1..10}
do
    mkdir $i
    echo "shard_descriptor:
    template: tinyimagenet_shard_descriptor.TinyImageNetShardDescriptor
    params:
        data_folder: tinyimagenet_data
        rank_worldsize: $i,10
" > $i/envoy_config.yaml
    echo "fx envoy start -n env_$i --disable-tls --envoy-config-path envoy_config.yaml -dh localhost -dp 50051" > $i/start_envoy.sh
    cp requirements.txt $i
    cp base.py $i
    cp tinyimagenet_shard_descriptor.py $i
done