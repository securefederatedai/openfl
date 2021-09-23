#!/bin/bash
set -e
ENVOY_NAME=$1
SHARD_CONFIG_PATH=$2
DIRECTOR_FQDN=$3

fx envoy start -n "$ENVOY_NAME" --shard-config-path "$SHARD_CONFIG_PATH" -dh "$DIRECTOR_FQDN" -dp 50051 -rc cert/root_ca.crt -pk cert/"$ENVOY_NAME".key -oc cert/"$ENVOY_NAME".crt
