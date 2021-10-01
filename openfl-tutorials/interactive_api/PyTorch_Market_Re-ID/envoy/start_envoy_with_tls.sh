#!/bin/bash
set -e
DIRECTOR_FQDN=$1
ENVOY_NAME=$2
SHARD_CONF=$3

: "${ENVOY_NAME:=env_one}"
: "${SHARD_CONF:=shard_config_one.yaml}"

fx envoy start -n "$ENVOY_NAME" --shard-config-path "$SHARD_CONF" -d "$DIRECTOR_FQDN":50051 -rc cert/root_ca.crt -pk cert/"$ENVOY_NAME".key -oc cert/"$ENVOY_NAME".crt
