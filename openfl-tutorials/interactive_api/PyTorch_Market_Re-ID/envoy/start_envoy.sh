#!/bin/bash
set -e
ENVOY_NAME=$1
SHARD_CONF=$2
: "${ENVOY_NAME:=env_one}"
: "${SHARD_CONF:=shard_config_one.yaml}"

fx envoy start -n "$ENVOY_NAME" --disable-tls -dh localhost -dp 50051  -sc "$SHARD_CONF"
