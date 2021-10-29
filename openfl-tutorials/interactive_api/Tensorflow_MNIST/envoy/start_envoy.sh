#!/bin/bash
set -e
ENVOY_NAME=$1
SHARD_CONF=$2

fx envoy start -n "$ENVOY_NAME" --disable-tls --envoy-config-path "$SHARD_CONF" -dh localhost -dp 50051
