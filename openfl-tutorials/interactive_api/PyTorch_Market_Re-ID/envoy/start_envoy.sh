#!/bin/bash
set -e
ENVOY_NAME=$1
SHARD_CONF=$2

fx envoy start -n "$ENVOY_NAME" --disable-tls -dh localhost -dp 50051  -sc "$SHARD_CONF"
