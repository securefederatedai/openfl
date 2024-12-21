#!/bin/bash
set -e
ENVOY_NAME=$1
ENVOY_CONF=$2

fx envoy start -n "$ENVOY_NAME" --disable-tls -dh localhost -dp 50050
