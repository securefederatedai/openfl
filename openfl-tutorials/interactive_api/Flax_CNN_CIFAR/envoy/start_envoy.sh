#!/bin/bash
set -e
ENVOY_NAME=$1
ENVOY_CONF=$2

DEFAULT_DEVICE='CPU'

if [[ $DEFAULT_DEVICE == 'CPU' ]]
then
    export JAX_PLATFORMS="cpu" # Force XLA to use CPU
    export CUDA_VISIBLE_DEVICES='-1' # Force TF to use CPU
else
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export TF_FORCE_GPU_ALLOW_GROWTH=true
fi

fx envoy start -n "$ENVOY_NAME" --disable-tls --envoy-config-path "$ENVOY_CONF" -dh localhost -dp 50055
