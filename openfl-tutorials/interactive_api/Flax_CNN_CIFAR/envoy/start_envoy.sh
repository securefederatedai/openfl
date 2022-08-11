#!/bin/bash
set -e
ENVOY_NAME=$1
ENVOY_CONF=$2

# export JAX_PLATFORM_NAME=cpu # Set preferred platform for computation - cpu/gpu/tpu.
export XLA_PYTHON_CLIENT_PREALLOCATE=false # Overide default behaviour. Incrementally allocate memory as required
export TF_FORCE_GPU_ALLOW_GROWTH=true

fx envoy start -n "$ENVOY_NAME" --disable-tls --envoy-config-path "$ENVOY_CONF" -dh localhost -dp 50055
