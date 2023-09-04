#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls --envoy-config-path envoy_config_no_gpu.yaml -dh localhost -dp 50050
