#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls --envoy-config-path envoy_config_one.yaml -dh nnlicv901.inn.intel.com -dp 50051
