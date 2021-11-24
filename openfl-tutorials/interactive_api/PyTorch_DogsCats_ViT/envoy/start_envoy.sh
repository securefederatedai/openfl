#!/bin/bash
set -e
sh check_dataset.sh
fx envoy start -n env_one --disable-tls --envoy-config-path envoy_config_one.yaml -dh 'localhost' -dp 50051
