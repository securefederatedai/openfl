#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls --shard-config-path envoy_config.yaml -dh localhost -dp 50051