#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls --envoy-config-path shard_config.yaml -dh localhost -dp 50051