#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls --shard-config-path shard_config_one.yaml -dh localhost -dp 50051
