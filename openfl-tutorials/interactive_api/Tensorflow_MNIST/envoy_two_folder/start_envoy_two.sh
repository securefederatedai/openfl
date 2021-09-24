#!/bin/bash
set -e

fx envoy start -n env_two --disable-tls --shard-config-path shard_config_two.yaml -dh localhost -dp 50051
