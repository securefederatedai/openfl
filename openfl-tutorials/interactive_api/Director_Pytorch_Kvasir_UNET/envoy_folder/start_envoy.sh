#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls --shard-config-path shard_config.yaml -d localhost:50051