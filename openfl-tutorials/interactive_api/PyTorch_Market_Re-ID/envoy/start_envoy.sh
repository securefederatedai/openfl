#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls -dh localhost -dp 50051 -ec envoy_config_one.yaml
