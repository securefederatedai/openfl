#!/bin/bash
set -e

fx envoy start -n env_two --disable-tls --envoy-config-path envoy_config.yaml -dh localhost -dp 50050