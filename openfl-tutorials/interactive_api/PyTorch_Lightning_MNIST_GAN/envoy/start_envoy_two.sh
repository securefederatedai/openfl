#!/bin/bash
set -e

fx envoy start -n env_two --disable-tls --envoy-config-path envoy_config_two.yaml -dh localhost -dp 50050
