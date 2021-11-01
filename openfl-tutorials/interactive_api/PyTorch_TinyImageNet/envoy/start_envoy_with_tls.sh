#!/bin/bash
set -e
ENVOY_NAME=$1
DIRECTOR_FQDN=$2

fx envoy start -n "$ENVOY_NAME" --envoy-config-path envoy_config.yaml -dh "$DIRECTOR_FQDN" -dp 50051 -rc cert/root_ca.crt -pk cert/"$ENVOY_NAME".key -oc cert/"$ENVOY_NAME".crt