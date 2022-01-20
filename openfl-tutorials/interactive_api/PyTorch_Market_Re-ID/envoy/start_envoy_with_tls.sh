#!/bin/bash
set -e
ENVOY_NAME=$1
DIRECTOR_FQDN=$2
ENVOY_CONFIG=$3

fx envoy start -n "$ENVOY_NAME" -dh "$DIRECTOR_FQDN" -dp 50051 -ec "$ENVOY_CONFIG" -rc cert/root_ca.crt -pk cert/"$ENVOY_NAME".key -oc cert/"$ENVOY_NAME".crt
