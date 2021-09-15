#!/bin/bash
set -e

fx envoy start -n env_two --disable-tls -dh nnlicv838.inn.intel.com -dp 50051
