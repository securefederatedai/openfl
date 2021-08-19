#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls -d localhost:50051
