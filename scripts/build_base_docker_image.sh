#!/bin/bash
set -e

docker build -t openfl \
        -f openfl-docker/Dockerfile.base .
