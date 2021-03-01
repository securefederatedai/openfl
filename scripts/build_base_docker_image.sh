#!/bin/bash
set -e

TAG=${1:-'openfl'}

docker build -t ${TAG} \
        -f openfl-docker/Dockerfile.base .
