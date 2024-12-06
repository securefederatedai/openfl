#!/bin/bash
set -e

TAG=${1:-'openfl'}
REPO=https://github.com/securefederatedai/openfl.git
REVISION=develop
echo "Using OpenFL: ${REPO}@${REVISION}"

docker build \
-t ${TAG} \
--build-arg OPENFL_REVISION=${REPO}@${REVISION} \
-f openfl-docker/Dockerfile.base .

echo "Successfully built OpenFL base docker image: ${TAG}"
