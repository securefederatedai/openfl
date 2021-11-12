#!/bin/bash
set -m

FEDERATION ?= fed_work12345alpha81671

docker build -t openfl_base:pre_sgx -f Dockerfile.base ..

docker build -t openfl_gramine -f Dockerfile.gramine ..

# docker build -t ${FEDERATION} -f Dockerfile.workspace \
#     --build-arg BASE_IMAGE=openfl_gramine \
#     --build-arg WORKSPACE_NAME=${FEDERATION} \
#     ${FEDERATION} 

# # aggregator
# docker run -it --rm --network=host --device=/dev/sgx_enclave --device=/dev/sgx_provision \
#     --volume=/var/run/aesmd:/var/run/aesmd \
#     --volume=/home/idavidyu/openfl/openfl-docker/${FEDERATION}/cert:/home/user/workspace/cert \
#     --volume=/home/idavidyu/openfl/openfl-docker/${FEDERATION}/plan:/home/user/workspace/plan \
#     --volume=/home/idavidyu/openfl/openfl-docker/${FEDERATION}/save:/home/user/workspace/save \
#     ${FEDERATION} fx aggregator start

# # collaborator


docker run -it --rm --network=host --device=/dev/sgx_enclave --device=/dev/sgx_provision \
    --volume=/var/run/aesmd:/var/run/aesmd \
    --volume=/home/idavidyu/openfl/openfl-docker/sample_app:/sample_app \
    openfl_gramine bash
