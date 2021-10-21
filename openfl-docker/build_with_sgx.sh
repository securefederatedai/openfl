#!/bin/bash
set -m

docker build -t openfl_base:pre_sgx -f Dockerfile.base ..

docker build -t openfl_graphene -f Dockerfile.graphene ..

docker run -it --network=host --device=/dev/sgx_enclave --device=/dev/sgx_provision openfl_graphene bash

# cd LibOS/shim/test/regression \
# make SGX=1 \
# make SGX=1 sgx-tokens \
# graphene-sgx helloworld
