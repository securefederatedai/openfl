#!/bin/bash
set -m

docker build -t openfl_base:pre_sgx -f Dockerfile.base ..

docker build -t openfl_graphene -f Dockerfile.graphene ..

docker run -it --network=host --device=/dev/sgx_enclave --device=/dev/sgx_provision openfl_graphene bash
