#!/bin/bash
set -m

cd LibOS/shim/test/regression
make SGX=1
make SGX=1 sgx-tokens
graphene-sgx helloworld