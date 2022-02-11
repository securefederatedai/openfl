# OpenFL + Gramine
This manual will help you run OpenFL with Aggregator-based workflow under Gramine SGX.
## TO-DO:
- import manifest and makefile from dist-package openfl 
- pass wheel repository to pip (for cpu versions of pytorch for example)
- get rid of command line args (insecure)
## Known issues:
- Kvasir experiment: aggregation takes really long, debug log-level does not show the reason
- We need workspace zip to import it and create certs. We need to know number of collaborators prior to zipping the workspace. SOLUTOIN: mount cols.yaml and data.yaml
- During plan initialization we need data to initialize the model. so at least one collaborator should be in data.yaml and its data should be available. cols.yaml may be empty at first
During cert sign request generation cols.yaml on collaborators remain empty, data.yaml is extended if needed. On aggregator, cols.yaml are updated during signing procedure, data.yaml remains unmodified
- `error: Disallowing access to file '/usr/local/lib/python3.8/__pycache__/signal.cpython-38.pyc.3423950304'; file is not protected, trusted or allowed.`
## Prerquisites
Building machine:
- OpenFL

Machines that will run an Aggregator and Collaborator containers should have the following:
- SGX enebled in BIOS
- Ubuntu with Linux kernel 5.11+
- ? SGX drivers, it is built in kernel: `/dev/sgx_enclave`
- [aesmd service](https://github.com/intel/linux-sgx) (`/var/run/aesmd/aesm.socket`)
This is a short list, see more in Gramine docs.

## Workflow
1. Build the base image with openfl installed from pip and gramine from apt:
```
DOCKER_BUILDKIT=1 docker build -t gramine_openfl -f openfl-gramine/Dockerfile.gramine .
```

2. Create a workspace. Modify code and the plan.yaml.
- make sure data loading code reads data from ./data folder inside the workspace
- make sure you do not use any CUDA driver-dependent packages
- Find out the FQDN of the aggregator machine and use during plan initialization

3. Do `fx plan initialize -a ${FQDN}`

4. Do `fx workspace export`

4.5 cd one folder up

5. Generate a signer key on building machine to hash trusted files
```
openssl genrsa -3 -out ./key.pem 3072
```
6. Build dockerized workspace and gramine app
```
EXP_NAME=kvasir
FEDERATION=fed_work12345alpha81671
DOCKER_BUILDKIT=1
docker build -t ${EXP_NAME} \
--build-arg WORKSPACE_ARCHIVE=${FEDERATION}/${FEDERATION}.zip \
--secret id=signer-key,src=./key.pem \
-f ./openfl-gramine/Dockerfile.graminized.workspace . 
```

7. Transfer the image to the aggregator an collaborator machines
```
EXP_NAME=kvasir
docker save ${EXP_NAME} | gzip > ${EXP_NAME}.tar.gz
```

transfer file:
```
scp idavidyu@nnlicv674.inn.intel.com:/home/idavidyu/openfl/kvasir.tar.gz .
```

on the running machines:
```
EXP_NAME=kvasir
docker load < ${EXP_NAME}.tar.gz
```
8. Transfer initial model weights from save/ directory to aggregator machine.

9. Register collaborators. Signing requests process is coupled with modifying cols.yaml on the aggregator machine and data.yaml files in collaborator machines
In the end of this step you should have signed certificates on all the machines, cols.yaml file on aggregator 
and data.yaml files on collaborators.

10. Run aggregator
```
EXP_NAME=kvasir
FEDERATION_PATH=/home/idavidyu/openfl-rebased-gramine/openfl/fed_work12345alpha81671
docker run -it --rm --device=/dev/sgx_enclave --volume=/var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
--network=host \
--volume=${FEDERATION_PATH}/cert:/workspace/cert \
--volume=${FEDERATION_PATH}/logs:/workspace/logs \
--volume=${FEDERATION_PATH}/plan/cols.yaml:/workspace/plan/cols.yaml \
--mount type=bind,src=${FEDERATION_PATH}/save,dst=/workspace/save,readonly=0 \
${EXP_NAME} aggregator start
```

No SGX (gramine-direct):

```
EXP_NAME=kvasir
FEDERATION=fed_work12345alpha81671
docker run -it --rm --security-opt seccomp=unconfined \
--network=host \
--volume=/home/idavidyu/openfl/${FEDERATION}/cert:/workspace/cert \
--volume=/home/idavidyu/openfl/${FEDERATION}/logs:/workspace/logs \
--mount type=bind,src=/home/idavidyu/openfl/${FEDERATION}/save,dst=/workspace/save,readonly=0 \
${EXP_NAME} aggregator start
```
11. Run collaborator
```
EXP_NAME=kvasir
FEDERATION=fed_work12345alpha81671
COL_NAME=two
COL_NAME=one
docker run -it --rm --device=/dev/sgx_enclave --volume=/var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
--network=host \
--volume=/home/idavidyu/openfl-rebased-gramine/openfl/${FEDERATION}/${COL_NAME}/${FEDERATION}/cert:/workspace/cert \
--volume=/home/idavidyu/openfl-rebased-gramine/openfl/${FEDERATION}/${COL_NAME}/${FEDERATION}/plan/data.yaml:/workspace/plan/data.yaml \
--volume=/home/idavidyu/openfl-rebased-gramine/openfl/${FEDERATION}/data:/workspace/data \
${EXP_NAME} collaborator start -n ${COL_NAME}
```