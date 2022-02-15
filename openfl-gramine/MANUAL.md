# OpenFL + Gramine
This manual will help you run OpenFL with Aggregator-based workflow inside SGX enclave with Gramine.
## TO-DO:
- [X] import manifest and makefile from dist-package openfl 
- [X] pass wheel repository to pip (for cpu versions of pytorch for example)
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
- Docker should be installed, user included into Docker group

Machines that will run an Aggregator and Collaborator containers should have the following:
- SGX enebled in BIOS
- Ubuntu with Linux kernel 5.11+
- ? SGX drivers, it is built in kernel: `/dev/sgx_enclave`
- [aesmd service](https://github.com/intel/linux-sgx) (`/var/run/aesmd/aesm.socket`)
This is a short list, see more in Gramine docs.

## Workflow
The user will mainly interact with OpenFL CLI, docker CLI and other commandline tools. But the user is also expected to modify plan.yaml file and Python code under workspace/src folder to set up an FL Experiment.
### On the building machine (Data Scientist's node):
1. As usual, **create a workspace**: 
```
fx workspace create --prefix WORKSPACE_NAME --template TEMPLATE_NAME
cd WORKSPACE_NAME
```
Modify the code and the plan.yaml, set up your training procedure. </br>
Pay attention to the following: 
- make sure data loading code reads data from ./data folder inside the workspace
- if you download data (developement scenario) make sure your code first checks if data exists, as connecting to the internet from an enclave may be problematic.
- make sure you do not use any CUDA driver-dependent packages

2. **Initialize the experiment plan** </br> 
Find out the FQDN of the aggregator machine and use for plan initialization.
```
fx plan initialize -a ${FQDN}
```
To find out FQDN (Unix-like OS) try `hostname --all-fqdns | awk '{print $1}'` command.

3. (Optional) **Generate a signing key** on building machine if you do not have one.</br>
It will be used to calculate hashes of trusted files. If you plan to test the application without SGX (gramine-direct) you also do not need a signer key.
```
openssl genrsa -3 -out KEY_LOCATION/key.pem 3072
```
This key will not be packed to the final Docker image.

4. **Build the Experiment Docker image**

```
fx workspace dockerize -s KEY_LOCATION/key.pem --sgx-target/--no-sgx-target
```
This command will build and save a Docker image with you Experiment. The saved image will contain all the required files to start a process in an enclave.</br>
If `--no-sgx-target` option passed to the command, the image will run processes under gramine-direct, in this case it is not necessary to pass signing key.


### Image distribution:
Data scientist now must transfer the Docker image to the aggregator and collaborator machines. Aggregator will also need initial model weights.

5. **Transfer files** to the aggregator an collaborator machines.
If there is a connaction between machines, you may use `scp`. In other case use the transfer channel that suits your situation.</br>
Send files to the aggregator machine:
```
scp DATA_SCIENTIST_MACHINE:WORKSPACE_PATH/WORKSPACE_NAME.tar.gz AGGREGATOR_MACHINE:SOME_PATH
scp DATA_SCIENTIST_MACHINE:WORKSPACE_PATH/save/WORKSPACE_NAME_init.pbuf AGGREGATOR_MACHINE:SOME_PATH
```

Send the image archive to collaborator machines:
```
scp DATA_SCIENTIST_MACHINE:WORKSPACE_PATH/WORKSPACE_NAME.tar.gz COLLABORATOR_MACHINE:SOME_PATH
```
### On the running machines (Aggregator and Collaborator nodes):
6. **Load the image.**
Execute the following command on all running machines:
```
docker load < WORKSPACE_NAME.tar.gz
```

7. **Prepare certificates**
Certificates exchange is a big separate topic. To run an experiment following OpenFL Aggregator-based workflow, user must follow the established procedure, please refer to [the docs](https://openfl.readthedocs.io/en/latest/running_the_federation.html#bare-metal-approach).
Following the above-mentioned procedure, running machines will acquire certificates. Moreover, as the result of this procedure, the aggregator machine will also obtain a `cols.yaml` file (required to start an experiment) with registered collaborators names, and the collaborator machines will obtain `data.yaml` files.

We recoment replicate the OpenFL workspace folder structure on all the machines and follow the usual certifying procedure. Finally, on aggregator you should have the following folder structure:
```
workspace/
--save/WORKSPACE_NAME_init.pbuf
--logs/
--plan/cols.yaml
--cert
```

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