# Open(FL)ower

This workspace demonstrates a new functionality in OpenFL to interoperate with [Flower](https://flower.ai/). In particular, a user can now use the Flower API to run on OpenFL infrastructure. OpenFL will act as an intermediary step between the Flower SuperLink and Flower SuperNode to relay messages across the network using OpenFL's transport mechanisms.

## Overview

In this repository, you'll notice a directory under `src` called `app-pytorch`. This is essentially a Flower PyTorch app created using Flower's `flwr new` command that has been modified to run a local federation. The `client_app.py` and `server_app.py` dictate what will be run by the client and server respectively. `task.py` defines the logic that will be executed by each app, such as the model definition, train/test tasks, etc. Under `server_app.py` a save model strategy is defined in order to save the best and last models from the experiment in your local workspace under `./save`.

## Getting Started

### Install OpenFL

Follow the [installation guide](https://openfl.readthedocs.io/en/latest/installation.html).

### Create a Workspace

Start by creating a workspace:

```sh
fx workspace create --template flower-app-pytorch --prefix my_workspace
cd my_workspace
pip install -r requirements.txt
```

Then create a certificate authority (CA)

```sh
fx workspace certify
```

This will create a workspace in your current working directory called `./my_workspace` as well as install the Flower app defined in `./src/app-pytorch.` This will be where the experiment takes place. The CA will be used to sign the certificates of the collaborators.

### Setup Data
We will be using CIFAR10 dataset. You can install an automatically partition it into 2 using the `./src/setup_data.py` script provided.

```sh
python ./src/setup_data.py 2
```

This will download the data, partition it into 2 shards and store it under the `./data/1` and `./data/2`, respectively.

```
data/
├── 1
│   ├── test
│   │   ├── 0
│   │   ├── 1
│   │   ├── 2
│   │   ├── 3
│   │   ├── 4
│   │   ├── 5
│   │   ├── 6
│   │   ├── 7
│   │   ├── 8
│   │   └── 9
│   └── train
│       ├── ...
└── 2
    ├── ...
```

### Configure the Experiment
Notice under `./plan`, you will find the familiar OpenFL YAML files to configure the experiment. `cols.yaml` and `data.yaml` will be populated by the collaborators that will run the Flower client app and the respective data shard or directory they will perform their training and testing on.
`plan.yaml` configures the experiment itself. The Open-Flower integration makes a few key changes to the `plan.yaml`:

1. Introduction of a new top-level key (`connector`) to configure a newly introduced component called `ConnectorFlower`. This component is run by the aggregator and is responsible for initializing the Flower `SuperLink` and connecting to the OpenFL server. Under `settings`, you will find the parameters for both the `flower-superlink` and `flower run` commands. All parameters are configuration by the user. By default, the `flower-superlink` will be run in `insecure` mode. The default `fleet_api_port` and `exec_api_port` will be automatically assigned, while the `exec_api_port` should be set to match the address configured in `./src/app-pytorch/pyproject.toml`. This is not set dynamically. In addition, since OpenFL handles cross network communication, `superlink_host` is set to a local host by default. For the `flwr run` command, the user should ensure that the `federation_name` and `flwr_app_name` is consistent with what is defined in `./src/<flwr_app_name>` (if different than `app-pytorch`) and `./src/<flwr_app_name>/pyproject.toml`. The Flower directory `flwr_dir` is set to save the FAB in `save/.flwr`. Should a user configure this, the save directory must be located inside the workspace. Additionally, the `ConnectorFlower` has a setting `automatic_shutdown` which is default set to `True`. When set to `True`, the task runner will shut the SuperNode at the completion of an experiment, otherwise, it will run continuously. 

```yaml
connector:
  defaults: plan/defaults/connector.yaml
  template: openfl.component.ConnectorFlower
  settings:
    automatic_shutdown: true
    insecure: true
    exec_api_port: 9093
    fleet_api_port: 57085
    serverappio_api_port: 58873
    federation_name: local-poc
    flwr_app_name: app-pytorch
    flwr_dir: save/.flwr
    superlink_host: 127.0.0.1
```

2. `FlowerTaskRunner` which will execute the `start_client_adapter` task. This task starts the Flower SuperNode and makes a connection to the OpenFL client. In addition, you will notice there are settings for the `flwr_app_name`, `flwr_dir`, and `sgx_enabled`. `flwr_app_name` and `flwr_dir` are for prebuilding and installing the Flower app and should follow the convention as the `Connector` settings. `sgx_enabled` enables secure execution of the Flower `ClientApp` within an Intel<sup>®</sup> SGX enclave. When set to `True`, the task runner will launch the client app in an isolated process suitable for enclave execution and handle additional setup required for SGX compatibility (see [Running in Intel<sup>®</sup> SGX Enclave](#running-in-intel®-sgx-enclave) for details).

```yaml
task_runner:
  defaults: plan/defaults/task_runner.yaml
  template: openfl.federated.task.FlowerTaskRunner
  settings :
    flwr_app_name : app-pytorch
    flwr_dir : save/.flwr
    sgx_enabled: False
```

3. `FlowerDataLoader` with similar high-level functionality to other dataloaders.

4. `Task` - we introduce a `tasks_connector.yaml` that will allow the collaborator to connect to Flower framework via the interop server. It also handles the task runner's `start_client_adapter` method, which actually starts the Flower component and interop server. In the `settings`, the `interop_server` points to the `FlowerInteropServer` module that will establish connect between the OpenFL client and the Flower `SuperNode`. Like the `SuperLink` the host is set to the local host because OpenFL handles cross network communication. The `interop_server_port` and `clientappio_api_port` are automatically allocated by OpenFL. Setting `local_simulation` to `True` will further offest the ports based on the collaborator names in order to avoid overlapping ports. This is not an issue when collaborators are remote.

```yaml
tasks:
  prepare_for_interop:
    function: start_client_adapter
    kwargs: {}
  settings:
    clientappio_api_port: 59731
    interop_server: openfl.transport.grpc.interop.FlowerInteropServer
    interop_server_host: 127.0.0.1
    interop_server_port: 51807
    local_simulation: true

```

5.`Collaborator` has an additional setting `interop_mode` which will invoke a callback to prepare the interop server that'll eventually be started by the Task Runner

```yaml
collaborator :
  defaults : plan/defaults/collaborator.yaml
  template : openfl.component.Collaborator
  settings: 
    interop_mode : True
```

> **Note**: `aggregator.settings.rounds_to_train` is set to 1. __Do not edit this__. The actual number of rounds for the experiment is controlled by Flower logic inside of `./app-pytorch/pyproject.toml`. The entirety of the Flower experiment will run in a single OpenFL round. Increasing this will cause OpenFL to attempt to run the experiment again. The aggregator round is there to stop the OpenFL components at the completion of the experiment.

> **Note**: `aggregator.settings.write_logs` will be set to `False`. While setting it to `True` will not result in an error, OpenFL's aggregator will not capture the logs since logging is handled by Flower directly.

> **Note**: This workspace does not currently support secure aggregation through OpenFL natively. Look into Flower's documentation to enable secure aggregation.

## Execution Methods
There are two ways to execute this:

1. Automatic shutdown which will spawn a `server-app` in isolation and trigger an experiment termination once the it shuts down. (Default/Recommended)
2. Running `SuperLink` and `SuperNode` as [long-lived components](#long-lived-superlink-and-supernode) that will indefinitely wait for new runs. (Limited Functionality)

## Running the Workspace
We proceed with the automatic shutdown method of execution.

Initialize the plan.

```SH
fx plan initialize -a localhost
```

Run the workspace as normal (aggregator setup, collaborator setup, etc.):

```SH
# Generate a Certificate Signing Request (CSR) for the Aggregator
fx aggregator generate-cert-request --fqdn localhost

# The CA signs the aggregator's request, which is now available in the workspace
fx aggregator certify --fqdn localhost --silent

################################
# Setup Collaborator 1 
################################

# Create a collaborator named "collaborator1" that will use shard "0"
fx collaborator create -n collaborator1 -d data/1

# Generate a CSR for collaborator1
fx collaborator generate-cert-request -n collaborator1

# The CA signs collaborator1's certificate
fx collaborator certify -n collaborator1 --silent

################################
# Setup Collaborator 2 
################################

# Create a collaborator named "collaborator2" that will use shard "1"
fx collaborator create -n collaborator2 -d data/2

# Generate a CSR for collaborator2
fx collaborator generate-cert-request -n collaborator2

# The CA signs collaborator2's certificate
fx collaborator certify -n collaborator2 --silent
```

Start the aggregator

```SH
fx aggregator start
```

This will prepare the workspace and start the OpenFL aggregator, Flower superlink, and Flower serverapp. You should see something like:

```SH
INFO     🧿 Starting the Aggregator Service.
.
.
.
INFO :      Starting Flower SuperLink
WARNING :   Option `--insecure` was set. Starting insecure HTTP server.
INFO :      Flower Deployment Engine: Starting Exec API on 127.0.0.1:9093
INFO :      Flower ECE: Starting ServerAppIo API (gRPC-rere) on 127.0.0.1:9091
INFO :      Flower ECE: Starting Fleet API (GrpcAdapter) on 127.0.0.1:9092
.
.
.
INFO :      [INIT]
INFO :      Using initial global parameters provided by strategy
INFO :      Starting evaluation of initial global parameters
INFO :      Evaluation returned no results (`None`)
INFO :      
INFO :      [ROUND 1]
```

### Start Collaborators
Open 2 additional terminals for collaborators.
For collaborator 1's terminal, run:
```SH
fx collaborator start -n collaborator1
```
For collaborator 2's terminal, run:
```SH
fx collaborator start -n collaborator2
```
This will start the collaborator nodes, the Flower `SuperNode`, and Flower `ClientApp`, and begin running the Flower experiment. You should see something like:

```SH
 INFO     🧿 Starting a Collaborator Service.
.
.
.
INFO :      Starting Flower SuperNode
WARNING :   Option `--insecure` was set. Starting insecure HTTP channel to 127.0.0.1:...
INFO :      Starting Flower ClientAppIo gRPC server on 127.0.0.1:...
INFO :      
INFO :      [RUN 297994661073077505, ROUND 1]
```
### Completion of the Experiment
Upon the completion of the experiment, on the `aggregator` terminal, the Flower components should send an experiment summary as the `SuperLink `continues to receive requests from the supernode:
```SH
INFO :      [SUMMARY]
INFO :      Run finished 3 round(s) in 93.29s
INFO :          History (loss, distributed):
INFO :                  round 1: 2.0937052175497555
INFO :                  round 2: 1.8027011854633406
INFO :                  round 3: 1.6812996898487116
```
If `automatic_shutdown` is enabled, this will be shortly followed by the OpenFL `aggregator` receiving "results" from the `collaborator` and subsequently shutting down:

```SH
INFO     Round 0: Collaborators that have completed all tasks: ['collaborator1', 'collaborator2']                                  
INFO     Experiment Completed. Cleaning up...
INFO     Sending signal to collaborator collaborator2 to shutdown...
INFO     Sending signal to collaborator collaborator1 to shutdown...
INFO     [OpenFL Connector] Stopping server process with PID: ...
INFO :    SuperLink terminated gracefully.
INFO     [OpenFL Connector] Server process stopped.
```    
Upon the completion of the experiment, on the `collaborator` terminals, the Flower components should be outputting the information about the run:

```SH
INFO :      [RUN ..., ROUND 3]
INFO :      Received: evaluate message 
INFO :      Start `flwr-clientapp` process
INFO :      [flwr-clientapp] Pull `ClientAppInputs` for token ...
INFO :      [flwr-clientapp] Push `ClientAppOutputs` for token ...
```

If `automatic_shutdown` is enabled, this will be shortly followed by the OpenFL `collaborator` shutting down:

```SH
INFO :      SuperNode terminated gracefully.
INFO     SuperNode process terminated.
INFO     Shutting down local gRPC server... 
INFO     local gRPC server stopped. 
INFO     Waiting for tasks...     
INFO     Received shutdown signal. Exiting...
``` 
Congratulations, you have run a Flower experiment through OpenFL's task runner!

## Advanced Usage
### Long-lived SuperLink and SuperNode
A user can set `automatic_shutdown: False` in the `Connector` settings of the `plan.yaml`. 

```yaml
connector : 
  defaults : plan/defaults/connector.yaml
  template : openfl.component.ConnectorFlower
  settings :
    automatic_shutdown: False
```

By doing so, Flower's `ServerApp` and `ClientApp` will still shut down at the completion of the Flower experiment, but the `SuperLink` and `SuperNode` will continue to run. As a result, on the `aggregator` terminal, you will see a constant request coming from the `SuperNode`:

```SH
INFO :      GrpcAdapter.PullTaskIns
INFO :      GrpcAdapter.PullTaskIns
INFO :      GrpcAdapter.PullTaskIns
```
You can run another experiment by opening another terminal, navigating to this workspace, and running:
```SH
flwr run ./src/app-pytorch
```
It will run another experiment. Once you are done, you can manually shut down OpenFL's `collaborator` and Flower's `SuperNode` with `CTRL+C`. This will trigger a task-completion by the task runner that'll subsequently begin the graceful shutdown process of the OpenFL and Flower components.

### Running in Intel<sup>®</sup> SGX Enclave
Intel SGX is a set of CPU extensions for creating isolated memory regions, called enclaves, that allow secure computation of sensitive data. Applications that run within enclaves are encrypted in memory and remain isolated from the rest of the system. Executing code within an enclave requires an [Intel SGX–supported Intel CPU](https://www.intel.com/content/www/us/en/support/articles/000028173/processors.html). To run this workspace in an SGX enclave, first, set `sgx_enabled: True` in the `plan.yaml` for the `task_runner`:

```yaml
task_runner :
  defaults : plan/defaults/task_runner.yaml
  template : src.runner.FlowerTaskRunner
  settings :
    flwr_app_name: app-pytorch
    sgx_enabled: False
```

Then, follow the [instructions](https://openfl.readthedocs.io/en/latest/about/features_index/taskrunner.html#docker-container-approach) to build/pull a base image, dockerize your workspace, and containerize each component.

Enabling this flag does two things. First, it runs the Flower `ClientApp` as an isolated process (`--isolation process`). This reduces the number of processes that need to be spawned in the enclave, reducing overhead time ([see](https://gramine.readthedocs.io/en/stable/performance.html#multi-process-workloads)). Second, Gramine does not support all Linux system calls. Flower FAB is built and installed at runtime. During this, `utime()` is called, which is an [unsupported call](https://gramine.readthedocs.io/en/latest/devel/features.html#list-of-system-calls), resulting in error or unexpected behavior. To navigate this, when running in an SGX enclave, we opt to build and install the FAB during initialization and package it alongside the OpenFL workspace when the `sgx_enabled` flag is set to `True`