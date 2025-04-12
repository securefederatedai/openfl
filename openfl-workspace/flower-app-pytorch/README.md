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

This will create a workspace in your current working directory called `./my_workspace` as well as install the Flower app defined in `./app-pytorch.` This will be where the experiment takes place. The CA will be used to sign the certificates of the collaborators.

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

1. Introduction of a new top-level key (`connector`) to configure a newly introduced component called `ConnectorFlower`. This component is run by the aggregator and is responsible for initializing the Flower `SuperLink` and connecting to the OpenFL server. The `SuperLink` parameters can be configured using `connector.settings.superlink_params`. If nothing is supplied, it will simply run `flower-superlink --insecure` with the command's default settings as dictated by Flower. It also includes the option to run the flwr run command via `connector.settings.flwr_run_params`. If `flwr_run_params` are not provided, the user will be expected to run `flwr run <app>` from the aggregator machine to initiate the experiment. Additionally, the `ConnectorFlower` has an additional setting `connector.settings.automatic_shutdown` which is default set to `True`. When set to `True`, the task runner will shut the SuperNode at the completion of an experiment, otherwise, it will run continuously. 

```yaml
connector:
  defaults: plan/defaults/connector.yaml
  template: openfl.component.ConnectorFlower
  settings:
    automatic_shutdown: True
    superlink_params:
      insecure: True
      serverappio-api-address: 127.0.0.1:9091
      fleet-api-address: 127.0.0.1:9092
      exec-api-address: 127.0.0.1:9093
    flwr_run_params:
      flwr_app_name: "app-pytorch"
      federation_name: "local-poc"
```

2. `FlowerTaskRunner` which will execute the `start_client_adapter` task. This task starts the Flower SuperNode and makes a connection to the OpenFL client.

```yaml
task_runner:
  defaults: plan/defaults/task_runner.yaml
  template: openfl.federated.task.runner_flower.FlowerTaskRunner
```

3. `FlowerDataLoader` with similar high-level functionality to other dataloaders.

4. `Task` - we introduce a `tasks_connector.yaml` that will allow the collaborator to connect to Flower framework via the local gRPC server. It also handles the task runner's `start_client_adapter` method, which actually starts the Flower component and local gRPC server. By setting `local_server_port` to 0, the port is dynamically allocated. This is mainly for local experiments to avoid overlapping the ports.

```yaml
tasks:
  settings:
    connect_to: Flower
  start_client_adapter:
    function: start_client_adapter
    kwargs:
      local_server_port: 0
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

### Running in SGX Enclave
Gramine does not support all Linux system calls. Flower FAB is built and installed at runtime. During this, `utime()` is called, which is an [unsupported call](https://gramine.readthedocs.io/en/latest/devel/features.html#list-of-system-calls), resulting in error or unexpected behavior. To navigate this, when running in an SGX enclave, we opt to build and install the FAB during initialization and package it alongside the OpenFL workspace. To make this work, we introduce some patches to Flower's build command, which helps circumvent the unsupported system call as well as minimize read/write access.

To run these patches, simply add `patch: True` to the `Connector` and `Task Runner` settings (if not already set). For the `Task Runner` also include the name of the Flower app for building and installation.

```yaml
connector : 
  defaults : plan/defaults/connector.yaml
  template : openfl.component.ConnectorFlower
  settings :
    automatic_shutdown : True
    superlink_params :
      insecure : True
      serverappio-api-address : 127.0.0.1:9091 
      fleet-api-address :  127.0.0.1:9092 
      exec-api-address : 127.0.0.1:9093
    flwr_run_params :
      flwr_app_name : "app-pytorch"
      federation_name : "local-poc"
      patch : True

task_runner :
  defaults : plan/defaults/task_runner.yaml
  template : openfl.federated.task.runner_flower.FlowerTaskRunner
  settings :
    patch : True
    flwr_app_name : "app-pytorch"
```