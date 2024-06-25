This readme provides instructions for setting up and running the Horovod example using OpenFL.

## Prerequisites
Before running the Horovod example, ensure that the following prerequisites are met:

1. Python environment should be set up on all nodes.
2. The environment should be sourced when logging into SSH.

## Setting up Horovod Dependencies
To set up the Horovod dependencies, follow these steps:

1. Run the `setup_env.sh` script located in `openfl-workspace/torch_llm_horovod/setup_env.sh` within your virtual environment (venv).
2. Create aggregator and collaborator workspaces. See [Running the experiment](#running-the-experiment)
3. Ensure that the collaborator workspace is present in each node with the same file structure.
4. Make sure the dataset is available in each node.

## Setting up Passwordless SSH Login
Horovod requires passwordless SSH login. Follow the instructions provided at [this link](http://www.linuxproblem.org/art_9.html) to set it up.

## Environmental Variables
Set the following environmental variables for Horovod:

- `OPENFL_HOROVOD_DEMO_NP`: Set this variable to the number of processes to run (e.g., "4").
- `OPENFL_HOROVOD_DEMO_NICS`: Set this variable to the common network interface name to use with all nodes (e.g., "en01,lo").
- `OPENFL_HOROVOD_DEMO_LOCALHOSTIP`: Set this variable to the IP address of the local node (e.g., "ip1").
- `OPENFL_HOROVOD_DEMO_HOSTS`: Set this variable to the IP address of each node and the number of slots (e.g., "ip1:2,ip2:2").

## Customizing Data and Models
To use your own data and models, follow these steps:

1. Copy the `openfl/openfl-workspace/torch_llm_horovod` directory to `openfl/openfl-workspace/name_of_your_template`.
2. In the `src/InHorovodrun` file, make the following changes:
   - Replace `EmotionDataLoader` with your own dataloader.
   - Replace `LLMTrainer` with your own training/validation scripts.

## Running the Experiment
To run the experiment, follow the instructions provided in the [OpenFL documentation](https://openfl.readthedocs.io/en/latest/running_the_federation.html#bare-metal-approach) using either the `torch_llm_horovod` template or your own template.

That's it! You're now ready to use the Horovod example with your own data and models. Enjoy!

