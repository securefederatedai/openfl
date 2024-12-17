# Installation

OpenFL needs to be installed on every node participating in the federation. The installation process depends on the environment in which you plan to run OpenFL. The following sections provide instructions for installing OpenFL in a Python virtual environment or in a Docker container.

## Using `pip`

We recommend using a Python virtual environment. Refer to the [venv installation guide](https://docs.python.org/3/library/venv.html) for details.

* From PyPI (latest stable release):

    ```bash
    # [Optional] Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate

    # Install OpenFL
    python -m pip install openfl
    ```

* For development:

    ```bash
    git clone https://github.com/securefederatedai/openfl.git && cd openfl
    python -m pip install -U pip setuptools wheel
    python -m pip install -e .
    ```

* Nightly (from the `develop` branch):

    ```bash
    python -m pip install git+https://github.com/securefederatedai/openfl.git@develop
    ```

Verify installation using the `fx --help` command.

```bash
OpenFL - Open Federated Learning                                                

BASH COMPLETE ACTIVATION

Run in terminal:
_FX_COMPLETE=bash_source fx > ~/.fx-autocomplete.sh
source ~/.fx-autocomplete.sh
If ~/.fx-autocomplete.sh already exists:
source ~/.fx-autocomplete.sh

CORRECT USAGE

fx [options] [command] [subcommand] [args]

GLOBAL OPTIONS

-l, --log-level TEXT  Logging verbosity level.
--no-warnings         Disable third-party warnings.
--help                Show this message and exit.

AVAILABLE COMMANDS
...
```

## Using `docker`

This method can be used to run federated learning experiments in an isolated environment. Install and verify installation of Docker engine on all nodes in the federation. Refer to the Docker installation [guide](https://docs.docker.com/engine/install/) for details.

* Pull the latest image:

	> **Note:** OpenFL image hosted on `docker.io` has not been updated since the 1.5 release due to a change in namespace. We are working on this issue. In the meantime, use the instructions below to build an image from source.

	```bash
	docker pull intel/openfl
	```
   
* Build from source:
	```bash
	git clone https://github.com/securefederatedai/openfl.git && cd openfl
	git checkout develop
	docker build 
        -t openfl \
        -f Dockerfile.base \
        --build-arg OPENFL_REVISION=https://github.com/securefederatedai/openfl.git@develop .
	```
