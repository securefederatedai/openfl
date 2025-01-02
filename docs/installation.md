# Installation

This document provides instructions for installing OpenFL; either in a Python virtual environment or as a docker container.

## Using `pip`

We recommend using a Python virtual environment. Refer to the [venv installation guide](https://docs.python.org/3/library/venv.html) for details.

* From PyPI (latest stable release):

    ```bash
    pip install openfl
    ```

* For development (editable build):

    ```bash
    git clone https://github.com/securefederatedai/openfl.git && cd openfl
    pip install -e .
    ```

* Nightly (from the tip of `develop` branch):

    ```bash
    pip install git+https://github.com/securefederatedai/openfl.git@develop
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
    ```
    ```bash
	./scripts/build_base_image.sh
	```
