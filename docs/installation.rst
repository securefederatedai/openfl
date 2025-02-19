Installation
============

This document provides instructions for installing OpenFL; either in a Python virtual environment or as a docker container.

Using ``pip``
-----------

We recommend using a Python virtual environment. Refer to the `venv installation guide <https://docs.python.org/3/library/venv.html>`_ for details.

* From PyPI (latest stable release):

  .. code-block:: bash

    pip install openfl

* For development (editable build):

  .. code-block:: bash

    git clone https://github.com/securefederatedai/openfl.git && cd openfl
    pip install -e .

* Nightly (from the tip of `develop` branch):

  .. code-block:: bash

    pip install git+https://github.com/securefederatedai/openfl.git@develop

Verify installation using the ``fx --help`` command.

.. code-block:: bash

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

Using ``docker``
--------------

This method can be used to run federated learning experiments in an isolated environment. Install and verify installation of Docker engine on all nodes in the federation. Refer to the Docker installation `guide <https://docs.docker.com/engine/install/>`_ for details.

* Pull the latest image:

  .. code-block:: bash

    docker pull ghcr.io/securefederatedai/openfl:latest

* Build from source:

  .. code-block:: bash

    git clone https://github.com/securefederatedai/openfl.git && cd openfl
    git checkout develop

  .. code-block:: bash

    docker build -t openfl -f openfl-docker/Dockerfile.base .

  .. note::
    This command copies current context (i.e. OpenFL root directory) to the base image. Ensure that the ``.dockerignore`` file is configured to exclude unnecessary files and directories (like secrets or local virtual environments).