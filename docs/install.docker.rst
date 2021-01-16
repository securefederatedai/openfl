.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _install_docker:

Docker Installation
###################

.. note::
    You'll need Docker installed on all nodes. To check
    that Docker is installed and running properly, you
    can run the Docker *Hello World* command like this:

    .. code-block:: console

      $ docker run hello-world
      Hello from Docker!
      This message shows that your installation appears to be working correctly.
      ...
      ...
      ...
      

Design Philosophy
~~~~~~~~~~~~~~~~~

The Docker version of |productName| was designed to be simple and embrace the Docker philosophy.
By building and running the Dockerfile, one will be able to have an isolated environment that is fully equipped
with all the right dependencies and prerequisites. Once the execution is over, the container can be destroyed and
the results of the computation will be available on a directory on local host.


Design Assumptions
~~~~~~~~~~~~~~~~~~

The current design is based on three assumptions:

  * Docker containers are the essential components to have a working environment that is able to initiate and run a federation. Orchestration and pipeline automation is not currently supported and would need to be handled manually.

  * Each machine hosting the aggregator or a collaborator container is expected to have a local workspace directory which can be mapped onto the hosted container. These directories are *not* expected to be shared among hosts.

  * `PKI exchange <https://en.wikipedia.org/wiki/Public_key_infrastructure>`_ required to validate and welcome new collaborators to the federation will need to be handled outside the containers through a bash script (not provided) or performed manually by:

     * Coping files manually (:code:`scp` or some other secure file transfer) from one host to another
     * Creating a shared file system across the federation hosts. *This option might not be ideal for hosts not connected to the same internal network*

.. figure:: images/docker_design.png
   :alt: Docker design
   :scale: 70%

   Docker design


Build the docker image
======================

Requirements
~~~~~~~~~~~~

In order to successfully build the image, the Dockerfile is expecting to access the following dependencies:

* Find the :code:`openfl` directory in the same location where we are going to execute the :code:`docker build` command.
* Find the :code:`docker_agg.sh` file
* Find the :code:`docker_col.sh` file

Command
~~~~~~~

If you have the |productName| repo available localy, run the following command from the base directory to build the docker image:
.. code-block:: console

   $ export HOST_USER=`whoami`
   $ docker build --build-arg USERNAME=`whoami` --build-arg USER_ID=`id -u $HOST_USER` --build-arg GROUP_ID=`id -g $HOST_USER` -t openfl/docker -f openfl-docker/Dockerfile .

If you have installed |productName| from a pip wheel or pypi, you can easily add your local workspace to a docker image with the following commands:
.. code-block:: console

   $ fx workspace create --prefix ~/WORKSPACE_PATH --template keras_cnn_mnist
   $ cd WORKSPACE_PATH
   $ fx workspace dockerize

