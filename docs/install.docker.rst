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

The Docker version of |productName| is designed to provide an isolated environment that is fully equipped
with all the necessary prerequisites to run a federation. Once the execution is over, 
the container can be destroyed and the results of the computation will be available on a directory on local host.


Design Assumptions
~~~~~~~~~~~~~~~~~~

The current design is based on three assumptions:

  * Docker containers are the essential components to have a working environment that is able to initiate and run a federation. Orchestration and pipeline automation is not currently supported and would need to be handled manually.

  * `PKI exchange <https://en.wikipedia.org/wiki/Public_key_infrastructure>`_ required to validate and welcome new collaborators to the federation will need to be handled outside the containers through a bash script (not provided) or performed manually by:

     * Copying files manually (:code:`scp` or some other secure file transfer) from one host to another
     * Creating a shared file system across the federation hosts. *This option might not be ideal for hosts not connected to the same internal network*

.. figure:: images/docker_design.png
   :alt: Docker design
   :scale: 70%

.. centered:: Docker design


Build the docker image
======================

To use the latest official |productName| release, simply run:

.. code-block:: console

   $ docker pull intel/openfl
   
Or if you would prefer to build an image from a specific commit or branch, run the following commands:

.. code-block:: console

   $ git clone https://github.com/intel/openfl.git
   $ cd openfl
   $ ./scripts/build_base_docker_image.sh

