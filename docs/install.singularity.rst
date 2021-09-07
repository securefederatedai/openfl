.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _install_singularity:

Singularity Installation
###################

.. note::

   Make sure you've run the :ref:`the initial steps <install_package>` section first.

.. note::
    You'll need Docker installed on the node where you'll 
    be building the Singularity containers. To check
    that Docker is installed and running properly, you
    can run the Docker *Hello World* command like this:

    .. code-block:: console

      $ docker run hello-world
      Hello from Docker!
      This message shows that your installation appears to be working correctly.
      ...
      ...
      ...

.. note::
    You'll need Singularity installed on all nodes. 
    To check that Singularity is installed, run the following:

    .. code-block:: console

      $ singularity help
     
      Linux container platform optimized for High Performance Computing (HPC) and
      Enterprise Performance Computing (EPC)
      ...
      ...
      ...


1. TODO