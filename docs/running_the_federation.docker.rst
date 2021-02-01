.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation_docker:

Running |productName| with Docker
#################

There are two ways one can use OpenFL in a docker container.
If you do not want to install OpenFL localy you can pull an image from Docker Hub and conduct experiments inside a container.
The other option is to set up a workspace on your machine and use the :code:`fx workspace dockerize` command, that 
can build an image from it.

Option #1: Run the base container
=================================

Pull |productName| image and run it:

.. code-block:: console

   $ docker run -it --network host openfl
   
Now one is free to experiment with OpenFL in the container, for instance run the :ref:`one-node Hello Federation script <running_the_federation.baremetal>`


Option #2: Deploying your workspace in Docker
=============================================

These steps assume that user have already set up a TaskRunner and run `fx plan initialize` in the workspace directory. 
One can do it following steps in the :ref:`Hello Federation <running_the_federation.baremetal>` script

1. Dockerize the workspace:

.. code-block:: console

   $ fx workspace dockerize 

This command will build an image with OpenFL installed and the workspace imported.
By default, it saves the image to a tarball named `WORKSPACE_NAME_image.tar` under the workspace directory.

2. The image then can be distributed and run on other machines without any environment prepartion.

.. code-block:: console

   $ docker run -it --rm \
        --network host \
        -v user_data_folder:/home/user/workspace/data \
        ${WORSPACE_IMAGE_NAME} \
        bash

Keep in mind that plan should be initialized with the FQDN of the node where the aggregator container will be running.

3. Generate PKI certificates for all collaborators and the aggregator.

4. Start the Federation.

