.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation_docker_based_workspace:

*****************************************
Deploy the Workspace in Docker
*****************************************

Prerequisites
=============

You have already set up a TaskRunner and run :code:`fx plan initialize` in the workspace directory. See :ref:`running_the_federation.baremetal` for details.

Prepare the Workspace in a Docker Container
===========================================

1. Build an image with |productName| installed and the workspace imported.

    .. code-block:: console

       $ fx workspace dockerize 


    By default, the image is saved as **WORKSPACE_NAME_image.tar** in the workspace directory.

2. The image can be distributed and run on other nodes without any environment preparation.

    .. code-block:: console

       $ docker run -it --rm \
            --network host \
            -v user_data_folder:/home/user/workspace/data \
            ${WORSPACE_IMAGE_NAME} \
            bash


    .. note::
    
        The FL Plan should be initialized with the FQDN of the node where the aggregator container will be running.

3. Generate PKI certificates for all collaborators and the aggregator. See :ref:`pki` for details.

4. Start the federation.

