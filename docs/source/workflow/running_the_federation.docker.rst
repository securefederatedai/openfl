.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation_docker:

********************
Docker\* \  Approach
********************

There are two ways you can run |productName| with Docker\*\.

- :ref:`Deploy a federation in a Docker container <running_the_federation_docker_base_image>`
- :ref:`Deploy the workspace in a Docker container <running_the_federation_docker_workspace>`


.. _running_the_federation_docker_base_image:

Option 1: Deploy a Federation in a Docker Container
===================================================

Prerequisites
-------------

You have already built an |productName| image. See :doc:`../../install.docker` for details.

Procedure
---------

1. Run the |productName| image.

    .. code-block:: console

       docker run -it --network host openfl
   

You can now experiment with |productName| in the container. For example, you can test the project pipeline with the :download:`"Hello Federation" bash script <../tests/github/test_hello_federation.sh>`.




.. _running_the_federation_docker_workspace:

Option 2: Deploy Your Workspace in a Docker Container
=====================================================

Prerequisites
-------------

You have already set up a TaskRunner and run :code:`fx plan initialize` in the workspace directory. See :ref:`creating_workspaces` for details.

Procedure
---------

1. Build an image with the workspace you created.

    .. code-block:: console

       fx workspace dockerize 


    By default, the image is saved as **WORKSPACE_NAME_image.tar** in the workspace directory.

2. The image can be distributed and run on other nodes without any environment preparation.

    .. code-block:: console

        docker run -it --rm \
            --network host \
            -v user_data_folder:/home/user/workspace/data \
            ${WORSPACE_IMAGE_NAME} \
            bash


    .. note::
    
        The FL Plan should be initialized with the FQDN of the node where the aggregator container will be running.

3. Generate PKI certificates for all collaborators and the aggregator. See :doc:`/source/utilities/pki` for details.

4. :doc:`running_the_federation.start_nodes`.

