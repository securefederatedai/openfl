.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_the_federation_docker:

Running on Docker
#################

First make sure you have :ref:`followed the Docker installation steps <install_docker>` to have the containerized version of |productName|. A demo script can be found at :code:`docker_keras_demo.sh`.


TL;DR
=====

Here's the :download:`DockerFile <../openfl-docker/Dockerfile>`. This image can be reused for aggregators and collaborators.

.. literalinclude:: ../openfl-docker/Dockerfile
  :language: docker
  

Here's the :download:`"Hello Docker Federation" <../openfl-docker/docker_keras_demo.sh>` demo. This is an end-to-end demo for the Keras CNN MNIST (:code:`docker_keras_demo.sh`).

.. literalinclude:: ../openfl-docker/docker_keras_demo.sh
  :language: bash

Custom execution
================

We provide two methods of packaging and running openfl through docker: The first is a set of bash scripts in the :code:`openfl-docker` directory that let you easily run the :code:`keras_mnist_cnn` template with a single collaborator. The second is the :code:`fx workspace dockerize` command, which packages an existing workspace into a docker image with relevant dependencies.

Option #1: Simple script to run the federation
==============================================

Once built, the current image can be instantiated in two modes:

As an **aggregator**:

.. code-block:: console

   $ container-home>> bash docker_agg.sh CMD
   
   
As a **collaborator**:

.. code-block:: console

   $ container-home>> bash docker_col.sh CMD
   
Each :code:`bash` file contains its own list of methods to implement and run the federation. Each method is relevant depending on which step of the pipeline one needs to address. 

+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| Command                         | Who          | What                                                                    | Assumptions                                                                                                   |
+=================================+==============+=========================================================================+===============================================================================================================+
| bash docker\_agg.sh init        | Aggregator   | Initialize and certify the workspace                                    | None                                                                                                          |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_agg.sh export      | Collaborator | Export the workspace into a "workspace\_name.zip" file                  | Workspace has been created                                                                                    |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_col.sh import\_ws  | Collaborator | Import the workspace "workspace\_name.zip"                              | File already transfered to the collaborator workspace directory on the host                                   |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_col.sh init        | Collaborator | initialize the collaborator (i.e. generates the "col\_$COL\_NAME" dir ) | Workspace has been exported                                                                                   |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_agg.sh col         | Aggregator   | certify the collaborator request                                        | "col\_$COL\_NAME" directory already transfered to the aggregator workspace directory                          |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_col.sh import\_crt | Collaborator | Import the validated certificate                                        | A signed certificate request zip archive has already been transfered to the aggregator                        |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_agg.sh start       | Aggregator   | Start the aggregator                                                    | None                                                                                                          |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| bash docker\_col.sh start       | Collaborator | Start the collabortor                                                   | The .crt files (collaborator and cert\_chain) have already been transfered to the correct collaborator        |
+---------------------------------+--------------+-------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+

Execution on hosts with non-root access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current image ensures that both environment with root and non-root access can run the docker openfl container smoothly. 
While users with root access wouldn’t require particular instructions, there are few considerations that are worth to be shared for those user with limited permissions. 
To achieve this result, the image will need to be built by providing the information of the current user at build time. This will ensure that all the actions taken by the container at runtime, will be owned by the same user logged in the host.


Single and Multi-node execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the current image one can create a full federation within a single node or distributed across multiple nodes. 
The complete pipeline on how to initiate a federation composed by one aggregator and one collaborator running on the same node is demonstrated in :code:`docker_keras_demo.sh`.
The multinode execution has only been tested on an isolated cluster of three machines connected on the same internal network, respectively running one aggregator and two collaborators.
To simulate a realistic environment, these machines didn’t have password-less access between each other. The file exchanged between the aggregator and the collaborators at the beginning of the process (workspace, certificate requests and validated certificates) have been manually performed by copying the files from one host to the others. The mechanism to automate such operations is currently under consideration by the development team.
At this stage, one can replicate the approach adopted in the attached demo to run a custom federation. 


Hello Federation Docker
=======================

This demo runs on a single node and creates a federation with two institutions: one aggregator and one collaborator.
Both the institutions are containerized and the |productName| software stack is self-contained within docker.

To emulate the workspaces of both components, it will create two separate directories (*host_agg_workspace* and *host_col_workspace*) in the :code:`home` directory on the local host.

The name of the docker image to be used for the demo can be set as first argument when calling the script. By default, the bash script will rely on the docker image name used to build it with the previous command (*e.g.* :code:`openfl/docker`).

The path where the two local directories will be created can be passed as second argument. If empty, it will default to :code:`/home/$USERNAME`.

.. code-block:: console

   $ bash docker_keras_demo.sh

Run the demo with custom parameters
===================================

You can run the same Docker container and pass your custom image name and path names as follows:

.. code-block:: console

   $ bash docker_keras_demo.sh myDockerImg/name /My/Local/Path


Option #2: Deploying your workspace in Docker
=============================================

These steps assume that you have already run `fx workspace dockerize` in your workspace directory. To simplify the setup experience, all partipants all use the same container.

0. Open a terminal for each partipant (e.g. If you plan to run an experiment with two collaborators then open three terminals (1 aggregator + 2 collaborators)

1. Start your docker container in detached-interactive mode:

.. code-block:: console

   $ host>> CONTAINER_ID=$(docker run -dit openfl/docker_WORKSPACE_NAME bash)

If you have a data directory to mount into the container, run this command instead:

.. code-block:: console

   $ host>> CONTAINER_ID=$(docker run -dit -v /path/to/data:/home/workspace/data openfl/docker_WORKSPACE_NAME bash)

2. In the first terminal (aggregator's terminal), enter the container:

.. code-block:: console

   $ host>> docker exec -it $CONTAINER_ID bash

Repeat the same command for each of the collaborator terminals

3. Because each of the partipants will be sharing a workspace for this example, you can run the following command to perform the full prerequisite setup. This can be run from any terminal, and assumes your experiment will include two collaborators:

.. code-block:: console

   $ container:          ~/home>> cd workspace
   $ container:~/home/workspace>> fx workspace certify && \
                                  fx aggregator generate-cert-request && \
                                  fx aggregator certify -s && \
                                  fx collaborator generate-cert-request -n one -d 1 && \
                                  fx collaborator certify -n one -s && \
                                  fx collaborator generate-cert-request -n two -d 2 && \
                                  fx collaborator certify -n two -s


4. Now that the workspace and collaborators are set up, we are ready to run the experiment.

4.a. (From the aggregator terminal)

.. code-block:: console

   $ container:~/home/workspace>> fx aggregator start

4.b. (From collaborator one's terminal)

.. code-block:: console

   $ container:~/home/workspace>> fx collaborator start -n one

4.c. (From collaborator two's terminal)

.. code-block:: console

   $ container:~/home/workspace>> fx collaborator start -n two 


