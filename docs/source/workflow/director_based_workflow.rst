.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _director_workflow:

************
Director-based workflow
************

.. toctree::
    :maxdepth: 2
 
    `Establishing a long-living Federation with Director`_
    interactive_python_api

.. _establishing_federation_director:

Establishing a long-living Federation with Director
#######################################

1. Install |productName| 
==================

Make sure that you installed |productName| in your virtual Python environment.
If not, use the instruction :ref:`install_initial_steps`.

2. Implement Shard Descriptors
==================

Then the data owners need to implement `Shard Descriptors` Python classes. 

|productName| framework provides a ‘Shard descriptor’ interface that should be described on every collaborator node 
to provide a unified data interface for FL experiments. Abstract “Shard descriptor” should be subclassed and 
all its methods should be implemented to describe the way data samples and labels will be loaded from disk 
during training. Shard descriptor is a subscriptable object that implements `__getitem__()` and `len()` methods 
as well as several additional methods to access ‘sample shape’, ‘target shape’, and ‘shard description’ text 
that may be used to identify participants during experiment definition and execution.

3. Start Director
==================

    .. code-block:: console

       $ fx director start --disable-tls -c director_config.yaml

    .. code-block:: console

       $ FQDN=$1
       $ fx director start -c director_config.yaml \
            -rc cert/root_ca.crt \
            -pk cert/"${FQDN}".key \
            -oc cert/"${FQDN}".crt

4. Start Envoys
==================

    .. code-block:: console

        $ fx envoy start -n env_one --disable-tls \
            --shard-config-path shard_config.yaml -d director_fqdn:port

    .. code-block:: console

        $ ENVOY_NAME=$1
        $ DIRECTOR_FQDN=$2

        $ fx envoy start -n "$ENVOY_NAME" \
            --shard-config-path shard_config.yaml \
            -d "$DIRECTOR_FQDN":50051 -rc cert/root_ca.crt \
            -pk cert/"$ENVOY_NAME".key -oc cert/"$ENVOY_NAME".crt


5. Describing an FL experimnet using Interactive Python API
====================================

At this point, data scientists may register their experiments to be executed in the federation. 
OpenFL provides a separate frontend Director’s client and :ref:`Interactive Python API <interactive_api>` 
to register experiments. 