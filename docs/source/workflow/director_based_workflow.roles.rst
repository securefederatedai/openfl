.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0


.. _establishing_federation_director:

Director Manager: Set Up the Director
=====================================

The *Director manager* sets up the *Director*, which is the central node of the federation.

    - :ref:`step1_install_director_prerequisites`
    - :ref:`STEP 2: Create Public Key Infrastructure (PKI) Certificate Using Step-CA (Optional) <step2_create_pki_using_step_ca>`
    - :ref:`step3_start_the_director`


.. _step1_install_director_prerequisites:

STEP 1: Install Open Federated Learning (|productName|) 
-------------------------------------------------------

Install |productName| in a virtual Python\*\  environment. See :ref:`install_package` for details.

.. _step2_create_pki_using_step_ca:

STEP 2: Create PKI Certificates Using Step-CA (Optional)
--------------------------------------------------------

The use of mutual Transport Layer Security (mTLS) is recommended for deployments in untrusted environments to establish participant identity and to encrypt communication. You may either import certificates provided by your organization or generate certificates with the :ref:`semi-automatic PKI <semi_automatic_certification>` provided by |productName|.

.. _step3_start_the_director:

STEP 3: Start the Director
--------------------------

Start the Director on a node with at least two open ports. See :ref:`openfl_ll_components` to learn more about the Director entity.

1. Create a Director workspace with a default config file.

    .. code-block:: console

        fx director create-workspace -p director_ws
        
 This workspace will contain received experiments and supplementary files (Director config file and certificates).

2. Modify the Director config file according to your federation setup.

 The default config file contains the Director node FQDN, an open port, path of certificates, and :code:`sample_shape` and :code:`target_shape` fields with string representation of the unified data interface in the federation.
 
3. Start the Director.

 If mTLS protection is not set up, run this command.
 
    .. code-block:: console

       fx director start --disable-tls -c director_config.yaml
 
 If you have a federation with PKI certificates, run this command.
 
    .. code-block:: console

       fx director start -c director_config.yaml \
            -rc cert/root_ca.crt \
            -pk cert/priv.key \
            -oc cert/open.crt



.. _establishing_federation_envoy:

Collaborator Manager: Set Up the Envoy
======================================

The *Collaborator manager* sets up the *Envoys*, which are long-lived components on collaborator nodes. Envoys receive an experiment archive and provide access to local data. When started, Envoys will try to connect to the Director.

    - :ref:`step1_install_envoy_prerequisites`
    - :ref:`STEP 2: Sign Public Key Infrastructure (PKI) Certificate (Optional) <step2_sign_pki_envoy>`
    - :ref:`step3_start_the_envoy`

.. _step1_install_envoy_prerequisites:

STEP 1: Install |productName| 
-----------------------------

Install |productName| in a virtual Python\*\  environment. See :ref:`install_package` for details.

.. _step2_sign_pki_envoy:

STEP 2: Sign PKI Certificates (Optional)
--------------------------------------------------------

The use of mTLS is recommended for deployments in untrusted environments to establish participant identity and to encrypt communication. You may either import certificates provided by your organization or use the :ref:`semi-automatic PKI certificate <semi_automatic_certification>` provided by |productName|.


.. _step3_start_the_envoy:

STEP 3: Start the Envoy
-----------------------

1. Create an Envoy workspace with a default config file and shard descriptor Python\*\  script.

    .. code-block:: console

        fx envoy create-workspace -p envoy_ws

2. Modify the config file and local shard descriptor template.

    - Provide the settings field with the arbitrary settings required to initialize the shard descriptor.
    - Complete the shard descriptor template field with the address of the local shard descriptor class.

    .. note::
        The shard descriptor is an object to provide a unified data interface for FL experiments. 
        The shard descriptor implements :code:`get_dataset()` method as well as several additional 
        methods to access **sample shape**, **target shape**, and **shard description** that may be used to identify 
        participants during experiment definition and execution.

        :code:`get_dataset()` method accepts the dataset_type (for instance train, validation, query, gallery) and returns 
        an iterable object with samples and targets.
        
        Abstract shard descriptor should be subclassed and all its methods should be implemented to describe the way data samples and labels will be loaded from disk during training. 
        
3. Start the Envoy.

 If mTLS protection is not set up, run this command.
 
    .. code-block:: console

        fx envoy start -n env_one --disable-tls \
            --envoy-config-path envoy_config.yaml -dh director_fqdn -dp port

 If you have a federation with PKI certificates, run this command.
 
    .. code-block:: console

        ENVOY_NAME=envoy_example_name

        fx envoy start -n "$ENVOY_NAME" \
            --envoy-config-path envoy_config.yaml \
            -dh director_fqdn -dp port -rc cert/root_ca.crt \
            -pk cert/"$ENVOY_NAME".key -oc cert/"$ENVOY_NAME".crt
            


.. _establishing_federation_experiment_manager:

Experiment Manager: Describe an Experiment
==========================================

The process of defining an experiment is decoupled from the process of establishing a federation. 
The Experiment manager (or data scientist) is able to prepare an experiment in a Python environment. 
Then the Experiment manager registers experiments into the federation using :ref:`Interactive Python API <interactive_api>` 
that is equiped with a gRPC client for communication with Director.
