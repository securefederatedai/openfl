.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

*******************************************************
|productName| Public Key Infrastructure (PKI) Solutions
*******************************************************

.. toctree::
   :maxdepth: 2

   pki_overview_
   semi_automatic_certification_
   manual_certification_


.. _pki_overview:

Overview
========

Transport Layer Security (`TLS <https://en.wikipedia.org/wiki/Transport_Layer_Security>`_) encryption is used for network connections in federated learning. Therefore, security keys and certificates will need to be created for the aggregator and collaborators to negotiate the connection securely. 

If you have trusted workspaces and connections, you can start your experiment with the :code:`disable_tls` option.


Otherwise, you can certify nodes with your own PKI solution or use the PKI solution workflows provided by |productName|. 

    - :ref:`semi_automatic_certification`
    - :ref:`manual_certification`

.. note::

    The |productName| PKI solution is based on `step-ca <https://github.com/smallstep/certificates>`_ as a server and `step <https://github.com/smallstep/cli>`_ as a client utilities. They are downloaded from the repository during the workspace setup.

.. note::

   Different certificates can be created for each project workspace.

.. _install_certs:

.. kroki:: mermaid/CSR_signing.mmd
    :caption: Manual certificate generation and signing
    :align: center
    :type: mermaid

.. kroki:: mermaid/pki_scheme.mmd
    :caption: Step-ca certificate generation and signing
    :align: center
    :type: mermaid

.. _semi_automatic_certification:

Semi-Automatic PKI Workflow
===========================

The |productName| PKI pipeline involves creating a local certificate authority (CA) on the aggregator node that listens for signing requests from collaborator nodes. Certificates from each node are signed by the CA via a token. The token must be copied to each collaborator node in a secure manner. 

1. Create the CA.

      .. code-block:: console

         fx pki install -p </path/to/ca/dir> --ca-url <host:port>
      | where you use
      | :code:`-p` to define the path to the folder that contains CA files, and
      | :code:`--ca-url` to define the host and port that the CA server will listen
      When executing this command, you will be prompted for a password and password confirmation. The password will encrypt some CA files.
      This command will also download `step-ca <https://github.com/smallstep/certificates>`_ and `step <https://github.com/smallstep/cli>`_ binaries from the repository.

2. Run CA https server.
      .. code-block:: console

         $ fx pki run -p </path/to/ca/dir>
      | :code:`-p` - path to folder, which will contain ca files.

3. Get token for some node.

      .. code-block:: console

         $ fx pki get-token -n <subject>
      | :code:`-n` - subject name, fqdn for director, collaborator name for envoy or api name for api-layer node

      Run this command on ca side, from ca folder. Output is a token which contains JWT (json web token) from CA server and CA
      root certificate concatenated together. This JWT have twenty-four hours time-to-live.

4. Copy token to node side (director or envoy) by some secure channel and run certify command.
      .. code-block:: console

         $ fx pki certify -n <subject> -t <token>
      | :code:`-n` - subject name, fqdn for director, collaborator name for envoy or api name for api-layer node
      | :code:`-t` - output token from previous command
      This command call step client, to connect to CA server over https.
      Https is provided by root certificate which was copy with JWT.
      Server authenticates client by JWT and client authenticates server by root certificate.

Now signed certificate and private key are stored on current node. Signed certificate has one year time-to-live. You should certify all node that will participate in federation: director, all envoys and api-layer node.
   


.. _manual_certification:


Manual PKI
************

This solution is embedded into the Aggregator-based |productName| workflow.
Please, refer to the :ref:`instruction_manual_certs` section.