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

    The |productName| PKI solution is based on `step-ca <https://github.com/smallstep/certificates>`_ as a server and `step <https://github.com/smallstep/cli>`_ as a client utilities. They are downloaded during the workspace setup.

.. note::

   Different certificates can be created for each project workspace.

.. _install_certs:

.. kroki:: ../../mermaid/CSR_signing.mmd
    :caption: Manual certificate generation and signing
    :align: center
    :type: mermaid

.. kroki:: ../../mermaid/pki_scheme.mmd
    :caption: Step-ca certificate generation and signing
    :align: center
    :type: mermaid

.. _semi_automatic_certification:

Semi-Automatic PKI Workflow
===========================

The |productName| PKI pipeline involves creating a local certificate authority (CA) on a \HTTPS \ server that listens for signing requests. Certificates from each client are signed by the CA via a token. The token must be copied to clients in a secure manner. 

1. Create the CA.

      .. code-block:: console

         fx pki install -p </path/to/ca/dir> --ca-url <host:port>
      | where
      | :code:`-p` defines the path to the directory that contains CA files, and
      | :code:`--ca-url` defines the host and port that the CA server will listen, if not specified, :code:`--ca-url` will be "localhost:9123"
      When executing this command, you will be prompted for a password and password confirmation. The password will encrypt some CA files.
      This command will also download `step-ca <https://github.com/smallstep/certificates>`_ and `step <https://github.com/smallstep/cli>`_ binaries.

2. Run the CA server.

      .. code-block:: console

         fx pki run -p </path/to/ca/dir>
      | where
      | :code:`-p` defines the path to the directory that contains CA files.

3. Create a token for client.

      .. code-block:: console

         fx pki get-token -n <subject> --ca-path </path/to/ca/dir> --ca-url <host:port>
      | where
      | :code:`-n` defines the subject name, FQDN for director, collaborator name for envoy, or API name for the API-layer node.
      | :code:`--ca-path` defines the path to the directory that contains CA files.
      | :code:`--ca-url` defines the host and port that the CA server will listen, if not specified, :code:`--ca-url` will be "localhost:9123"

      Run this command from the CA directory on the CA server. The output is a token which contains a JWT (JSON web token) from the CA server and the CA root certificate concatenated together. This JWT is valid for 24 hours.

4. Copy the token to the clients (director or envoy) via a secure channel, and certify the token.

      .. code-block:: console

         cd <path/to/subject/folder>
         fx pki certify -n <subject> -t <generated token for subject>
      | where
      | :code:`-n` defines the subject name, FQDN for director, collaborator name for envoy, or API name for the API-layer node.
      | :code:`-t` defines the output token from the previous command.
      With this command, the client connects to the CA server over \HTTPS\, which is provided by the root certificate which was copied together with the JWT. The CA server authenticates the client via the JWT, and the client authenticates the server via the root certificate.

The signed certificate and private key are stored on each node in the federation. The signed certificate is valid for one year. You should certify all nodes that will participate in the federation director, which includes all envoys and API-layer nodes.
   


.. _manual_certification:


Manual PKI Workflow 
===================

This solution is embedded into the aggregator-based workflow. See :ref:`Configure the Federation <configure_the_federation>` for details.