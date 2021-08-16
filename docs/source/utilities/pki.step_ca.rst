.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

******************************************
Federation actors certification with Semi-automatic PKI
******************************************

If you have trusted workspace and connection should not be encrypted you can use :code:`disable_tls` option while starting experiment.
Otherwise it is necessary to certify each node participating in the federation. Certificates allow to use mutual tls connection between nodes.
You can certify nodes by your own pki system or use pki provided by OpenFL. It is based on `step-ca <https://github.com/smallstep/certificates>`_
as a server and `step <https://github.com/smallstep/cli>`_ as a client utilities. They are downloaded from github during workspace setup. Regardless of the certification method,
paths to certificates on each node are provided at start of experiment. Pki workflow from OpenFL will be discussed below.

OpenFL PKI workflow
===================
Openfl PKI pipeline asumes creating local CA with https server which listen signing requests.
Certificates from each node can be signed by requesting to CA server with special token.
Token must be copied to each node by some secure way. Each step is considered in detail below.

1. Create CA, i.e create root key pair, CA server config and other.
    .. code-block:: console

       $ fx pki install -p </path/to/ca/dir> --password <123> --ca-url <host:port>
    | :code:`-p` - path to folder, which will contain ca files.
    | :code:`--password` - password that will encrypts some ca files.
    | :code:`--ca-url` - host and port which ca server will listen
    This command will also download `step-ca <https://github.com/smallstep/certificates>`_ and `step <https://github.com/smallstep/cli>`_ binaries from github.

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
