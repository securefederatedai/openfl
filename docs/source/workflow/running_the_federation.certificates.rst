.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _instruction_manual_certs:

********************************
STEP 2: Configure the Federation
********************************

The objectives in this step:

    - Ensure each node in the federation has a valid public key infrastructure (PKI) certificate. See :doc:`/source/utilities/pki` for details on available workflows.
    - Distribute the workspace from the aggregator node to the other collaborator nodes.

    
.. _install_certs_agg:

On the Aggregator Node
======================

Setting Up the Certificate Authority
------------------------------------

1. Change to the path of your workspace:

    .. code-block:: console
    
       cd WORKSPACE.PATH

2. Set up the aggregator node as the `certificate authority <https://en.wikipedia.org/wiki/Certificate_authority>`_ for the federation. 

 All certificates will be signed by the aggregator node. Follow the instructions and enter the information as prompted. The command will create a simple database file to keep track of all issued certificates. 

    .. code-block:: console
    
       fx workspace certify

3. Run the aggregator certificate creation command, replacing :code:`AFQDN` with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ for the aggregator node.

    .. code-block:: console
    
       fx aggregator generate-cert-request --fqdn AFQDN
       
    .. note::
    
       On Linux\*\, you can discover the FQDN with this command:
    
           .. code-block:: console
        
              hostname --all-fqdns | awk '{print $1}'
            
   .. note::
   
      You can override the apparent FQDN of the system by setting an FQDN environment variable before creating the certificate.
      
        .. code-block:: console
        
            fx aggregator generate-cert-request export FQDN=x.x.x.x
      
      If you omit the :code:`--fdqn` parameter, then :code:`fx` will automatically use the FQDN of the current node assuming the node has been correctly set with a static address. 
   
        .. code-block:: console
    
            fx aggregator generate-cert-request
       
4. Run the aggregator certificate signing command, replacing :code:`AFQDN` with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ for the aggregator node. 

    .. code-block:: console
    
       fx aggregator certify --fqdn AFQDN
       

   .. note::
   
      You can override the apparent FQDN of the system by setting an FQDN environment variable (:code:`export FQDN=x.x.x.x`) before signing the certificate.

        .. code-block:: console
        
           fx aggregator certify export FQDN=x.x.x.x

5. This node now has a signed security certificate as the aggregator for this new federation. You should have the following files.

    +---------------------------+--------------------------------------------------+
    | File Type                 | Filename                                         |
    +===========================+==================================================+
    | Certificate chain         | WORKSPACE.PATH/cert/cert_chain.crt               |
    +---------------------------+--------------------------------------------------+
    | Aggregator certificate    | WORKSPACE.PATH/cert/server/agg_AFQDN.crt         |
    +---------------------------+--------------------------------------------------+
    | Aggregator key            | WORKSPACE.PATH/cert/server/agg_AFQDN.key         |
    +---------------------------+--------------------------------------------------+
    
    where **AFQDN** is the fully-qualified domain name of the aggregator node.

.. _workspace_export:

Exporting the Workspace
-----------------------

1. Export the workspace so that it can be imported to the collaborator nodes.

    .. code-block:: console
    
       fx workspace export

   The :code:`export` command will archive the current workspace (with a :code:`zip` file extension) and create a **requirements.txt** of the current Python\*\ packages in the virtual environment. 
   
2. The next step is to transfer this workspace archive to each collaborator node.


.. _install_certs_colab:

On the Collaborator Nodes
=========================

1. Copy the :ref:`workspace archive <workspace_export>` from the aggregator node to the collaborator nodes.

2. Import the workspace archive.

    .. code-block:: console
    
       fx workspace import --archive WORKSPACE.zip

 where **WORKSPACE.zip** is the name of the workspace archive. This will unzip the workspace to the current directory and install the required Python packages within the current virtual environment.
   
3. For each test machine you want to run as collaborator nodes, create a collaborator certificate request to be signed by the certificate authority. 

 Replace :code:`COL.LABEL` with the label you assigned to the collaborator. This label does not have to be the FQDN; it can be any unique alphanumeric label.

    .. code-block:: console
    
       fx collaborator generate-cert-request -n COL.LABEL


 The creation script will also ask you to specify the path to the data. For this example, enter the integer that represents which MNIST shard to use on this collaborator node. For the first collaborator node enter **1**. For the second collaborator node enter **2**.

 This will create the following files:

    +-----------------------------+------------------------------------------------------+
    | File Type                   | Filename                                             |
    +=============================+======================================================+
    | Collaborator CSR            | WORKSPACE.PATH/cert/client/col_COL.LABEL.csr         |
    +-----------------------------+------------------------------------------------------+
    | Collaborator key            | WORKSPACE.PATH/cert/client/col_COL.LABEL.key         |
    +-----------------------------+------------------------------------------------------+
    | Collaborator CSR Package    | WORKSPACE.PATH/col_COL.LABEL_to_agg_cert_request.zip |
    +-----------------------------+------------------------------------------------------+


4. On the aggregator node (i.e., the certificate authority in this example), sign the Collaborator CSR Package from the collaborator nodes.
   
    .. code-block:: console
        
       fx collaborator certify --request-pkg /PATH/TO/col_COL.LABEL_to_agg_cert_request.zip
          
   where :code:`/PATH/TO/col_COL.LABEL_to_agg_cert_request.zip` is the path to the Collaborator CSR Package containing the :code:`.csr` file from the collaborator node. The certificate authority will sign this certificate for use in the federation.

   The command packages the signed collaborator certificate, along with the **cert_chain.crt** file needed to verify certificate signatures, for transport back to the collaborator node:

    +---------------------------------+----------------------------------------------------------+
    | File Type                       | Filename                                                 |
    +=================================+==========================================================+
    | Certificate and Chain Package   | WORKSPACE.PATH/agg_to_col_COL.LABEL_signed_cert.zip      |
    +---------------------------------+----------------------------------------------------------+

5. On the collaborator node, import the signed certificate and certificate chain into your workspace. 

    .. code-block:: console
        
       fx collaborator certify --import /PATH/TO/agg_to_col_COL.LABEL_signed_cert.zip

