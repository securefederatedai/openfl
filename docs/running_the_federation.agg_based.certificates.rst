.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

**************************
Configuring the Federation
**************************

`TLS <https://en.wikipedia.org/wiki/Transport_Layer_Security>`_ encryption is
used for the network connections.
Therefore, security keys and certificates will need to be created for the
aggregator and collaborators
to negotiate the connection securely. For the :ref:`Hello Federation <running_the_federation>` demo
we will run the aggregator and collaborators on the same localhost server
so these configuration steps just need to be done once on that machine. We have two pki
workflows: manual and semi-automatic (with step-ca).

    .. note::
    
       Different certificates can be created for each project workspace.

.. _install_certs:

.. kroki:: mermaid/CSR_signing.mmd
    :caption: Manual certificate generation and signing
    :align: center
    :type: mermaid

.. _install_certs:

.. kroki:: mermaid/pki_scheme.mmd
    :caption: Step-ca certificate generation and signing
    :align: center
    :type: mermaid

    
.. _install_certs_agg:

On the Aggregator Node
######################

Before you run the federation make sure you have activated a Python virtual environment (e.g. :code:`conda activate`), installed the |productName| package
:ref:`using these instructions <install_initial_steps>`, and are in the correct directory for the :ref:`project workspace <creating_workspaces>`.

1. Change directory to the path for your project's workspace:

    .. code-block:: console
    
       $ cd WORKSPACE.PATH

2. Run the Certificate Authority command. This will setup the Aggregator node as the `Certificate Authority <https://en.wikipedia.org/wiki/Certificate_authority>`_ for the Federation. All certificates will be signed by the aggregator. Follow the command-line instructions and enter in the information as prompted. The command will create a simple database file to keep track of all issued certificates. 

    .. code-block:: console
    
       $ fx workspace certify

3. Run the aggregator certificate creation command, replacing **AFQDN** with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ for the aggregator machine. Alternatively, you can override the apparent FQDN of the system by setting an FQDN environment variable (:code:`export FQDN=x.x.x.x`) before creating the certificate.

    .. code-block:: console
    
       $ fx aggregator generate-cert-request --fqdn AFQDN
       
    .. note::
    
       On Linux, you can discover the FQDN with the command:
    
           .. code-block:: console
        
              $ hostname --all-fqdns | awk '{print $1}'
            
   .. note::
   
      If you omit the :code:`--fdqn` parameter, then :code:`fx` will automatically use the FQDN of the current node assuming the node has been correctly set with a static address. 
   
      .. code-block:: console
    
         $ fx aggregator generate-cert-request
       
4. Run the aggregator certificate signing command, replacing **AFQDN** with the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ for the aggregator machine. Alternatively, you can override the apparent FQDN of the system by setting an FQDN environment variable (:code:`export FQDN=x.x.x.x`) before signing the certificate.

    .. code-block:: console
    
       $ fx aggregator certify --fqdn AFQDN

5. This node now has a signed security certificate as the aggreator for this new federation. You should have the following files.

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
~~~~~~~~~~~~~~~~~~~~~~~

1. Export the workspace so that it can be imported to the collaborator nodes.

    .. code-block:: console
    
       $ fx workspace export

   The :code:`export` command will archive the current workspace (as a :code:`zip`) and create a :code:`requirements.txt` file of the current Python packages in the virtual environment. Transfer this zip file to each collaborator node.

.. _install_certs_colab:

On the Collaborator Nodes
#########################

Before you run the federation make sure you have activated a Python virtual environment (e.g. :code:`conda activate`) and installed the |productName| package :ref:`using these instructions <install_initial_steps>`.

1. Make sure you have copied the :ref:`workspace archive <workspace_export>` (:code:`.zip`) from the aggregator node to the collaborator node.

2. Import the workspace archive using the following command:

    .. code-block:: console
    
       $ fx workspace import --archive WORKSPACE.zip

   where **WORKSPACE.zip** is the name of the workspace archive. This will unzip the workspace to the current directory and install the required Python packages within the current virtual environment.
   
3. For each test machine you want to run collaborators on, we create a collaborator certificate request to be signed by the certificate authority, replacing **COL.LABEL** with the label you've assigned to this collaborator. Note that this does not have to be the FQDN. It can be any unique alphanumeric label. 

    .. code-block:: console
    
       $ fx collaborator generate-cert-request -n COL.LABEL


   The creation script will also ask you to specify the path to the data. For the "Hello Federation" demo, simply enter the an integer that represents which shard of MNIST to use on this Collaborator For the first collaborator enter **1**. For the second collaborator enter **2**.
   This will create the following 3 files:

    +-----------------------------+------------------------------------------------------+
    | File Type                   | Filename                                             |
    +=============================+======================================================+
    | Collaborator CSR            | WORKSPACE.PATH/cert/client/col_COL.LABEL.csr         |
    +-----------------------------+------------------------------------------------------+
    | Collaborator key            | WORKSPACE.PATH/cert/client/col_COL.LABEL.key         |
    +-----------------------------+------------------------------------------------------+
    | Collaborator CSR Package    | WORKSPACE.PATH/col_COL.LABEL_to_agg_cert_request.zip |
    +-----------------------------+------------------------------------------------------+


    Only the Collaborator CSR Package file needs to be sent to the certificate authority to be signed. In this "Hello Federation" demo, the certificate authority is the Aggregator node.
       
4. On the Aggregator node (i.e. the Certificate Authority for this demo), run the following command:
   
    .. code-block:: console
        
       $ fx collaborator certify --request-pkg /PATH/TO/col_COL.LABEL_to_agg_cert_request.zip
          
   where **/PATH/TO/col_COL.LABEL_to_agg_cert_request.zip** is the path to the package containing the :code:`.csr` file from the collaborator. The Certificate Authority will sign this certificate for use in the Federation.

5. The previous command will package the signed collaborator certificate for transport back to the Collaborator node along with the :code:`cert_chain.crt` needed to verify certificate signatures. The only file needed to send back to the Collaborator node is the following:

    +---------------------------------+----------------------------------------------------------+
    | File Type                       | Filename                                                 |
    +=================================+==========================================================+
    | Certificate and Chain Package   | WORKSPACE.PATH/agg_to_col_COL.LABEL_signed_cert.zip      |
    +---------------------------------+----------------------------------------------------------+

6. Back on the Collaborator node, import the signed certificate and certificate chain into your workspace with this final command: 

    .. code-block:: console
        
       $ fx collaborator certify --import /PATH/TO/agg_to_col_COL.LABEL_signed_cert.zip

