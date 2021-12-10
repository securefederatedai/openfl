.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _install_software_root:

=====================
Installation
=====================

Depending on how you want to set up |productName|, choose one of the following installation procedure.


.. toctree::
   :maxdepth: 2
   :hidden:

   `install_package`_
   `install_docker`_


.. _install_package:

*********************************
Install the Package
*********************************

Follow this procedure to prepare the environment and install the |productName| package.
Perform this procedure on every node in the federation.

1. Install a Python 3.8 (>=3.6, <3.9) virtual environment using venv.
   
 See the `Venv installation guide <https://docs.python.org/3/library/venv.html>`_ for details.

2. Create a new Virtualenv environment for the project.

   .. code-block:: console

      python3 -m venv venv

3. Activate the virtual environment.

   .. code-block:: console

      source venv/bin/activate

4. Install the |productName| package.

    A. Installation from PyPI: 
    
        .. code-block:: console
        
            pip install openfl
   
    B. Installation from source:

        #. Clone the |productName| repository:
        
            .. code-block:: console
            
                git clone https://github.com/intel/openfl.git 


        #. From inside the Python environment, call :code:`pip install`: 

            .. code-block:: console
            
                cd openfl/
                pip install .



5. Run the :code:`fx` command in the virtual environment to confirm |productName| is installed.

   .. figure:: images/fx_help.png
      :scale: 70 %

.. centered:: Output of the fx Command


.. _install_docker:

****************************************
|productName| with Docker\* \ 
****************************************

Follow this procedure to download or build a Docker\*\  image of |productName|, which you can use to run your federation in an isolated environment.

.. note::

   The Docker\* \  version of |productName| is to provide an isolated environment complete with the prerequisites to run a federation. When the execution is over, the container can be destroyed and the results of the computation will be available on a directory on the local host.

1. Install Docker on all nodes in the federation.

 See the `Docker installation guide <https://docs.docker.com/engine/install/>`_ for details. 

2. Check that Docker is running properly with the *Hello World* command:

    .. code-block:: console

      $ docker run hello-world
      Hello from Docker!
      This message shows that your installation appears to be working correctly.
      ...
      ...
      ...
      
3. Build an image from the latest official |productName| release:

	.. code-block:: console

	   docker pull intel/openfl
   
	If you prefer to build an image from a specific commit or branch, perform the following commands:

	.. code-block:: console

	   git clone https://github.com/intel/openfl.git
	   cd openfl
	   docker build -f openfl-docker/Dockerfile.base .