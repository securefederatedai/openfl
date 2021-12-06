.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _install_package:

*********************************
Install the Package
*********************************

Perform this procedure on every node in the federation.

1. Install a Python\* \  3.6 (or higher) virtual environment. 
   
 See the `Virtualenv installation guide <https://virtualenv.pypa.io/en/latest/installation.html>`_ for details.

2. Create a new Virtualenv environment for the project.

   .. code-block:: console

      python3 -m virtualenv env_name

3. Activate the virtual environment.

   .. code-block:: console

      source env_name/bin/activate

4. Install the Open Federated Learning (|productName|) package.

    A. **Linux**\* \  installation: 
    
        .. parsed-literal::
        
            pip install \ |productWheel|\
   
    B. **Windows**\* \  (and probably **macOS**\* \) installation:

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
