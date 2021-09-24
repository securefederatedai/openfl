.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _install_package:

*********************************
Install the |productName| Package
*********************************

Perform this procedure on every node in the federation.

1. Install a Python\* \  3.6 (or higher) virtual environment. Conda\* \  (version 4.9 or above) is preferred, but other virtual environments should work as well.
   Conda can either be installed via the `Anaconda <https://www.anaconda.com/products/individual>`_\* \  or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_\* \  distributions. 
   
 See the `conda installation guide <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ for details.

2. Create a new conda environment for the project.

   .. code-block:: console

      conda create -n env_name pip python=3.6

3. Activate the virtual environment.

   .. code-block:: console

      conda activate env_name

4. Install the |productName| package.

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



5. Run the :code:`fx` command in the virtual environment to confirm |productName| is available.

   .. figure:: images/fx_help.png
      :scale: 70 %

.. centered:: Output of the fx Command
