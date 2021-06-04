.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _install_initial_steps:

Initial Steps
#############

On every node in the federation you will need to install the |productName| package.

1. Install a Python 3.6 (or higher) virtual environment. Conda (version 4.9 or above) is preferred, but other virtual environments should work as well.
   Conda can either be installed via the `Anaconda <https://www.anaconda.com/products/individual>`_
   or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ distributions.

   Once :code:`conda` is installed you may need to logout and back in for the changes to take effect.

2. Create a new conda environment for the project.

   .. code-block:: console

      $ conda create -n env_name pip python=3.6

3. Activate the virtual environment.

   .. code-block:: console

      $ conda activate env_name

4. Install |productName| package:

   **Please Note:** PyPI (pip) package is currently (May 2021) provided ONLY for Linux OS. **For installation on Windows and MacOS, please follow the manual installation from source.** 

   A. **Linux** installation: 

      .. parsed-literal::

         $ pip install \ |productWheel|\
   
   B. **Windows** (and probably **MacOS**) installation:

      i) Clone |productName| repository:

         .. code-block:: console

           $ git clone https://github.com/intel/openfl.git 

      ii) Inside your python environment, call `pip install`: 

         .. code-block:: console

            $ cd openfl/
            $ pip install .



5. At this point |productName| should be available within the virtual environment. To test, run the :code:`fx` command. This command is only available within this virtual environment.

   .. figure:: images/fx_help.png
      :scale: 70 %

.. centered:: fx command
