.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _quick_start:

=====================
Quick Start
=====================

|productName| has a variety of APIs to choose from when setting up and running a federation. 
In this quick start guide, we will demonstrate how to run a simple federated learning example using the Task Runner API and Hello Federation script

.. note::

    The example used in this section is designed primarily to demonstrate functionality of the package and its components. It is not the recommended method for running a real world federation.

    See :ref:`openfl_examples` for details.

.. _hello_federation:

*********************************
Hello Federation
*********************************
.. note::

    Ensure you have installed the |productName| package.

    See :ref:`install_package` for details.

We will use the `"Hello Federation" python script <https://github.com/intel/openfl/blob/develop/tests/github/test_hello_federation.py>`_ to quickly create a federation (an aggregator node and two collaborator nodes) to test the project pipeline.

.. literalinclude:: ../../tests/github/test_hello_federation.py
  :language: python

Run the script

.. code-block:: console

    python test_hello_federation.py