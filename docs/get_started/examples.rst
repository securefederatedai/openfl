.. # Copyright (C) 2020-2024 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _openfl_examples:

=================================
Examples for Running a Federation
=================================

|productName| currently offers three ways to set up and run experiments with a federation: 
the Task Runner API, the Interactive API, and the experimental workflow interface. 
The Interactive API introduces a convenient way to set up a federation and brings “long-lived” components in a federation (“Director” and “Envoy”), 
while the Task Runner API workflow is advised for scenarios where the workload needs to be verified prior to execution. In contrast, the experimental workflow interface 
is introduce to provide significant flexility to researchers and developers in the construction of federated learning experiments.

-------------------------
Task Runner API
-------------------------
Formulate the experiment as a series of tasks, or a flow.

See :ref:`taskrunner_pytorch_mnist`

.. toctree::
    :hidden:
    :maxdepth: 1

    examples/taskrunner_pytorch_mnist

-------------------------
Interactive API
-------------------------
Setup long-lived components to run many experiments in series.

See :ref:`interactive_tensorflow_mnist`

.. toctree::
    :hidden:
    :maxdepth: 1

    examples/interactive_tensorflow_mnist

-------------------------
Workflow Interface
-------------------------
Formulate the experiment as a series of tasks, or a flow. 

See :ref:`workflowinterface_pytorch_mnist`

.. toctree::
    :hidden:
    :maxdepth: 1

    examples/workflowinterface_pytorch_mnist


.. note:: 

    Please visit `repository <https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials>`_ for a full list of tutorials