.. # Copyright (C) 2020-2024 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _openfl_examples:

=================================
Examples for Running a Federation
=================================

OpenFL currently offers four ways to set up and run experiments with a federation: 
the Task Runner API, Python Native API, the Interactive API (Deprecated), and the Workflow API. 
the Task Runner API is advised for production scenarios where the workload needs to be verified prior to execution, whereas the python native API provides a clean python interface on top of it intended for simulation purposes.
The Interactive API (Deprecated) introduces a convenient way to set up a federation and brings “long-lived” components in a federation (“Director” and “Envoy”), 
while the Task Runner API workflow is advised for scenarios where the workload needs to be verified prior to execution. In contrast, the currently experimental Workflow API
is introduced to provide significant flexility to researchers and developers in the construction of federated learning experiments.

As OpenFL nears it's 2.0 release, we expect to consolidate these APIs and make the Workflow API the primary interface going forward. See our `roadmap <https://github.com/securefederatedai/openfl/blob/develop/ROADMAP.md>`_ for more details. 

-------------------------
Task Runner API
-------------------------
Formulate the experiment as a series of tasks coordinated by a Federated Learning Plan

See :ref:`running_the_task_runner`

.. toctree::
    :hidden:
    :maxdepth: 1

    :ref:`running_the_task_runner`

-------------------------
Python Native API (Deprecated)
-------------------------
Intended for quick simulation purposes

See :ref:`python_native_pytorch_mnist`

.. toctree::
    :hidden:
    :maxdepth: 1

    examples/python_native_pytorch_mnist


----------------------------
Interactive API (Deprecated)
----------------------------
Setup long-lived components to run many experiments

See :ref:`interactive_tensorflow_mnist`

.. toctree::
    :hidden:
    :maxdepth: 1

    examples/interactive_tensorflow_mnist

---------------------------------
Workflow Interface (Experimental)
---------------------------------
Formulate the experiment as a series of tasks, or a flow. 

See :ref:`workflowinterface_pytorch_mnist`

.. toctree::
    :hidden:
    :maxdepth: 1

    examples/workflowinterface_pytorch_mnist


.. note:: 

    Please visit `repository <https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials>`_ for a full list of tutorials
