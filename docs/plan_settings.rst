.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _plan_settings:

***************
Plan Settings
***************

The Federated Learning (FL) plan is described by ``plan.yaml`` file located in the ``plan`` folder of the workspace.


Each YAML top-level section contains 3 main subsections:

* ``template``: name of the class including top-level packages names.
  An instance of this class is created when plan gets initialized.
* ``settings``: arguments that are passed to the class constructor
* ``defaults``: file that contains default settings for this subsection.
  Any setting from defaults file can be overriden in ``plan.yaml`` file.

Example of ``plan.yaml``:

.. literalinclude:: ../fledge-workspace/torch_cnn_mnist/plan/plan.yaml
  :language: yaml

======================
Configurable settings
======================

- :class:`Aggregator <fledge.component.Aggregator>`
- :class:`Collaborator <fledge.component.Collaborator>`
- :class:`Data Loader <fledge.federated.data.loader.DataLoader>`
- :class:`Task Runner <fledge.federated.task.runner.TaskRunner>`
- :class:`Assigner <fledge.component.Assigner>`

++++++++++++++
Tasks
++++++++++++++

Each task subsection should contain:

- ``function``: function name to call.
  The function must be the one defined in :class:`TaskRunner <fledge.federated.TaskRunner>` class.
- ``kwargs``: kwargs passed to the ``function``.