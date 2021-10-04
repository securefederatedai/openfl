.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _plan_settings:

***************
Plan Settings
***************

The Federated Learning plan (FL plan) is described by the **plan.yaml** file located in the **plan** directory of the workspace.


Each YAML top-level section contains the following subsections:

- ``template``: name of the class including top-level packages names. An instance of this class is created when plan gets initialized.
- ``settings``: arguments that are passed to the class constructor
- ``defaults``: file that contains default settings for this subsection.
  Any setting from defaults file can be overriden in the **plan.yaml** file.

The following is an example of a **plan.yaml**:

.. literalinclude:: ../openfl-workspace/torch_cnn_mnist/plan/plan.yaml
  :language: yaml


Configurable Settings
=====================

- :class:`Aggregator <openfl.component.Aggregator>`
- :class:`Collaborator <openfl.component.Collaborator>`
- :class:`Data Loader <openfl.federated.data.loader.DataLoader>`
- :class:`Task Runner <openfl.federated.task.runner.TaskRunner>`
- :class:`Assigner <openfl.component.Assigner>`


Tasks
-----

Each task subsection contains the following:

- ``function``: function name to call.
  The function must be the one defined in :class:`TaskRunner <openfl.federated.TaskRunner>` class.
- ``kwargs``: kwargs passed to the ``function``.