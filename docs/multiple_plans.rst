.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _multiple_plans:

***********************
Manage Multiple Plans
***********************

With aggregator-based workflow, you can use multiple Federated Learning plans (FL plan) for the same workspace. All FL plans are located in the **WORKSPACE.FOLDER/plan/plans** directory. 

The following are the :code:`fx` commands to manage your FL plans:

    - :ref:`creating_new_plans`
    - :ref:`saving_new_plans`
    - :ref:`switching_plans`
    - :ref:`removing_plans`
    
.. _creating_new_plans:

Create a New FL Plan
====================

All workspaces begin with a :code:`default` FL plan. See :ref:`Create a Workspace on the Aggregator <creating_workspaces>` for details.

.. _saving_new_plans:

Save a New FL Plan
==================

When you are working on an FL plan, you can save it for future use.

    .. code-block:: console
    
       fx plan save -n NEW.PLAN.NAME
      
 
    where :code:`NEW.PLAN.NAME` is the new FL plan for your workspace. 
    This command also combines switching to the :code:`NEW.PLAN.NAME` plan.
    
.. _switching_plans:

Switch FL Plans
===============

To switch to a different FL plan, run the following command from the workspace directory.

    .. code-block:: console
    
       fx plan switch -n PLAN.NAME

    where :code:`PLAN.NAME` is the FL plan to which you want to switch. 

    .. note::

       If you have changed the **plan.yaml** file, you should :ref:`save the FL plan <creating_new_plans>` before switching. Otherwise, any changes will be lost.
       
.. _removing_plans:

Remove FL Plans
===============

To remove an FL plan, run the following command from the workspace directory.

    .. code-block:: console
    
        fx plan remove -n PLAN.NAME

    where :code:`PLAN.NAME` is the FL plan you wish to remove. 
