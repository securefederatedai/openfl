.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0
.. not used

*****************
Design Philosophy
*****************

The overall design centers around the *Federated Learning (FL) Plan*.
The plan is just a `YAML <https://en.wikipedia.org/wiki/YAML>`_
file that defines the
collaborators, aggregator, connections, models, data,
and any other parameters that describes how the training will evolve.
In the “Hello Federation” demos, the plan will be located in the
YAML file within the federation project's workspace: *plan/plan.yaml*.
As you modify federation workspace, you’ll effectively
just be editing the plan along with the Python code defining
the model and the data loader in order to meet your requirements.
