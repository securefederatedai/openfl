.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _director_communications:

******
|productName| Director service communication diagrams
******

Following diagrams depict existing procedure calls to the Director service.
Interactions with Director's inner representations created to better understand their signatures.

Director-Envoy communication
##########

The following diagram depicts a typical process of establishing a Federation and registering an experiment.  

.. kroki:: director_envoy.mmd
    :caption: Basic scenario of Director-Envoy communication
    :align: center
    :type: mermaid

Director's Envoy representation and related RPCs
##########

This diagram shows possible interactions with Envoy handles on the Director side.

.. kroki:: envoy_representation_and_RPCs.mmd
    :caption: Communications altering / requesting Envoy-related information
    :align: center
    :type: mermaid

Director's Experiment representation and related RPCs
##########

This diagram shows possible interactions with Experiment handles on the Director side.

.. kroki:: experiment_representation_and_RPCs.mmd
    :caption: Communications altering / requesting Experiment-related information
    :align: center
    :type: mermaid