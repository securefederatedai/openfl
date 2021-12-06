.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _director_communications:

***************************************
Director Service Communication Diagrams
***************************************

The following diagrams depict existing procedure calls to the Director service. Included are interactions with the Director's inner representations to better understand their signatures.

Director-Envoy Communication
============================

The following diagram depicts a typical process of establishing a Federation and registering an experiment.  

.. kroki:: director_envoy.mmd
    :caption: Basic Scenario of Director-Envoy Communication
    :align: center
    :type: mermaid

Director Side Envoy Representation and Related Remote Procedure Calls
=====================================================================

This diagram shows possible interactions with Envoy handles on the Director side.

.. kroki:: envoy_representation_and_RPCs.mmd
    :caption: Communications Altering or Requesting Envoy-Related Information
    :align: center
    :type: mermaid

Director Side Experiment Representation and Related Remote Procedure Calls
==========================================================================

This diagram shows possible interactions with Experiment handles on the Director side.

.. kroki:: experiment_representation_and_RPCs.mmd
    :caption: Communications Altering or Requesting Experiment-Related Information
    :align: center
    :type: mermaid