.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _straggler_handling_algorithms:

*****************************
Straggler Handling Interface
*****************************

The Open Federated Learning (OpenFL) framework supports straggler handling interface for identifying stragglers or slow collaborators for a round and ending the round early as a result of it. The updates from these stragglers are not aggregated in the aggregator model.

The following are the straggler handling algorithms supported in OpenFL:

``CutoffTimeBasedStragglerHandling``
    Identifies stragglers based on the cutoff time specified in the settings. Arguments to the function are:
        - *Cutoff Time* (straggler_cutoff_time), specifies the cutoff time by which the aggregator should end the round early.
        - *Minimum Reporting* (minimum_reporting), specifies the minimum number of collaborators needed to aggregate the model.

    For example, in a federation of 5 collaborators, if :code:`straggler_cutoff_time` (in seconds) is set to 20 and :code:`minimum_reporting` is set to 2, atleast 2 collaborators (or more) would be included in the round, provided that the time limit of 20 seconds is not exceeded.
    In an event where :code:`minimum_reporting` collaborators don't make it within the :code:`straggler_cutoff_time`, the straggler handling policy is disregarded. 

``PercentageBasedStragglerHandling``
    Identifies stragglers based on the percetage specified. Arguments to the function are:
        - *Percentage of collaborators* (percent_collaborators_needed), specifies a percentage of collaborators enough to end the round early.
        - *Minimum Reporting* (minimum_reporting), specifies the minimum number of collaborators needed to aggregate the model.

    For example, in a federation of 5 collaborators, if :code:`percent_collaborators_needed` is set to 0.8 and :code:`minimum_reporting` is set to 1, the first 4 collaborators (80%) to report to aggregator are selected for that round.   

Demonstration of adding the straggler handling interface
=========================================================

The example template, **torch_cnn_mnist_straggler_check**, uses the ``PercentageBasedStragglerHandling``. To gain a better understanding of how experiments perform, you can modify the **percent_collaborators_needed** or **minimum_reporting** parameter in the template **plan.yaml** or even choose **CutoffTimeBasedStragglerHandling** function instead:

    .. code-block:: yaml
    
        straggler_handling_policy :
            template : openfl.component.straggler_handling_functions.CutoffTimeBasedStragglerHandling
            settings :
                straggler_cutoff_time : 20
                minimum_reporting : 1
