.. # Copyright (C) 2022-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _straggler_handling_algorithms:

*****************************
Straggler Handling Interface
*****************************

The Open Federated Learning (|productName|) framework supports straggler handling interface for identifying stragglers or slow collaborators for a round and ending the round early as a result of it. The updates from these stragglers are not aggregated in the aggregator model.

The following are the straggler handling algorithms supported in |productName|:

``CutoffTimeBasedStragglerHandling``
    Identifies stragglers based on the cutoff time specified in the settings. Arguments to the function are:
        - *Cutoff Time* (straggler_cutoff_time), specifies the cutoff time by which the aggregator should end the round early.
        - *Minimum Reporting* (minimum_reporting), specifies the minimum number of collaborators needed to aggregate the model.
    
``PercentageBasedStragglerHandling``
    Identifies stragglers based on the percetage specified. Arguments to the function are:
        - *Percentage of collaborators* (percent_collaborators_needed), specifies a percentage of collaborators enough to end the round early.
        - *Minimum Reporting* (minimum_reporting), specifies the minimum number of collaborators needed to aggregate the model.
    

Demonstration of adding the straggler handling interface
=========================================================

The example template, **torch_cnn_mnist_straggler_check**, uses the ``PercentageBasedStragglerHandling``. To gain a better understanding of how experiments perform, you can modify the **percent_collaborators_needed** or **minimum_reporting** parameter in the template **plan.yaml** or even choose **CutoffTimeBasedStragglerHandling** function instead:

    .. code-block:: console
    
        straggler_handling_policy :
            template : openfl.component.straggler_handling_functions.CutoffTimeBasedStragglerHandling
            settings :
                straggler_cutoff_time : 20
                minimum_reporting : 1

