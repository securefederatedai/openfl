.. # Copyright (C) 2020-2024 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

==========
Features
==========

.. _running_a_federation:

---------------------
Running a Federation
---------------------

|productName| has multiple options for setting up a federation and running experiments, depending on the users needs. 

Task Runner
    Define an experiment and distribute it manually. All participants can verify model code and FL plan prior to execution. 
    The federation is terminated when the experiment is finished. Formerly known as the aggregator-based workflow.
    For more info see :doc:`features_index/taskrunner`

    .. toctree::
        :hidden:

        features_index/taskrunner

Interactive (Deprecated)
    Setup long-lived components to run many experiments in series. Recommended for FL research when many changes to model, dataloader, or hyperparameters are expected.
    Formerly known as the director-based workflow.
    For more info see :doc:`features_index/interactive`

    .. toctree::
        :hidden:

        features_index/interactive

Workflow Interface (Experimental)
    Formulate the experiment as a series of tasks, or a flow. Every flow begins with the start task and concludes with end.
    Heavily influenced by the interface and design of Netflix's Metaflow, the popular framework for data scientists. 
    For more info see :doc:`features_index/workflowinterface`

    .. toctree::
        :hidden:

        features_index/workflowinterface

.. _aggregation_algorithms:

-----------------------
Aggregation Algorithms
-----------------------

FedAvg
    Paper: `McMahan et al., 2017 <https://arxiv.org/pdf/1602.05629.pdf>`_
    Default aggregation algorithm in |productName|. Multiplies local model weights with relative data size and averages this multiplication result.

FedProx
    Paper: `Li et al., 2020 <https://arxiv.org/abs/1812.06127>`_

    FedProx in |productName| is implemented as a custom optimizer for PyTorch/TensorFlow. In order to use FedProx, do the following:

    1. PyTorch:

    - replace your optimizer with SGD-based :class:`openfl.utilities.optimizers.torch.FedProxOptimizer` 
        or Adam-based :class:`openfl.utilities.optimizers.torch.FedProxAdam`.
        Also, you should save model weights for the next round via calling `.set_old_weights()` method of the optimizer
        before the training epoch.

    2. TensorFlow:

    - replace your optimizer with SGD-based :py:class:`openfl.utilities.optimizers.keras.FedProxOptimizer`.

    For more details, see :code:`../openfl-tutorials/Federated_FedProx_*_MNIST_Tutorial.ipynb` where * is the framework name.

FedOpt
    Paper: `Reddi et al., 2020 <https://arxiv.org/abs/2003.00295>`_

    FedOpt in |productName|: :ref:`adaptive_aggregation_functions`

FedCurv
    Paper: `Shoham et al., 2019 <https://arxiv.org/abs/1910.07796>`_

    Requires PyTorch >= 1.9.0. Other frameworks are not supported yet.

    Use :py:class:`openfl.utilities.fedcurv.torch.FedCurv` to override train function using :code:`.get_penalty()`, :code:`.on_train_begin()`, and :code:`.on_train_end()` methods.
    In addition, you should override default :code:`AggregationFunction` of the train task with :class:`openfl.interface.aggregation_functions.FedCurvWeightedAverage`.

.. _federated_evaluation:

---------------------
Federated Evaluation
---------------------

Evaluate the accuracy and performance of your model on data distributed across decentralized nodes without comprimising data privacy and security. For more info see :doc:`features_index/fed_eval`

.. toctree::
    :hidden:

    features_index/fed_eval

.. _privacy_meter:

---------------------
Privacy Meter
---------------------

Quantitatively audit data privacy in statistical and machine learning algorithms. For more info see :doc:`features_index/privacy_meter`
    
.. toctree::
    :hidden:

    features_index/privacy_meter
    
