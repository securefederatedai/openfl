.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

*********************************
Supported aggregation algorithms
*********************************
===========
FedAvg
===========
Default aggregation algorithm in OpenFL.
Multiplies local model weights with relative data size and averages this multiplication result.

=========
FedProx
=========
Paper: https://arxiv.org/abs/1812.06127

FedProx in OpenFL is implemented as a custom optimizer for PyTorch/TensorFlow. In order to use FedProx, do the following:

1. PyTorch:

  - replace your optimizer with SGD-based :class:`openfl.utilities.optimizers.torch.FedProxOptimizer` 
    or Adam-based :class:`openfl.utilities.optimizers.torch.FedProxAdam`.
    Also, you should save model weights for the next round via calling `.set_old_weights()` method of the optimizer
    before the training epoch.

2. TensorFlow:

  - replace your optimizer with SGD-based :py:class:`openfl.utilities.optimizers.keras.FedProxOptimizer`.

For more details, see :code:`openfl-tutorials/Federated_FedProx_*_MNIST_Tutorial.ipynb` where * is the framework name.

=========
FedOpt
=========
Paper: https://arxiv.org/abs/2003.00295

FedOpt in OpenFL: :ref:`adaptive_aggregation_functions`

==========
FedCurv 
==========
Paper: https://arxiv.org/abs/1910.07796

Requires PyTorch >= 1.9.0. Other frameworks are not supported yet.

Use :py:class:`openfl.utilities.fedcurv.torch.FedCurv` to override train function using :code:`.get_penalty()`, :code:`.on_train_begin()`, and :code:`.on_train_end()` methods.
In addition, you should override default :code:`AggregationFunction` of the train task with :class:`openfl.interface.aggregation_functions.FedCurvWeightedAverage`.
See :code:`PyTorch_Histology_FedCurv` tutorial in :code:`openfl-tutorials/interactive_api` directory for more details.