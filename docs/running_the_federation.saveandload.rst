.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_saveandload:

Saving and Loading Models
#########################

The models trained in federation (or other pre-trained models) can be saved once the federation is over so that the next time federation is run, the previously saved model is picked up.
The functionality of saving and loading models are available in these templates:

 - :code:`keras_cnn_mnist`: workspace with a simple `Keras <http://keras.io/>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.
 - :code:`torch_cnn_mnist`: workspace with a simple `PyTorch <http://pytorch.org>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.

Saving the model
~~~~~~~~~~~~~~~~
:code:`save_native` is used to save a Tensorflow or pytorch model at the end of federation. For a pytorch model, the model state dict and optimizer state dict are stored.

.. code-block:: console

    $ model.save_native(filepath)


For the saved model to get picked up in the next federation experiment, the model should be saved under `model` directory.

For template :code:`keras_cnn_mnist` the filepath should be `model/saved_model`

For template :code:`torch_cnn_mnist` the filepath should be `model/saved_model.pth`


Loading the model
~~~~~~~~~~~~~~~~~

Any model that is stored under `model/saved_model` in case of a Tensorflow model or `model/saved_model.pth` in case of a Pytorch model gets picked up by the federation.
Therefore, instead of building the model, the saved model is loaded and federation is performed.

For PyTorch models, a loss function can be specified as an argument in `plan.yaml` (under task_runner/settings/loss), otherwise the default metric of binary cross-entropy is taken.
Acceptable loss functions are: L1Loss, MSELoss, NLLLoss, GaussianNLLLoss, KLDivLoss, SmoothL1Loss, CrossEntropyLoss.
