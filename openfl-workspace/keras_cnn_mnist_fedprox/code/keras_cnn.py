# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import tensorflow.keras as ke
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from tensorflow.python.ops import standard_ops

from openfl.federated import KerasTaskRunner


class FedProxOptimizer(ke.optimizers.Optimizer):
    """Custom Optimizer."""

    def __init__(self, lr=0.01, mu=0.01, name="PGD", **kwargs):
        """Initialize."""
        super().__init__(name, **kwargs)

        self._set_hyper("lr", lr)
        self._set_hyper("mu", mu)

        self._lr_t = None
        self._mu_t = None

    def _prepare(self, var_list):
        self._lr_t = tf.convert_to_tensor(self._get_hyper('lr'), name="lr")
        self._mu_t = tf.convert_to_tensor(self._get_hyper('mu'), name="mu")

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, "vstar")

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = tf.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        var_update = var.assign_sub(lr_t * (grad + mu_t * (var - vstar)))

        return tf.group(*[var_update, ])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = tf.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")
        v_diff = vstar.assign(mu_t * (var - vstar), use_locking=self._use_locking)

        with tf.control_dependencies([v_diff]):  # run v_diff operation before scatter_add
            scaled_grad = scatter_add(vstar, indices, grad)
        var_update = var.assign_sub(lr_t * scaled_grad)

        return tf.group(*[var_update, ])

    def _resource_apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: standard_ops.scatter_add(x, i, v))

    def get_config(self):
        """Return the config of the optimizer.

        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.

        Returns:
            Python dictionary.
        """
        base_config = super(FedProxOptimizer, self).get_config()
        return {
            **base_config,
            "lr": self._serialize_hyperparameter("lr"),
            "mu": self._serialize_hyperparameter("mu")
        }


class KerasCNN(KerasTaskRunner):
    """A basic convolutional neural network model."""

    def __init__(self, **kwargs):
        """
        Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        self.model = self.build_model(self.feature_shape, self.data_loader.num_classes, **kwargs)

        self.initialize_tensorkeys_for_functions()

        self.model.summary(print_fn=self.logger.info)

        if self.data_loader is not None:
            self.logger.info(f'Train Set Size : {self.get_train_data_size()}')
            self.logger.info(f'Valid Set Size : {self.get_valid_data_size()}')

    def build_model(self,
                    input_shape,
                    num_classes,
                    conv_kernel_size=(4, 4),
                    conv_strides=(2, 2),
                    conv1_channels_out=16,
                    conv2_channels_out=32,
                    final_dense_inputsize=100,
                    **kwargs):
        """
        Define the model architecture.

        Args:
            input_shape (numpy.ndarray): The shape of the data
            num_classes (int): The number of classes of the dataset

        Returns:
            tensorflow.python.keras.engine.sequential.Sequential: The model defined in Keras

        """
        model = Sequential()

        model.add(Conv2D(conv1_channels_out,
                         kernel_size=conv_kernel_size,
                         strides=conv_strides,
                         activation='relu',
                         input_shape=input_shape))

        model.add(Conv2D(conv2_channels_out,
                         kernel_size=conv_kernel_size,
                         strides=conv_strides,
                         activation='relu'))

        model.add(Flatten())

        model.add(Dense(final_dense_inputsize, activation='relu'))

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=ke.losses.categorical_crossentropy,
                      optimizer=FedProxOptimizer(),
                      metrics=['accuracy'])

        # initialize the optimizer variables
        opt_vars = model.optimizer.variables()

        for v in opt_vars:
            v.initializer.run(session=self.sess)

        return model
