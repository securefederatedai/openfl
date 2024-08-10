# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""FedProx Keras optimizer module."""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.ops import standard_ops


@keras.utils.register_keras_serializable()
class FedProxOptimizer(keras.optimizers.Optimizer):
    """FedProx optimizer.

    Implements the FedProx algorithm as a Keras optimizer. FedProx is a
    federated learning optimization algorithm designed to handle non-IID data.
    It introduces a proximal term to the federated averaging algorithm to
    reduce the impact of devices with outlying updates.

    Paper: https://arxiv.org/pdf/1812.06127.pdf

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        mu (float): The proximal term coefficient.
    """

    def __init__(self, learning_rate=0.01, mu=0.01, name="FedProxOptimizer", **kwargs):
        """
        Initialize the FedProxOptimizer.

        Args:
            learning_rate (float, optional): The learning rate for the
                optimizer. Defaults to 0.01.
            mu (float, optional): The proximal term coefficient. Defaults}
                to 0.01.
            name (str, optional): The name of the optimizer. Defaults to
                'FedProxOptimizer'.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, **kwargs)

        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("mu", mu)

        self._lr_t = None
        self._mu_t = None

    def _prepare(self, var_list):
        """
        Prepare the optimizer's state.

        Args:
            var_list (list): List of variables to be optimized.
        """
        self._lr_t = tf.convert_to_tensor(self._get_hyper("learning_rate"), name="lr")
        self._mu_t = tf.convert_to_tensor(self._get_hyper("mu"), name="mu")

    def _create_slots(self, var_list):
        """Create slots for the optimizer's state.

        Args:
            var_list (list): List of variables to be optimized.
        """
        for v in var_list:
            self.add_slot(v, "vstar")

    def _resource_apply_dense(self, grad, var):
        """Apply gradients to variables.

        Args:
            grad: Gradients.
            var: Variables.

        Returns:
            A tf.Operation that applies the specified gradients.
        """
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = tf.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        var_update = var.assign_sub(lr_t * (grad + mu_t * (var - vstar)))

        return tf.group(
            *[
                var_update,
            ]
        )

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        """Apply sparse gradients to variables.

        Args:
            grad: Gradients.
            var: Variables.
            indices: A tensor of indices into the first dimension of `var`.
            scatter_add: A scatter operation.

        Returns:
            A tf.Operation that applies the specified gradients.
        """
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = tf.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")
        v_diff = vstar.assign(mu_t * (var - vstar), use_locking=self._use_locking)

        with tf.control_dependencies([v_diff]):
            scaled_grad = scatter_add(vstar, indices, grad)
        var_update = var.assign_sub(lr_t * scaled_grad)

        return tf.group(
            *[
                var_update,
            ]
        )

    def _resource_apply_sparse(self, grad, var):
        """
        Apply sparse gradients to variables.

        Args:
            grad: Gradients.
            var: Variables.

        Returns:
            A tf.Operation that applies the specified gradients.
        """
        return self._apply_sparse_shared(grad.values, var, grad.indices, standard_ops.scatter_add)

    def get_config(self):
        """Return the config of the optimizer.

        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.

        Returns:
            dict: The optimizer configuration.
        """
        base_config = super().get_config()
        return {
            **base_config,
            "lr": self._serialize_hyperparameter("learning_rate"),
            "mu": self._serialize_hyperparameter("mu"),
        }
