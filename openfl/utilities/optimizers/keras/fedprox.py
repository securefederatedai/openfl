# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FedProx Keras optimizer module."""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.ops import standard_ops


@keras.utils.register_keras_serializable()
class FedProxOptimizer(keras.optimizers.Optimizer):
    """FedProx optimizer.

    Paper: https://arxiv.org/pdf/1812.06127.pdf
    """

    def __init__(self, learning_rate=0.01, mu=0.01, name='FedProxOptimizer', **kwargs):
        """Initialize."""
        super().__init__(name=name, **kwargs)

        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('mu', mu)

        self._lr_t = None
        self._mu_t = None

    def _prepare(self, var_list):
        self._lr_t = tf.convert_to_tensor(self._get_hyper('learning_rate'), name='lr')
        self._mu_t = tf.convert_to_tensor(self._get_hyper('mu'), name='mu')

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, 'vstar')

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = tf.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, 'vstar')

        var_update = var.assign_sub(lr_t * (grad + mu_t * (var - vstar)))

        return tf.group(*[var_update, ])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = tf.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, 'vstar')
        v_diff = vstar.assign(mu_t * (var - vstar), use_locking=self._use_locking)

        with tf.control_dependencies([v_diff]):
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
            'lr': self._serialize_hyperparameter('learning_rate'),
            'mu': self._serialize_hyperparameter('mu')
        }
