# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""FedProx Keras optimizer module."""

import tensorflow as tf
import tensorflow.keras as keras


@keras.utils.register_keras_serializable()
class FedProxOptimizer(keras.optimizers.Optimizer):
    """FedProx optimizer (Keras3 based API).

    Implements the FedProx algorithm as a Keras optimizer. FedProx is a
    federated learning optimization algorithm designed to handle non-IID data.
    It introduces a proximal term to the federated averaging algorithm to
    reduce the impact of devices with outlying updates.

    Paper: https://arxiv.org/pdf/1812.06127.pdf

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        mu (float): The proximal term coefficient.
    """

    def __init__(
        self,
        learning_rate=0.01,
        mu=0.0,
        name="FedProxOptimizer",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            **kwargs,
        )
        self.mu = mu

    def build(self, variables):
        """Initialize optimizer variables.

        Args:
            variables (list): List of model variables to build FedProx variables on.
        """
        if self.built:
            return
        super().build(variables)
        self.vstars = []
        for variable in variables:
            self.vstars.append(
                self.add_variable_from_reference(reference_variable=variable, name="vstar")
            )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable.
            In the update_step method, variable is updated using the
            gradient and the proximal term (mu). The proximal term helps
            to regularize the update by considering the difference between
            the current value of variable and its initial value (vstar),
            which was stored during the build method.
        Args:
            gradient (tf.Tensor): The gradient tensor for the variable.
            variable (tf.Variable): The model variable to be updated.
            learning_rate (float): The learning rate for the update step.
        """
        lr_t = tf.cast(learning_rate, variable.dtype)
        mu_t = tf.cast(self.mu, variable.dtype)
        gradient_t = tf.cast(gradient, variable.dtype)
        # Get the corresponding vstar for the current variable
        vstar = self.vstars[self._get_variable_index(variable)]

        # Update the variable using the gradient and the proximal term
        self.assign_sub(variable, lr_t * (gradient_t + mu_t * (variable - vstar)))

    def get_config(self):
        """Return the config of the optimizer.
        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.
        Returns:
            dict: The optimizer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "mu": self.mu,
            }
        )
        return config
