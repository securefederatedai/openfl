# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import os

# This guide can only be run with the TF backend.
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras

from openfl.federated import KerasTaskRunner

class CNNModel(keras.Model):
    """
    Custom Keras Model for a Convolutional Neural Network (CNN) with custom training and testing steps.
    This model showcase how to define a custom training and testing step for a Keras model with Tensorflow.
    Methods
    -------
    train_step(data)
        Performs a single training step, including forward pass, loss computation, gradient calculation, 
        and weight updates. Also updates the metrics.
    test_step(data)
        Performs a single testing step, including forward pass, loss computation, and metric updates.
    """

    def train_step(self, data):
        """
        Perform a single training step.
        Args:
            data (tuple): A tuple containing the input data and labels. If the tuple has three elements,
                          it should be (x, y, sample_weight). Otherwise, it should be (x, y).
        Returns:
            dict: A dictionary mapping metric names to their current values. This includes the loss and
                  any other metrics configured in `compile()`.
        Notes:
            - The loss function and metrics are configured in the `compile()` method.
            - The optimizer is used to apply the computed gradients to the model's trainable variables.
        """

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compute_loss(
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply(gradients, trainable_vars)

        # Update the metrics.
        # Metrics are configured in `compile()`.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        Perform a single test step.
        Args:
            data (tuple): A tuple containing the input data (x) and the true labels (y).
        Returns:
            dict: A dictionary mapping metric names to their current values. This includes the loss and other metrics tracked in self.metrics.
        """

        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss = self.compute_loss(y=y, y_pred=y_pred)
        # Update the metrics.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


class CNNTaskruner(KerasTaskRunner):
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
            models: The model defined in Keras

        """
        inputs = keras.Input(shape=input_shape)
        outputs = keras.layers.Conv2D(conv1_channels_out,
                                    kernel_size=conv_kernel_size,
                                    strides=conv_strides,
                                    activation='relu',
                                    input_shape=input_shape)(inputs)
        outputs = keras.layers.Conv2D(conv2_channels_out,
                         kernel_size=conv_kernel_size,
                         strides=conv_strides,
                         activation='relu')(outputs)

        outputs = keras.layers.Flatten()(outputs)

        outputs = keras.layers.Dense(final_dense_inputsize, activation='relu')(outputs)

        outputs = keras.layers.Dense(num_classes, activation='softmax')(outputs)

        model = CNNModel(inputs, outputs)

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        return model
