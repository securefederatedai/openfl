# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import torch
import keras

class CNNModel(keras.Model):
    """
    A custom Keras model for a Convolutional Neural Network (CNN) that overrides
    the `train_step` and `test_step` methods to integrate with PyTorch's gradient
    computation and optimization.
    Methods
    -------
    train_step(data)
        Performs a single training step, including forward pass, loss computation,
        backward pass, and weight updates.
    test_step(data)
        Performs a single evaluation step, including forward pass, loss computation,
        and metric updates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.loss_fn = keras.losses.MeanSquaredError()

    def train_step(self, data):
        """
        Perform a single training step using torch.

        Args:
            data (tuple): A tuple containing the input data and labels. If the tuple has three elements,
                          the third element is considered as sample weights.

        Returns:
            dict: A dictionary mapping metric names to their current values, including the loss.

        The method performs the following steps:
        1. Unpacks the input data.
        2. Clears the gradients from the previous training step.
        3. Performs a forward pass to compute the predictions.
        4. Computes the loss based on the predictions and true labels.
        5. Computes the gradients by performing a backward pass on the loss.
        6. Updates the model weights using the computed gradients.
        7. Updates the metrics, including the loss.
        8. Returns the current values of the metrics.
        """

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        # Compute loss
        y_pred = self(x, training=True)  # Forward pass
        loss = self.compute_loss(
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        Perform a single test step using torch.
        Args:
            data (tuple): A tuple containing the input data (x) and the true labels (y).
        Returns:
            dict: A dictionary mapping metric names to their current values, including the loss.
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
