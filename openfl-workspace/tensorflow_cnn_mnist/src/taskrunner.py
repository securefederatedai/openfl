# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import os

# This guide can only be run with the TF backend.
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras

from openfl.federated import KerasTaskRunner

class CustomModel(keras.Model):

    def train_step(self, data):
        print("#"*50)
        print("train_step")
        print("#"*50)
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
        print("#"*50)
        print("test_step")
        print("#"*50)
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



# # Construct an instance of CustomModel
# inputs = keras.Input(shape=(32,))
# outputs = keras.layers.Dense(1)(inputs)
# model = CustomModel(inputs, outputs)

# # We don't pass a loss or metrics here.
# model.compile(optimizer="adam")

# # Just use `fit` as usual -- you can use callbacks, etc.
# x = np.random.random((1000, 32))
# y = np.random.random((1000, 1))
# model.fit(x, y, epochs=5)


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
            keras.models.Sequential: The model defined in Keras

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

        model = CustomModel(inputs, outputs)

        # model.add(Conv2D(conv1_channels_out,
        #                  kernel_size=conv_kernel_size,
        #                  strides=conv_strides,
        #                  activation='relu',
        #                  input_shape=input_shape))

        # model.add(Conv2D(conv2_channels_out,
        #                  kernel_size=conv_kernel_size,
        #                  strides=conv_strides,
        #                  activation='relu'))

        # model.add(Flatten())

        # model.add(Dense(final_dense_inputsize, activation='relu'))

        # model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        return model
