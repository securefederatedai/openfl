# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import keras

from .model import CNNModel
from openfl.federated import KerasTaskRunner

import logging
logger = logging.getLogger(__name__)

class CNNTaskruner(KerasTaskRunner):
    """A basic convolutional neural network model."""

    def __init__(self, **kwargs):
        """
        Initializes the TaskRunner instance. Builds the Keras model, initializes required tensors for all publicly accessible methods that
        could be called as part of a task and initializes the logger.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the superclass and used for model building.

        Attributes:
            model (keras.Model): The Keras model built using the provided feature shape and number of classes.
            logger (logging.Logger): Logger instance for logging information.

        Methods:
            build_model: Constructs the Keras model.
            initialize_tensorkeys_for_functions: Initializes tensor keys for various functions.
            get_train_data_size: Returns the size of the training dataset.
            get_valid_data_size: Returns the size of the validation dataset.
        """
        super().__init__(**kwargs)

        self.model = self.build_model(self.feature_shape, self.data_loader.num_classes, **kwargs)

        self.initialize_tensorkeys_for_functions()

        self.model.summary(print_fn=logger.info)

        logger.info(f'Train Set Size : {self.get_train_data_size()}')
        logger.info(f'Valid Set Size : {self.get_valid_data_size()}')

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
        Builds and compiles a Convolutional Neural Network (CNN) model.

        Args:
            input_shape (tuple): Shape of the input data (height, width, channels).
            num_classes (int): Number of output classes.
            conv_kernel_size (tuple, optional): Size of the convolutional kernels. Defaults to (4, 4).
            conv_strides (tuple, optional): Strides of the convolutional layers. Defaults to (2, 2).
            conv1_channels_out (int, optional): Number of output channels for the first convolutional layer. Defaults to 16.
            conv2_channels_out (int, optional): Number of output channels for the second convolutional layer. Defaults to 32.
            final_dense_inputsize (int, optional): Number of units in the final dense layer before the output layer. Defaults to 100.
            **kwargs: Additional keyword arguments.
        Returns:
            keras.Model: Compiled CNN model.
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
