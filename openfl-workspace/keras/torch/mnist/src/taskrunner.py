# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import logging

import keras

from openfl.federated import KerasTaskRunner
from src.model import CNNModel

logger = logging.getLogger(__name__)

class KerasCNN(KerasTaskRunner):
    """A basic convolutional neural network model."""

    def __init__(self, **kwargs):
        """
        Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        self.model = self.build_model(
            self.data_loader.get_feature_shape(),
            self.data_loader.get_num_classes(),
            **kwargs
        )

        self.initialize_tensorkeys_for_functions()

        self.model.summary(print_fn=logger.info)

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

        model = CNNModel(inputs, outputs)

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        return model
