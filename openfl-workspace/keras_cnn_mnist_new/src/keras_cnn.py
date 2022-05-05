# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
from tensorflow import keras
from tensorflow.keras import layers

from openfl.federated import KerasTaskRunner


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
                    **kwargs):
        """
        Define the model architecture.

        Args:
            input_shape (numpy.ndarray): The shape of the data
            num_classes (int): The number of classes of the dataset

        Returns:
            tensorflow.python.keras.engine.sequential.Sequential: The model defined in Keras

        """

        inputs = keras.Input(shape=(784,), name='digits')
        x1 = layers.Dense(64, activation='relu')(inputs)
        x2 = layers.Dense(64, activation='relu')(x1)
        outputs = layers.Dense(num_classes, name='predictions')(x2)
        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'],)
        # initialize the optimizer variables
        opt_vars = model.optimizer.variables()

        for v in opt_vars:
            v.initializer.run(session=self.sess)

        return model
