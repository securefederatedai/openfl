# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import keras

from openfl.federated import KerasTaskRunner


class TensorFlowCNN(KerasTaskRunner):
    """Initialize.

    Args:
        **kwargs: Additional parameters to pass to the function

    """

    def __init__(self, **kwargs):
        """Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function

        """
        super().__init__(**kwargs)

        self.model = self.create_model(
            self.feature_shape,
            self.data_loader.num_classes,
            **kwargs
        )
        self.initialize_tensorkeys_for_functions()

    def create_model(self,
                     input_shape,
                     num_classes,
                     training_smoothing=32.0,
                     validation_smoothing=1.0,
                     **kwargs):
        """Create the TensorFlow CNN Histology model.

        Args:
            training_smoothing (float): (Default=32.0)
            validation_smoothing (float): (Default=1.0)
            **kwargs: Additional parameters to pass to the function

        """
        # ## Define Model
        #
        # Convolutional neural network model

        inputs = keras.layers.Input(shape=input_shape)
        conv = keras.layers.Conv2D(
            filters=16, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
        conv = keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv)
        maxpool = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)

        conv = keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding='same', activation='relu')(maxpool)
        conv = keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv)
        concat = keras.layers.concatenate([maxpool, conv])
        maxpool = keras.layers.MaxPooling2D(pool_size=(2, 2))(concat)

        conv = keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding='same', activation='relu')(maxpool)
        conv = keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding='same', activation='relu')(conv)
        concat = keras.layers.concatenate([maxpool, conv])
        maxpool = keras.layers.MaxPooling2D(pool_size=(2, 2))(concat)

        conv = keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding='same', activation='relu')(maxpool)
        conv = keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding='same', activation='relu')(conv)
        concat = keras.layers.concatenate([maxpool, conv])
        maxpool = keras.layers.MaxPooling2D(pool_size=(2, 2))(concat)

        flat = keras.layers.Flatten()(maxpool)
        dense = keras.layers.Dense(128)(flat)
        drop = keras.layers.Dropout(0.5)(dense)

        predict = keras.layers.Dense(num_classes)(drop)

        model = keras.models.Model(inputs=[inputs], outputs=[predict])

        self.optimizer = keras.optimizers.Adam()

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=self.optimizer,
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        self.tvars = model.layers
        print(f'layer names: {[var.name for var in self.tvars]}')

        self.opt_vars = self.optimizer.variables
        print(f'optimizer vars: {self.opt_vars}')

        return model
