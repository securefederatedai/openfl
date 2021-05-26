# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import tensorflow as tf

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
        print(tf.config.threading.get_intra_op_parallelism_threads())
        print(tf.config.threading.get_inter_op_parallelism_threads())
        # physical_devices = tf.config.list_physical_devices('GPU')
        # try:
        #      tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # except:
        #      # Invalid device or cannot modify virtual devices once initialized.
        #        pass

        # ## Define Model
        #
        # Convolutional neural network model

        inputs = tf.keras.layers.Input(shape=input_shape)
        conv = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
        conv = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding="same", activation="relu")(conv)
        maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)

        conv = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation="relu")(maxpool)
        conv = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu")(conv)
        concat = tf.keras.layers.concatenate([maxpool, conv])
        maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(concat)

        conv = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding="same", activation="relu")(maxpool)
        conv = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv)
        concat = tf.keras.layers.concatenate([maxpool, conv])
        maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(concat)

        conv = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding="same", activation="relu")(maxpool)
        conv = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv)
        concat = tf.keras.layers.concatenate([maxpool, conv])
        maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(concat)

        flat = tf.keras.layers.Flatten()(maxpool)
        dense = tf.keras.layers.Dense(128)(flat)
        drop = tf.keras.layers.Dropout(0.5)(dense)

        predict = tf.keras.layers.Dense(num_classes)(drop)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[predict])

        self.optimizer = tf.keras.optimizers.Adam()

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=self.optimizer,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        self.tvars = model.layers
        print(f"layer names: {[var.name for var in self.tvars]}")

        self.opt_vars = self.optimizer.variables()
        print(f"optimizer vars: {self.opt_vars}")

        # Two opt_vars for one tvar: gradient and square sum for RMSprop.
        self.fl_vars = self.tvars + self.opt_vars

        return model
