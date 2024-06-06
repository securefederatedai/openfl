# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import numpy as np
import tensorflow as tf

from openfl.utilities import Metric
from openfl.federated import TensorFlowTaskRunner


class TensorFlowCNN(TensorFlowTaskRunner):
    """Initialize.

    Args:
        **kwargs: Additional parameters to pass to the function

    """

    def __init__(self, **kwargs):
        """
        Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function

        """
        super().__init__(**kwargs)

        self.model = self.build_model(
            self.feature_shape,
            self.data_loader.num_classes,
            **kwargs
        )
        self.initialize_tensorkeys_for_functions()

        self.model.summary(print_fn=self.logger.info)

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
            **kwargs: Additional parameters to pass to the function

        Returns:
            keras.src.engine.functional.Functional

        """
        print(tf.config.threading.get_intra_op_parallelism_threads())
        print(tf.config.threading.get_inter_op_parallelism_threads())

        ## Define Model using Functional API

        inputs = tf.keras.layers.Input(shape=input_shape)
        conv = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
        conv = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv)
        maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)

        conv = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding='same', activation='relu')(maxpool)
        conv = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv)
        concat = tf.keras.layers.concatenate([maxpool, conv])
        maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(concat)

        conv = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding='same', activation='relu')(maxpool)
        conv = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding='same', activation='relu')(conv)
        concat = tf.keras.layers.concatenate([maxpool, conv])
        maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(concat)

        conv = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding='same', activation='relu')(maxpool)
        conv = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding='same', activation='relu')(conv)
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

        return model


    def train_(self, batch_generator, metrics: list = None, **kwargs):
        """Train single epoch.

        Override this function for custom training.

        Args:
            batch_generator: Generator of training batches.
                Each batch is a tuple of N train images and N train labels
                where N is the batch size of the DataLoader of the current TaskRunner instance.

            epochs: Number of epochs to train.
            metrics: Names of metrics to save.
        """
        if metrics is None:
            metrics = []

        model_metrics_names = self.model.metrics_names

        for param in metrics:
            if param not in model_metrics_names:
                raise ValueError(
                    f'TensorFlowTaskRunner does not support specifying new metrics. '
                    f'Param_metrics = {metrics}, model_metrics_names = {model_metrics_names}'
                )

        history = self.model.fit(batch_generator,
                                 verbose=1,
                                 **kwargs)
        results = []
        for metric in metrics:
            value = np.mean([history.history[metric]])
            results.append(Metric(name=metric, value=np.array(value)))
        return results
