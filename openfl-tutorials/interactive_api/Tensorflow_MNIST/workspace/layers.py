# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Useful keras functions."""
from tensorflow import keras
from tensorflow.keras import layers


def create_model():
    """Create keras model."""
    inputs = keras.Input(shape=(784,), name='digits')
    x1 = layers.Dense(64, activation='relu')(inputs)
    x2 = layers.Dense(64, activation='relu')(x1)
    outputs = layers.Dense(10, name='predictions')(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Instantiate an optimizer.
# Instantiate a loss function.


optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
