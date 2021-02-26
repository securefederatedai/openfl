# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import numpy as np
import tensorflow.keras as ke

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from openfl import CoreTaskRunner


def build_model(input_shape=(28,28,1),
                num_classes=10,
                conv_kernel_size=(4, 4),
                conv_strides=(2, 2),
                conv1_channels_out=16,
                conv2_channels_out=32,
                final_dense_inputsize=100):

    model = Sequential()

    model.add(Conv2D(conv1_channels_out,
                        kernel_size=conv_kernel_size,
                        strides=conv_strides,
                        activation='relu',
                        input_shape=input_shape))

    model.add(Conv2D(conv2_channels_out,
                        kernel_size=conv_kernel_size,
                        strides=conv_strides,
                        activation='relu'))

    model.add(Flatten())

    model.add(Dense(final_dense_inputsize, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=ke.losses.categorical_crossentropy,
                    optimizer=ke.optimizers.Adam(),
                    metrics=['accuracy'])

    # initialize the optimizer variables
    for v in model.optimizer.variables():
        v.initializer.run()

    return model

class ModelProvider:
    def __init__(self) -> None:
        self.model_inst = build_model()

    def provide_model(self):
        return self.model_inst

    def provide_optimizer(self):
        return self.model_inst.optimizer


model_provider = ModelProvider()


CTR = CoreTaskRunner(model_provider)

@CTR.register_fl_task(task_type='train', task_name='train')
def train_epoch(model, train_loader, device, optimizer):
    history = model.fit(train_loader.X_train,
                            train_loader.y_train,
                            batch_size=train_loader.batch_size,
                            epochs=1,
                            verbose=0, )
    
    output_metrics = {str(metric_name) : np.mean([history.history[metric_name]]) for metric_name in model.metrics_names}

    return output_metrics


@CTR.register_fl_task(task_type='validate', task_name='validate')
def validation(model, val_loader, device):
    vals = model.evaluate(
            val_loader.X_valid,
            val_loader.y_valid,
            batch_size=val_loader.batch_size,
            verbose=0
        )
    model_metrics_names = model.metrics_names
    if type(vals) is not list:
        vals = [vals]
    ret_dict = dict(zip(model_metrics_names, vals))

    return ret_dict