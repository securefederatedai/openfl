# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import time
import tensorflow.keras as ke
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications.resnet50 import ResNet50

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
            tensorflow.python.keras.engine.sequential.Sequential: The model defined in Keras

        """
        # model = Sequential()

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
        model = Sequential()
        model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet', input_shape=(32, 32, 3)))
        model.add(Dense(10, activation = 'softmax'))
        # Say not to train first layer (ResNet) model as it is already trained
        model.layers[0].trainable = False
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # model = ke.applications.ResNet50(include_top=False,
        #     weights="imagenet",
        #     input_shape=(32,32,3))
        # model.compile(loss=ke.losses.categorical_crossentropy,
        #               optimizer=ke.optimizers.Adam(),
        #               metrics=['accuracy'])

        # initialize the optimizer variables
        #opt_vars = model.optimizer.variables()

        #for v in opt_vars:
        #    v.initializer.run(session=self.sess)

        return model

    def train_iteration(self, batch_generator, metrics: list = None, **kwargs):
        """Train single epoch.

        Override this function for custom training.

        Args:
            batch_generator: Generator of training batches.
                Each batch is a tuple of N train images and N train labels
                where N is the batch size of the DataLoader of the current TaskRunner instance.

            epochs: Number of epochs to train.
            metrics: Names of metrics to save.
        """
        start = time.time()
        results = super().train_iteration(batch_generator=batch_generator, metrics=metrics)
        end = time.time()
        with open('times_keras_cnn_mnist.csv', 'a+') as f:
            f.write(f'{end-start}\n')
        f.close()
        return results
