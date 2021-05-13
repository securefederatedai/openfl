# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Python native tests."""

import numpy as np

import openfl.native as fx

import tensorflow as tf
import tensorflow.keras as ke

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense


def one_hot(labels, classes):
    """
    One Hot encode a vector.

    Args:
        labels (list):  List of labels to onehot encode
        classes (int): Total number of categorical classes

    Returns:
        np.array: Matrix of one-hot encoded labels
    """
    return np.eye(classes)[labels]


def build_model(input_shape,
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
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 112
    config.inter_op_parallelism_threads = 1
    sess = tf.compat.v1.Session(config=config)
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
    opt_vars = model.optimizer.variables()

    for v in opt_vars:
        v.initializer.run(session=sess)

    return model


fx.init('keras_cnn_mnist')

if __name__ == '__main__':
    from openfl.federated import FederatedModel, FederatedDataSet
    from tensorflow.python.keras.utils.data_utils import get_file

    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    path = get_file('mnist.npz',
                    origin=origin_folder + 'mnist.npz',
                    file_hash='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')

    with np.load(path) as f:
        # get all of mnist
        X_train = f['x_train']
        y_train = f['y_train']

        X_valid = f['x_test']
        y_valid = f['y_test']
    img_rows, img_cols = 28, 28
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_valid /= 255

    classes = 10
    y_train = one_hot(y_train, classes)
    y_valid = one_hot(y_valid, classes)

    feature_shape = X_train.shape[1]

    fl_data = FederatedDataSet(X_train, y_train, X_valid, y_valid,
                               batch_size=32, num_classes=classes)
    fl_model = FederatedModel(build_model=build_model, data_loader=fl_data)
    collaborator_models = fl_model.setup(num_collaborators=2)
    collaborators = {'one': collaborator_models[0], 'two': collaborator_models[1]}
    print(f'Original training data size: {len(X_train)}')
    print(f'Original validation data size: {len(X_valid)}\n')

    # Collaborator one's data
    print(f'Collaborator one\'s training data size: \
            {len(collaborator_models[0].data_loader.X_train)}')
    print(f'Collaborator one\'s validation data size: \
            {len(collaborator_models[0].data_loader.X_valid)}\n')

    # Collaborator two's data
    print(f'Collaborator two\'s training data size: \
            {len(collaborator_models[1].data_loader.X_train)}')
    print(f'Collaborator two\'s validation data size: \
            {len(collaborator_models[1].data_loader.X_valid)}\n')

    print(fx.get_plan())
    final_fl_model = fx.run_experiment(collaborators, {'aggregator.settings.rounds_to_train': 5})
    final_fl_model.save_native('final_pytorch_model.h5')
