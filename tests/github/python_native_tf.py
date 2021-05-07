# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Python native tests."""

import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

import openfl.native as fx
from openfl.federated import FederatedModel, FederatedDataSet
from openfl.interface.cli import setup_logging


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
    import tensorflow as tf
    import tensorflow.keras as ke

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, Dense
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


setup_logging()


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Test FX native API with Torch')
    parser.add_argument('--batch_size', metavar='B', type=int, nargs='?', help='batch_size',
                        default=32)
    parser.add_argument('--dataset_multiplier', metavar='M', type=int, nargs='?',
                        help='dataset_multiplier', default=1)
    parser.add_argument('--rounds_to_train', metavar='R', type=int, nargs='?',
                        help='rounds_to_train', default=5)
    parser.add_argument('--collaborators_amount', metavar='C', type=int, nargs='?',
                        help='collaborators_amount', default=2)
    parser.add_argument('--is_multi', const=True, nargs='?',
                        help='is_multi', default=False)
    parser.add_argument('--max_workers', metavar='W', type=int, nargs='?',
                        help='max_workers', default=0)
    parser.add_argument('--mode', metavar='W', type=str, nargs='?',
                        help='mode', default='p=c*r')
    parsed_args = parser.parse_args()
    print(parsed_args)
    return parsed_args


if __name__ == '__main__':

    args = _parse_args()

    fx.init('keras_cnn_mnist')
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

    X_train = np.concatenate([X_train for _ in range(args.dataset_multiplier)])
    y_train = np.concatenate([y_train for _ in range(args.dataset_multiplier)])

    feature_shape = X_train.shape[1]

    fl_data = FederatedDataSet(X_train, y_train, X_valid, y_valid,
                               batch_size=32, num_classes=classes)
    fl_model = FederatedModel(build_model=build_model, data_loader=fl_data)
    collaborator_models = fl_model.setup(num_collaborators=args.collaborators_amount)
    collaborators = {str(i): c for i, c in enumerate(collaborator_models)}

    print(f'Original training data size: {len(X_train)}')
    print(f'Original validation data size: {len(y_train)}\n')

    final_fl_model = fx.run_experiment(collaborators, {
        'aggregator.settings.rounds_to_train': args.rounds_to_train,
    }, is_multi=args.is_multi, max_workers=args.max_workers, mode=args.mode)
    final_fl_model.save_native('final_pytorch_model.h5')
