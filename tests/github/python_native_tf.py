# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Python native tests."""

import numpy as np
import json

import openfl.native as fx
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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


def load_mnist():
    from tensorflow.keras.datasets.mnist import load_data
    return load_data()


fx.init('keras_cnn_mnist')

if __name__ == '__main__':
    from openfl.federated import FederatedModel, FederatedDataSet
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(load_mnist)
    (X_train, y_train), (X_valid, y_valid) = f.result()
    X_train = np.expand_dims(X_train, -1)
    X_valid = np.expand_dims(X_valid, -1)
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
        from tensorflow import keras
        from tensorflow.keras import layers, losses, optimizers
        if tf.compat.v1.get_default_graph():
            keras.backend.clear_session()
        print('Building model...')
        
        s = tf.compat.v1.keras.backend.get_session()
        with s.graph.as_default():
            model = keras.Sequential(layers=[
                layers.Conv2D(conv1_channels_out,
                                    kernel_size=conv_kernel_size,
                                    strides=conv_strides,
                                    activation='relu',
                                    input_shape=input_shape),
                layers.Conv2D(conv2_channels_out,
                                    kernel_size=conv_kernel_size,
                                    strides=conv_strides,
                                    activation='relu'),
                layers.Flatten(),
                layers.Dense(final_dense_inputsize, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
            print('Created Sequential')
            print('Built model. Compiling...')
            model.compile(loss=losses.categorical_crossentropy,
                        optimizer=optimizers.Adam(),
                        metrics=['accuracy'])
            print('Compiled. Initializing variables...')
            print(f'[{id(s.graph)}] Global variables: {tf.compat.v1.global_variables()}')
            s.run(tf.compat.v1.global_variables_initializer())
            print('Initialized!')
        opt_vars = model.optimizer.variables()
        for v in opt_vars:
            v.initializer.run(session=s)
        return model, s

    fl_model = FederatedModel(build_model, data_loader=fl_data)
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

    print(json.dumps(fx.get_plan(), indent=4, sort_keys=True))
    final_fl_model = fx.run_experiment(collaborators, {'aggregator.settings.rounds_to_train': 5})
    final_fl_model.save_native('final_pytorch_model.h5')
