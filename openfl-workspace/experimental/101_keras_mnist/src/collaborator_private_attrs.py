# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from keras.datasets import mnist
from keras.utils import np_utils


nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255.0
X_test /= 255.0
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

train_dataset = (X_train, Y_train)
test_dataset = (X_test, Y_test)

def collaborator_private_attrs(n_collaborators, index, train_dataset, test_dataset, batch_size):
    from openfl.utilities.data_splitters import EqualNumPyDataSplitter
    train_splitter = EqualNumPyDataSplitter()
    test_splitter = EqualNumPyDataSplitter()

    X_train, y_train = train_dataset
    X_test, y_test = test_dataset

    train_idx = train_splitter.split(y_train, n_collaborators)
    valid_idx = test_splitter.split(y_test, n_collaborators)

    train_dataset = X_train[train_idx[index]], y_train[train_idx[index]]
    test_dataset = X_test[valid_idx[index]], y_test[valid_idx[index]]

    return {
        "train_loader": train_dataset, "test_loader": test_dataset,
        "batch_size": batch_size
    }