# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import random

from Crypto.PublicKey import ECC
import torch
import torchvision


# Download Train and Test datasets
mnist_train = torchvision.datasets.MNIST(
    "../files/",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_test = torchvision.datasets.MNIST(
    "../files/",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

# shard the dataset according to collaborator index
portland_col_idx = 1
n_collaborators = 2
batch_size = 32

train = deepcopy(mnist_train)
test = deepcopy(mnist_test)

train.data = mnist_train.data[portland_col_idx::n_collaborators]
train.targets = mnist_train.targets[portland_col_idx::n_collaborators]
test.data = mnist_test.data[portland_col_idx::n_collaborators]
test.targets = mnist_test.targets[portland_col_idx::n_collaborators]

portland_attrs = {
    "train_loader": torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=False,
    ),
    "test_loader": torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
    ),
    "private_seed": random.random(),
    "private_key": [
        ECC.generate(curve="ed25519").export_key(format="PEM"),
        ECC.generate(curve="ed25519").export_key(format="PEM"),
    ],
}
