# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import torch
from copy import deepcopy
import torchvision

train_dataset = torchvision.datasets.MNIST(
    "files/",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

test_dataset = torchvision.datasets.MNIST(
    "files/",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)


def collaborator_private_attrs(index, n_collaborators, train_dataset,
                               test_dataset, batch_size_train):
    local_train = deepcopy(train_dataset)
    local_test = deepcopy(test_dataset)
    local_train.data = train_dataset.data[index:: n_collaborators]
    local_train.targets = train_dataset.targets[index:: n_collaborators]
    local_test.data = test_dataset.data[index:: n_collaborators]
    local_test.targets = test_dataset.targets[index:: n_collaborators]
    return {
        "train_loader": torch.utils.data.DataLoader(
            local_train, batch_size=batch_size_train, shuffle=True
        ),
        "test_loader": torch.utils.data.DataLoader(
            local_test, batch_size=batch_size_train, shuffle=True
        ),
    }
