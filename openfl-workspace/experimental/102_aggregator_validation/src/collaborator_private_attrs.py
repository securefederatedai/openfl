# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import torch
import torchvision


mnist_train = torchvision.datasets.MNIST('files/', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))

mnist_test = torchvision.datasets.MNIST('files/', train=False, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                        ]))

train_dataset = mnist_train
test_dataset = mnist_test


# Setup collaborators private attributes via callable function
def callable_to_initialize_collaborator_private_attributes(
        index, n_collaborators, train_dataset, test_dataset, batch_size):
    local_train = deepcopy(train_dataset)
    local_test = deepcopy(test_dataset)
    local_train.data = train_dataset.data[index::n_collaborators]
    local_train.targets = train_dataset.targets[index::n_collaborators]
    local_test.data = test_dataset.data[index::n_collaborators]
    local_test.targets = test_dataset.targets[index::n_collaborators]

    return {
        'train_loader': torch.utils.data.DataLoader(
            local_train, batch_size=batch_size, shuffle=True),
        'test_loader': torch.utils.data.DataLoader(
            local_test, batch_size=batch_size, shuffle=True)
    }
