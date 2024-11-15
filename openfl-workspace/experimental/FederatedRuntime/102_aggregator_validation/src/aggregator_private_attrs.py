# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import torch
import torchvision


mnist_test = torchvision.datasets.MNIST('files/', train=False, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                        ]))

test_dataset = mnist_test


def callable_to_initialize_aggregator_private_attributes(n_collaborators,
                                                         test_dataset, batch_size):
    aggregator_test = deepcopy(test_dataset)
    aggregator_test.targets = test_dataset.targets[n_collaborators::n_collaborators + 1]
    aggregator_test.data = test_dataset.data[n_collaborators::n_collaborators + 1]

    return {
        'test_loader': torch.utils.data.DataLoader(
            aggregator_test, batch_size=batch_size, shuffle=True)
    }
