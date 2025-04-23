# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import torch
import torchvision\



# Download Test datasets

mnist_test = torchvision.datasets.MNIST(
    "./files/",
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
portland_col_idx = 0
n_collaborators = 2
batch_size_test = 1000

local_test = deepcopy(mnist_test)

local_test.data = mnist_test.data[portland_col_idx::n_collaborators]
local_test.targets = mnist_test.targets[portland_col_idx::n_collaborators]


portland_attrs = {
   'test_loader': torch.utils.data.DataLoader(local_test,batch_size=batch_size_test, shuffle=True)
}
