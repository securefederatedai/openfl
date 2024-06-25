# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from torch import nn, optim
from torchvision import datasets, transforms
import torch

input_size = 784
hidden_sizes = [128, 640]
batch_size = 2048

data_model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
)

data_model_optimizer = optim.SGD(data_model.parameters(), lr=0.03)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
trainset = datasets.MNIST('mnist', download=True, train=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def collaborator_private_attrs(data_model, data_model_optimizer, train_loader):
    return {
        "data_model": data_model,
        "data_model_optimizer": data_model_optimizer,
        "trainloader": deepcopy(train_loader)
    }
