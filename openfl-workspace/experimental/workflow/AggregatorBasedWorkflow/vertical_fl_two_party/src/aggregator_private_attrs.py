# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import datasets, transforms
from torch import nn, optim


hidden_sizes = [128, 640]
output_size = 10
batch_size = 2048

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
trainset = datasets.MNIST('mnist', download=True, train=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

label_model = nn.Sequential(
    nn.Linear(hidden_sizes[1], output_size),
    nn.LogSoftmax(dim=1)
)

label_model_optimizer = optim.SGD(label_model.parameters(), lr=0.03)


def aggregator_private_attrs(train_loader, label_model, label_model_optimizer):
    return {
        "trainloader": train_loader,
        "label_model": label_model,
        "label_model_optimizer": label_model_optimizer
    }
