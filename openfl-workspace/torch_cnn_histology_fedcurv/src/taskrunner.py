# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Iterator, Tuple
from openfl.utilities.fedcurv.torch.fedcurv import FedCurv

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import Metric


class PyTorchCNNWithFedCurv(PyTorchTaskRunner):
    """
    Simple CNN for classification.

    PyTorchTaskRunner inherits from nn.module, so you can define your model
    in the same way that you would for PyTorch
    """

    def __init__(self, device="cpu", **kwargs):
        """Initialize.

        Args:
            device: The hardware device to use for training (Default = "cpu")
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(device=device, **kwargs)

        # Define the model
        channel = self.data_loader.get_feature_shape()[0]  # (channel, dim1, dim2)
        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128 + 32, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512 + 128 + 32, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1184 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 8)

        # `self.optimizer` must be set for optimizer weights to be federated
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self._fedcurv = FedCurv(self, importance=1e7)

        # Set the loss function
        self.loss_fn = F.cross_entropy

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Data input to the model for the forward pass
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        maxpool = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(maxpool))
        x = F.relu(self.conv4(x))
        concat = torch.cat([maxpool, x], dim=1)
        maxpool = F.max_pool2d(concat, 2, 2)

        x = F.relu(self.conv5(maxpool))
        x = F.relu(self.conv6(x))
        concat = torch.cat([maxpool, x], dim=1)
        maxpool = F.max_pool2d(concat, 2, 2)

        x = F.relu(self.conv7(maxpool))
        x = F.relu(self.conv8(x))
        concat = torch.cat([maxpool, x], dim=1)
        maxpool = F.max_pool2d(concat, 2, 2)

        x = maxpool.flatten(start_dim=1)
        x = F.dropout(self.fc1(x), p=0.5)
        x = self.fc2(x)
        return x

    def train_(
        self, train_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]
    ) -> Metric:
        """Train single epoch.

        Override this function in order to use custom training.

        Args:
            batch_generator: Train dataset batch generator. Yields (samples, targets) tuples of
            size = `self.data_loader.batch_size`.
        Returns:
            Metric: An object containing name and np.ndarray value.
        """
        losses = []
        model = self
        self._fedcurv.on_train_begin(model)

        for data, target in train_dataloader:
            data, target = torch.tensor(data).to(self.device), torch.tensor(target).to(
                self.device
            )
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.loss_fn(output, target) + self._fedcurv.get_penalty(model)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().numpy())
            break
        loss = np.mean(losses)
        return Metric(name=self.loss_fn.__name__, value=np.array(loss))

    def validate_(
        self, validation_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]
    ) -> Metric:
        """
        Perform validation on PyTorch Model

        Override this function for your own custom validation function

        Args:
            validation_data_loader: Validation dataset batch generator.
                                    Yields (samples, targets) tuples.
        Returns:
            Metric: An object containing name and np.ndarray value
        """

        total_samples = 0
        val_score = 0
        with torch.no_grad():
            for data, target in validation_dataloader:
                samples = target.shape[0]
                total_samples += samples
                data, target = torch.tensor(data).to(self.device), torch.tensor(
                    target
                ).to(self.device, dtype=torch.int64)
                output = self(data)
                # get the index of the max log-probability
                pred = output.argmax(dim=1)
                val_score += pred.eq(target).sum().cpu().numpy()

        accuracy = val_score / total_samples
        return Metric(name="accuracy", value=np.array(accuracy))
