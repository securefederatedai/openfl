# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

from openfl.experimental.interface import FLSpec
from openfl.experimental.placement import aggregator, collaborator

learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

convolutional_block = nn.Sequential
sequential_block = nn.Sequential
conv2d1 = nn.Conv2d
conv2d2 = nn.Conv2d(10, 20, 5)
maxpool2d1 = nn.MaxPool2d
maxpool2d2 = nn.MaxPool2d(2)
relu = nn.ReLU()
dropout2d = nn.Dropout2d()


class Net(nn.Module):
    def __init__(self, convolutional_block,
                 in_features: int, out_features: int):
        super(Net, self).__init__()
        self.conv_block = convolutional_block
        self.linear_block = nn.Sequential(
            nn.Linear(320, in_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(-1, 320)
        x = self.linear_block(x)
        return F.log_softmax(x)


def inference(network, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: "
        + f"{correct}/{len(test_loader.dataset)} "
        + f"({100.0 * correct / len(test_loader.dataset):.0f}%)\n"
    )

    accuracy = float(correct / len(test_loader.dataset))
    return accuracy


def fedavg(models):
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():
        state_dict[key] = np.sum(
            np.array([state[key] for state in state_dicts], dtype=object), axis=0
        ) / len(models)
    new_model.load_state_dict(state_dict)
    return new_model


class MNISTFlow(FLSpec):
    def __init__(self, model=None, optimizer=None, rounds=3, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            self.optimizer = optimizer
        else:
            self.model = Net()
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=learning_rate, momentum=momentum
            )
        self.rounds = rounds

    @aggregator
    def start(self):
        print("Performing initialization for model")
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.current_round = 0
        self.next(
            self.aggregated_model_validation,
            foreach="collaborators",
            exclude=["private"],
        )

    @collaborator
    def aggregated_model_validation(self):
        print(f"Performing aggregated model validation for collaborator {self.input}")
        self.agg_validation_score = inference(self.model, self.test_loader)
        print(f"{self.input} value of {self.agg_validation_score}")
        self.next(self.train)

    @collaborator
    def train(self):
        self.model.train()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum
        )
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    f"Train Epoch: 1 [{batch_idx * len(data)}/"
                    + f"{len(self.train_loader.dataset)} ("
                    + f"{100.0 * batch_idx / len(self.train_loader):.0f}%)"
                    + f"]\tLoss: {loss.item():.6f}"
                )

                self.loss = loss.item()
                torch.save(self.model.state_dict(), "model.pth")
                torch.save(self.optimizer.state_dict(), "optimizer.pth")
        self.training_completed = True
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_validation_score = inference(self.model, self.test_loader)
        print(
            "Doing local model validation for collaborator "
            + f"{self.input}: {self.local_validation_score}"
        )
        self.next(self.join, exclude=["training_completed"])

    @aggregator
    def join(self, inputs):
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs
        ) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs
        ) / len(inputs)
        print(
            f"Average aggregated model validation values = {self.aggregated_model_accuracy}"
        )
        print(f"Average training loss = {self.average_loss}")
        print(f"Average local model validation values = {self.local_model_accuracy}")
        self.model = fedavg([input.model for input in inputs])
        self.optimizer = [input.optimizer for input in inputs][0]
        self.next(self.internal_loop)

    @aggregator
    def internal_loop(self):
        self.current_round += 1
        if self.current_round < self.rounds:
            self.next(
                self.aggregated_model_validation,
                foreach="collaborators",
                exclude=["private"],
            )
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print("This is the end of the flow")
