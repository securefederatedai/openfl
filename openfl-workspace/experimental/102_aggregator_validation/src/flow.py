# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from openfl.experimental.interface import FLSpec
from openfl.experimental.placement import aggregator, collaborator

import numpy as np


learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def fedavg(models, weights=None):
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():
        state_dict[key] = np.average(
            np.array([state[key] for state in state_dicts], dtype=object), axis=0, weights=weights)
    new_model.load_state_dict(state_dict)
    return new_model


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
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Avg. loss: {test_loss:.4f},'
          + f' Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return accuracy


class AggregatorValidationFlow(FLSpec):

    def __init__(self, model=None, optimizer=None, rounds=3, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            self.optimizer = optimizer
        else:
            self.model = Net()
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                       momentum=momentum)
        self.rounds = rounds

    @aggregator
    def start(self):
        print('Performing initialization for model')
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.current_round = 0
        self.next(self.aggregated_model_validation, foreach='collaborators', exclude=['private'])

    @collaborator
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input}')
        self.agg_validation_score = inference(self.model, self.test_loader)
        print(f'{self.input} value of {self.agg_validation_score}')
        self.next(self.train)

    @collaborator
    def train(self):
        self.model.train()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                   momentum=momentum)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                self.loss = loss.item()
                accuracy = 100. * batch_idx / len(self.train_loader)
                print(f'Train Epoch: 1 [{batch_idx * len(data)}/{len(self.train_loader.dataset)}'
                      + f' ({accuracy:.0f}%)]\tLoss: {self.loss:.6f}')
                torch.save(self.model.state_dict(), 'model.pth')
                torch.save(self.optimizer.state_dict(), 'optimizer.pth')
        self.training_completed = True
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_validation_score = inference(self.model, self.test_loader)
        print(f'Doing local model validation for collaborator {self.input}:'
              + f' {self.local_validation_score}')
        self.next(self.join, exclude=['training_completed'])

    @aggregator
    def join(self, inputs):
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs) / len(inputs)
        print(f'Average aggregated model validation values = {self.aggregated_model_accuracy}')
        print(f'Average training loss = {self.average_loss}')
        print(f'Average local model validation values = {self.local_model_accuracy}')

        highest_accuracy = 0
        highest_accuracy_model_idx = -1
        for idx, col in enumerate(inputs):
            accuracy_for_held_out_agg_data = inference(col.model, self.test_loader)
            if accuracy_for_held_out_agg_data > highest_accuracy:
                highest_accuracy = accuracy_for_held_out_agg_data
                highest_accuracy_model_idx = idx

        relative_model_weights = len(inputs) * [1]
        # Give highest accuracy model (on held out aggregator data) 2x the importance
        relative_model_weights[highest_accuracy_model_idx] = 2
        print(f'Aggregator validation score: {highest_accuracy}')
        print(f'Highest accuracy model sent from {inputs[highest_accuracy_model_idx].input}.'
              + ' Receiving 2x weight in updated model')
        self.model = fedavg([input.model for input in inputs], weights=relative_model_weights)
        self.optimizer = [input.optimizer for input in inputs][0]
        self.current_round += 1
        if self.current_round < self.rounds:
            self.next(self.aggregated_model_validation,
                      foreach='collaborators', exclude=['private'])
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print('This is the end of the flow')
