# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitRecognizerCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) model for digit recognition.

    This model consists of two convolutional layers followed by two fully connected layers.
    It uses ReLU activations and max pooling.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.

    Methods:
        forward(x):
            Defines the forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, 10).
    """

    def __init__(self, **kwargs):
        """
        Initializes the DigitRecognizerCNN model.

        Args:
            **kwargs: Additional keyword arguments to pass to the parent class initializer.

        Attributes:
            conv1 (nn.Conv2d): First convolutional layer with 1 input channel, 20 output channels,
                               kernel size of 2, and stride of 1.
            conv2 (nn.Conv2d): Second convolutional layer with 20 input channels, 50 output channels,
                               kernel size of 5, and stride of 1.
            fc1 (nn.Linear): First fully connected layer with 800 input features and 500 output features.
            fc2 (nn.Linear): Second fully connected layer with 500 input features and 10 output features.
        """
        super(DigitRecognizerCNN, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 20, 2, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        """
        Defines the forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where
                              N is the batch size,
                              C is the number of channels,
                              H is the height, and
                              W is the width.

        Returns:
            torch.Tensor: Output tensor after passing through the CNN layers.
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def train(model, optimizer, loss_fn, dataloader, device="cpu", epochs=1):
    """
    Trains the given model using the specified optimizer and loss function over the provided dataloader.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        loss_fn (callable): The loss function used to compute the loss.
        dataloader (torch.utils.data.DataLoader): The dataloader providing the training data.
        device (str, optional): The device to run the training on ("cpu" or "cuda"). Defaults to "cpu".
        epochs (int, optional): The number of epochs to train the model. Defaults to 1.

    Returns:
        float: The average loss of the last epoch.
    """
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        average_loss = train_epoch(model, optimizer, loss_fn, dataloader, device)
        print(f"Completed epoch {epoch + 1}/{epochs} with average loss: {average_loss}")

    return average_loss

def validate(model, test_dataloader, device="cpu"):
    """
    Validate the given model using the provided test data loader.

    Args:
        model (torch.nn.Module): The model to be validated.
        test_dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (str, optional): The device to run the validation on (default is "cpu").

    Returns:
        float: The accuracy of the model on the test dataset.
    """
    total_correct = 0
    total_samples = 0

    for data, target in test_dataloader:
        data = torch.as_tensor(data).to(device)
        target = torch.as_tensor(target).to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        total_correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += len(target)

    return total_correct / total_samples

def train_epoch(model, optimizer, loss_fn, dataloader, device="cpu"):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model parameters.
        loss_fn (callable): The loss function used to compute the loss.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the training data.
        device (str, optional): The device to run the training on ("cpu" or "cuda"). Defaults to "cpu".

    Returns:
        float: The average loss over all batches in the epoch.
    """
    total_loss = 0
    num_batches = 0
    for data, target in dataloader:
        data = torch.as_tensor(data).to(device)
        target = torch.as_tensor(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().item()
        num_batches += 1
    average_loss = total_loss / num_batches

    return average_loss
