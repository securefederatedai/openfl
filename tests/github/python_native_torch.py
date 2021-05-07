# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Python native tests."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import openfl.native as fx
from openfl.federated import FederatedModel, FederatedDataSet
from openfl.interface.cli import setup_logging


def one_hot(labels, classes):
    """One-hot encode `labels` using `classes` classes."""
    return np.eye(classes)[labels]


def cross_entropy(output, target):
    """Binary cross-entropy metric."""
    return F.cross_entropy(input=output, target=target)


def get_optimizer(x):
    """Optimizer function."""
    return optim.Adam(x, lr=1e-4)


class Net(nn.Module):
    """PyTorch Neural Network."""

    def __init__(self):
        """Initialize."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass of the network."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Test FX native API with Torch')
    parser.add_argument('--batch_size', metavar='B', type=int, nargs='?', help='batch_size',
                        default=32)
    parser.add_argument('--dataset_multiplier', metavar='M', type=int, nargs='?',
                        help='dataset_multiplier', default=1)
    parser.add_argument('--rounds_to_train', metavar='R', type=int, nargs='?',
                        help='rounds_to_train', default=5)
    parser.add_argument('--collaborators_amount', metavar='C', type=int, nargs='?',
                        help='collaborators_amount', default=2)
    parser.add_argument('--is_multi', const=True, nargs='?',
                        help='is_multi', default=False)
    parser.add_argument('--max_workers', metavar='W', type=int, nargs='?',
                        help='max_workers', default=0)
    parser.add_argument('--mode', metavar='W', type=str, nargs='?',
                        help='mode', default='p=c*r')
    parsed_args = parser.parse_args()
    print(parsed_args)
    return parsed_args


setup_logging()

if __name__ == '__main__':
    args = _parse_args()
    fx.init('torch_cnn_mnist')

    classes = 10
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])

    trainset = datasets.MNIST(root='./data',
                              train=True,
                              download=True,
                              transform=transform)

    validset = datasets.MNIST(root='./data',
                              train=False,
                              download=True,
                              transform=transform,
                              target_transform=lambda labels: one_hot(labels, classes))

    (train_images, train_labels), (valid_images, valid_labels) = (zip(*dataset) for dataset in
                                                                  [trainset, validset])
    train_images, valid_images = (torch.stack(images).numpy() for images in
                                  [train_images, valid_images])
    train_labels, valid_labels = (np.stack(labels) for labels in [train_labels, valid_labels])
    feature_shape = train_images.shape[1]
    train_images = np.concatenate([train_images for _ in range(args.dataset_multiplier)])
    train_labels = np.concatenate([train_labels for _ in range(args.dataset_multiplier)])

    fl_data = FederatedDataSet(train_images, train_labels, valid_images, valid_labels,
                               batch_size=args.batch_size, num_classes=classes)
    fl_model = FederatedModel(build_model=Net, optimizer=get_optimizer,
                              loss_fn=cross_entropy, data_loader=fl_data)
    collaborator_models = fl_model.setup(num_collaborators=args.collaborators_amount)
    collaborators = {str(i): c for i, c in enumerate(collaborator_models)}

    print(f'Original training data size: {len(train_images)}')
    print(f'Original validation data size: {len(valid_images)}\n')

    final_fl_model = fx.run_experiment(collaborators, {
        'aggregator.settings.rounds_to_train': args.rounds_to_train,
    }, is_multi=args.is_multi, max_workers=args.max_workers, mode=args.mode)
    final_fl_model.save_native('final_pytorch_model')
    print('FINISH')
