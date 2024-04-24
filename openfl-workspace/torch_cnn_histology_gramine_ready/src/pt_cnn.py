# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import TensorKey


def cross_entropy(output, target):
    """Calculate Cross-entropy loss."""
    return F.cross_entropy(input=output, target=target)


class PyTorchCNN(PyTorchTaskRunner):
    """Simple CNN for classification."""

    def __init__(self, **kwargs):
        """Initialize.

        Args:
            **kwargs: Additional arguments to pass to the function
        """
        super().__init__(loss_fn=cross_entropy, **kwargs)

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.num_classes = self.data_loader.num_classes
        self.init_network(device=self.device, **kwargs)
        self._init_optimizer(lr=kwargs.get('lr'))
        self.initialize_tensorkeys_for_functions()

    def _init_optimizer(self, lr):
        """Initialize the optimizer."""
        self.optimizer = optim.Adam(self.parameters(), lr=float(lr or 1e-3))

    def init_network(self,
                     device,
                     print_model=True,
                     **kwargs):
        """Create the network (model).

        Args:
            device: The hardware device to use for training
            print_model (bool): Print the model topology (Default=True)
            **kwargs: Additional arguments to pass to the function

        """
        channel = self.data_loader.get_feature_shape()[
            0]  # (channel, dim1, dim2)
        conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        self.conv1 = nn.Conv2d(channel, 16, **conv_kwargs)
        self.conv2 = nn.Conv2d(16, 32, **conv_kwargs)
        self.conv3 = nn.Conv2d(32, 64, **conv_kwargs)
        self.conv4 = nn.Conv2d(64, 128, **conv_kwargs)
        self.conv5 = nn.Conv2d(128 + 32, 256, **conv_kwargs)
        self.conv6 = nn.Conv2d(256, 512, **conv_kwargs)
        self.conv7 = nn.Conv2d(512 + 128 + 32, 256, **conv_kwargs)
        self.conv8 = nn.Conv2d(256, 512, **conv_kwargs)
        self.fc1 = nn.Linear(1184 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 8)
        if print_model:
            print(self)
        self.to(device)

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

    def validate_task(self, col_name, round_num, input_tensor_dict,
                 use_tqdm=False, **kwargs):
        """Validate.

        Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress
                                 bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB

        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.eval()
        val_score = 0
        total_samples = 0

        loader = self.data_loader.get_valid_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc='validate')

        with torch.no_grad():
            for data, target in loader:
                samples = target.shape[0]
                total_samples += samples
                data, target = (torch.tensor(data).to(self.device),
                                torch.tensor(target).to(self.device))
                output = self(data)
                # get the index of the max log-probability
                pred = output.argmax(dim=1)
                val_score += pred.eq(target).sum().cpu().numpy()

        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)
        # TODO figure out a better way to pass in metric for
        #  this pytorch validate function
        output_tensor_dict = {
            TensorKey('acc', origin, round_num, True, tags):
                np.array(val_score / total_samples)
        }

        # empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def reset_opt_vars(self):
        """Reset optimizer variables.

        Resets the optimizer state variables.

        """
        self._init_optimizer(lr=self.optimizer.defaults.get('lr'))
