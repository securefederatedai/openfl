# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import TensorKey


class PyTorchCNN(PyTorchTaskRunner):
    """Simple CNN for classification."""

    def __init__(self, device='cpu', **kwargs):
        """Initialize.

        Args:
            data: The data loader class
            device: The hardware device to use for training (Default = "cpu")
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(device=device, **kwargs)

        self.num_classes = self.data_loader.num_classes
        self.init_network(device=self.device, **kwargs)
        self.initialize_tensorkeys_for_functions()

    def init_network(self,
                     device,
                     print_model=True,
                     pool_sqrkernel_size=2,
                     conv_sqrkernel_size=5,
                     conv1_channels_out=20,
                     conv2_channels_out=50,
                     fc2_insize=500,
                     **kwargs):
        """Create the network (model).

        Args:
            device: The hardware device to use for training
            print_model (bool): Print the model topology (Default=True)
            pool_sqrkernel_size (int): Max pooling kernel size (Default=2),
                                       assumes square 2x2
            conv_sqrkernel_size (int): Convolutional filter size (Default=5),
                                       assumes square 5x5
            conv1_channels_out (int): Number of filters in first
                                      convolutional layer (Default=20)
            conv2_channels_out: Number of filters in second convolutional
                                layer (Default=50)
            fc2_insize (int): Number of neurons in the
                              fully-connected layer (Default = 500)
            **kwargs: Additional arguments to pass to the function

        FIXME: We are tracking only side lengths (rather than
        length and width) as we are assuming square
        shapes for feature and kernels.
        In order that all of the input and activation components are
        used (not cut off), we rely on a criterion: appropriate integers
        are divisible so that all casting to int performed below does no
        rounding (i.e. all int casting simply converts a float with '0'
        in the decimal part to an int.)

        (Note this criterion held for the original input sizes considered
        for this model: 28x28 and 32x32 when used with the default values
        above)
        """
        self.pool_sqrkernel_size = pool_sqrkernel_size
        channel = self.data_loader.get_feature_shape()[0]  # (channel, dim1, dim2)
        self.conv1 = nn.Conv2d(channel, conv1_channels_out, conv_sqrkernel_size, 1)

        # perform some calculations to track the size of the single channel activations
        # channels are first for pytorch
        conv1_sqrsize_in = self.feature_shape[-1]
        conv1_sqrsize_out = conv1_sqrsize_in - (conv_sqrkernel_size - 1)
        # a pool operation happens after conv1 out
        # (note dependence on 'forward' function below)
        conv2_sqrsize_in = int(conv1_sqrsize_out / pool_sqrkernel_size)

        self.conv2 = nn.Conv2d(conv1_channels_out, conv2_channels_out, conv_sqrkernel_size, 1)

        # more tracking of single channel activation size
        conv2_sqrsize_out = conv2_sqrsize_in - (conv_sqrkernel_size - 1)
        # a pool operation happens after conv2 out
        # (note dependence on 'forward' function below)
        l0 = int(conv2_sqrsize_out / pool_sqrkernel_size)
        self.fc1_insize = l0 * l0 * conv2_channels_out
        self.fc1 = nn.Linear(self.fc1_insize, fc2_insize)
        self.fc2 = nn.Linear(fc2_insize, self.num_classes)
        if print_model:
            print(self)
        self.to(device)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Data input to the model for the forward pass
        """
        x = F.relu(self.conv1(x))
        pl = self.pool_sqrkernel_size
        x = F.max_pool2d(x, pl, pl)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, pl, pl)
        x = x.view(-1, self.fc1_insize)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs):
        """Validate.

        Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)

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
                data, target = torch.tensor(data).to(
                    self.device), torch.tensor(target).to(
                    self.device, dtype=torch.int64)
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
        # TODO figure out a better way to pass
        #  in metric for this pytorch validate function
        output_tensor_dict = {
            TensorKey('acc', origin, round_num, True, tags):
                np.array(val_score / total_samples)
        }

        # Empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def save_native(self, filepath):
        """
        Save model in a picked file specified by the filepath.
        Uses torch.save().

        Args:
            filepath (string)                 : Path to pickle file to be
                                                created by pt.save().
        Returns:
            None
        """
        torch.save(self, filepath)
