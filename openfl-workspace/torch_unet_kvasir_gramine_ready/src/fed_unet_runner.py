# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import TensorKey
from .pt_unet_parts import DoubleConv
from .pt_unet_parts import Down
from .pt_unet_parts import soft_dice_coef
from .pt_unet_parts import soft_dice_loss
from .pt_unet_parts import Up


class PyTorchFederatedUnet(PyTorchTaskRunner):
    """Simple Unet for segmentation."""

    def __init__(self, device='cpu', **kwargs):
        """Initialize.

        Args:
            device: The hardware device to use for training (Default = "cpu")
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(device=device, **kwargs)
        self.init_network(device=self.device, **kwargs)
        self._init_optimizer()
        self.loss_fn = soft_dice_loss
        self.initialize_tensorkeys_for_functions()

    def _init_optimizer(self):
        """Initialize the optimizer."""
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def init_network(self,
                     device,
                     n_channels,
                     n_classes,
                     print_model=True,
                     **kwargs):
        """Create the network (model).

        Args:
            device: The hardware device to use for training
            n_channels: Number of input image channels
            n_classes: Number of output classes (1 for segmentation)
            print_model: Print the model topology (Default=True)
            **kwargs: Additional arguments to pass to the function

        """
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, self.n_classes, 1)
        if print_model:
            print(self)
        self.to(device)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Data input to the model for the forward pass
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x

    def validate_task(self, col_name, round_num, input_tensor_dict, use_tqdm=True, **kwargs):
        """Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm:     Use tqdm to print a progress bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.eval()
        self.to(self.device)
        val_score = 0
        total_samples = 0

        loader = self.data_loader.get_valid_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc='validate')

        with torch.no_grad():
            for data, target in loader:
                samples = target.shape[0]
                total_samples += samples
                data, target = torch.tensor(data).to(self.device), torch.tensor(
                    target).to(self.device)
                output = self(data)
                # get the index of the max log-probability
                val = soft_dice_coef(output, target)
                val_score += val.sum().cpu().numpy()

        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)
        # TODO figure out a better way to pass in metric for this pytorch
        #  validate function
        output_tensor_dict = {
            TensorKey('dice_coef', origin, round_num, True, tags):
                np.array(val_score / total_samples)
        }

        return output_tensor_dict, {}

    def reset_opt_vars(self):
        """Reset optimizer variables.

        Resets the optimizer state variables.

        """
        self._init_optimizer()
