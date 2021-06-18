# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from typing import Iterator
from typing import Tuple

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import TensorKey
from pytorch3dunet.unet3d.model import UNet3D

from openfl.utilities import Metric


def soft_dice_loss(output, target):
    """Dice loss function."""
    num = target.size(0)
    m1 = output.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score.sum() / num
    return score


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
        self._init_optimizer()
        self.loss_fn = soft_dice_loss
        self.initialize_tensorkeys_for_functions()

    def _init_optimizer(self):
        """Initialize the optimizer."""
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def init_network(self,
                     device,
                     **kwargs):
        self.model = UNet3D(1, 1)
        self.to(device)
        
    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Data input to the model for the forward pass
        """
        
        return self.model(x)

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

    def train_epoch(self, batch_generator: Iterator[Tuple[np.ndarray, np.ndarray]]) -> Metric:
        """Train single epoch.

        Override this function in order to use custom training.

        Args:
            batch_generator: Train dataset batch generator. Yields (samples, targets) tuples of
            size = `self.data_loader.batch_size`.
        Returns:
            Metric: An object containing name and np.ndarray value.
        """
        import time
        start = time.time()
        results = super().train_epoch(batch_generator=batch_generator)
        end = time.time()
        with open('times_torch_cnn_3dunet.csv', 'a+') as f:
            f.write(f'{end-start}\n')
        return results

    def reset_opt_vars(self):
        """Reset optimizer variables.

        Resets the optimizer state variables.

        """
        self._init_optimizer()
