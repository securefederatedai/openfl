# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import TensorKey, split_tensor_dict_for_holdouts


def cross_entropy(output, target):
    """Binary cross-entropy metric.

    Args:
        output: The mode prediction
        target: The target (ground truth label)

    Returns:
        Binary cross-entropy with logits

    """
    return F.binary_cross_entropy_with_logits(input=output, target=target)


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
        self.loss_fn = cross_entropy
        self.initialize_tensorkeys_for_functions()

    def _init_optimizer(self):
        """Initialize the optimizer."""
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

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

        """
        """
        FIXME: We are tracking only side lengths (rather than
        length and width) as we are assuming square
        shapes for feature and kernels.
        In order that all of the input and activation components are
        used (not cut off), we rely on a criterion: appropriate integers
        are divisible so that all casting to int perfomed below does no
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
        return F.log_softmax(x, dim=1)

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
            loader = tqdm.tqdm(loader, desc="validate")

        with torch.no_grad():
            for data, target in loader:
                samples = target.shape[0]
                total_samples += samples
                data, target = torch.tensor(data).to(
                    self.device), torch.tensor(target).to(
                    self.device, dtype=torch.int64)
                output = self(data)
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                target_categorical = target.argmax(dim=1, keepdim=True)
                val_score += pred.eq(target_categorical).sum().cpu().numpy()

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

    def train_batches(self, col_name, round_num, input_tensor_dict,
                      num_batches=None, use_tqdm=False, **kwargs):
        """Train batches.

        Train the model on the requested number of batches.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            num_batches:         The number of batches to
                                 train on before returning
            use_tqdm (bool):     Use tqdm to print a progress
                                 bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """
        self.rebuild_model(round_num, input_tensor_dict)
        # set to "training" mode
        self.train()

        losses = []

        loader = self.data_loader.get_train_loader(num_batches)
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="train epoch")
        for data, target in loader:
            data, target = torch.tensor(data).to(
                self.device), torch.tensor(target).to(
                self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.loss_fn(output=output, target=target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().numpy())

        # Output metric tensors (scalar)
        origin = col_name
        tags = ('trained',)
        output_metric_dict = {
            TensorKey(
                self.loss_fn.__name__, origin, round_num, True, ('metric',)
            ): np.array(np.mean(losses))}

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )

        # Create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags):
                nparray for tensor_name, nparray in global_model_dict.items()}
        # Create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags):
                nparray for tensor_name, nparray in local_model_dict.items()}
        # The train/validate aggregated function of the next
        # round will look for the updated model parameters.
        # This ensures they will be resolved locally
        next_local_tensorkey_model_dict = {TensorKey(
            tensor_name, origin, round_num + 1, False, ('model',)): nparray
            for tensor_name, nparray in local_model_dict.items()}

        global_tensor_dict = {**output_metric_dict, **global_tensorkey_model_dict}
        local_tensor_dict = {**local_tensorkey_model_dict, **next_local_tensorkey_model_dict}

        # Update the required tensors if they need to be
        # pulled from the aggregator
        # TODO this logic can break if different collaborators
        #  have different roles between rounds.
        # For example, if a collaborator only performs validation
        # in the first round but training
        # in the second, it has no way of knowing the optimizer
        # state tensor names to request from the aggregator
        # because these are only created after training occurs.
        # A work around could involve doing a single epoch of training
        # on random data to get the optimizer names,
        # and then throwing away the model.
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        # This will signal that the optimizer values are now present,
        # and can be loaded when the model is rebuilt
        self.train_round_completed = True

        # Return global_tensor_dict, local_tensor_dict
        return global_tensor_dict, local_tensor_dict

    def reset_opt_vars(self):
        """Reset optimizer variables.

        Resets the optimizer state variables.

        """
        self._init_optimizer()
