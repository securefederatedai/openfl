# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.interface.interactive_api.federation import Federation
from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface
from openfl.interface.interactive_api.experiment import ModelInterface, FLExperiment
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from openfl.component.aggregation_functions import Median
from layers import DoubleConv, Down, Up

client_id = 'frontend'
director_node_fqdn = 'localhost'
director_port = 50050

federation = Federation(
    client_id=client_id,
    director_node_fqdn=director_node_fqdn,
    director_port=director_port,
    tls=False
)

shard_registry = federation.get_shard_registry()
print(f'shard_registry : \n{shard_registry}')

dummy_shard_desc = federation.get_dummy_shard_descriptor(size=10)
dummy_shard_dataset = dummy_shard_desc.get_dataset('train')
sample, target = dummy_shard_dataset[0]
print(f'Sample shape : {sample.shape}, target.shape = {target.shape}')


class brainMriDataset(Dataset):

    def __init__(self, dataset) -> None:
        super().__init__()
        self._dataset = dataset

    def __getitem__(self, idx):
        img, msk = self._dataset[idx]
        img = img
        msk = msk
        return img, msk

    def __len__(self):
        return len(self._dataset)


class brainMRISD(DataInterface):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def shard_descriptor(self):
        return self._shard_descriptor

    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        self._shard_descriptor = shard_descriptor
        self._shard_dataset_train = brainMriDataset(shard_descriptor.get_dataset('train'))
        self._shard_dataset_valid = brainMriDataset(shard_descriptor.get_dataset('valid'))
        self.train_indices = np.arange(len(self._shard_dataset_train))
        self.valid_indices = np.arange(len(self._shard_dataset_valid))

    def get_train_loader(self, **kwargs):
        return DataLoader(
            self._shard_dataset_train,
            num_workers=8,
            batch_size=self.kwargs['train_bs'],
        )

    def get_valid_loader(self, **kwargs):
        return DataLoader(
            self._shard_dataset_valid,
            num_workers=8,
            batch_size=self.kwargs['valid_bs']
        )

    def get_train_data_size(self):
        return len(self.train_indices)

    def get_valid_data_size(self):
        return len(self.valid_indices)


brain_dataset = brainMRISD(train_bs=4, valid_bs=4)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 16)
        self.down0 = Down(16, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.up0 = Up(256, 128)
        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 16)
        self.outc = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down0(x1)
        x3 = self.down1(x2)
        x4 = self.down2(x3)
        x5 = self.down3(x4)
        x = self.up0(x5, x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x


# Define the model class
model_unet = UNet()

# Define the optimizer to use
optimizer_adam = optim.Adam(model_unet.parameters(), lr=1e-4)

# Framework_adapter that will convert the regular PyTorch code
framework_adapter = 'openfl.plugins.frameworks_adapters.pytorch_adapter.FrameworkAdapterPlugin'
MI = ModelInterface(model=model_unet, optimizer=optimizer_adam, framework_plugin=framework_adapter)

# Save the initial model state
initial_model = deepcopy(model_unet)

TI = TaskInterface()


# The Interactive API supports registering functions definied in main module or imported.
def function_defined_in_notebook(some_parameter):
    print(f'Also I accept a parameter and it is {some_parameter}')


def soft_dice_coef(output, target, epsilon=1):
    """Calculate loss."""
    inter = torch.dot(output.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(output) + torch.sum(target)
    if sets_sum.item() == 0:
        sets_sum = 2 * inter
    return (2 * inter + epsilon) / (sets_sum + epsilon)


def soft_dice_loss(output, target, epsilon=1):
    return 1 - soft_dice_coef(output, target, epsilon=epsilon)


class Config:
    num_workers = 2
    batch_size = 8
    n_epoches = 20
    lr = 1e-4


# The Interactive API supports overriding of the aggregation function
aggregation_function = Median()


# Task interface currently supports only standalone functions.
@TI.add_kwargs(**{'some_parameter': 0})  # Just to showcase how we can send additional paramters
@TI.register_fl_task(model='unet_model', data_loader='train_loader',
                     device='device', optimizer='optimizer')
@TI.set_aggregation_function(aggregation_function)
def train(unet_model, train_loader, optimizer, device, loss_fn=soft_dice_loss,
          some_parameter=None):

    """
    The following constructions, that may lead to resource race
    is no longer needed:

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    """

    print(f'\n\n TASK TRAIN GOT DEVICE {device}\n\n')

    function_defined_in_notebook(some_parameter)

    train_loader = tqdm.tqdm(train_loader, desc="train")

    unet_model.train()
    unet_model.to(device)

    losses = []

    for data, target in train_loader:
        data, target = torch.tensor(data).to(device, dtype=torch.float32).requires_grad_(True),
        torch.tensor(target).to(device, dtype=torch.float32).requires_grad_(True)
        optimizer.zero_grad()
        output = unet_model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())

    return {'train_loss': np.mean(losses)}


@TI.register_fl_task(model='unet_model', data_loader='val_loader', device='device')
def validate(unet_model, val_loader, device, loss_fn=soft_dice_coef):
    print(f'\n\n TASK VALIDATE GOT DEVICE {device}\n\n')

    unet_model.eval()
    unet_model.to(device)

    val_loader = tqdm.tqdm(val_loader, desc="validate")

    val_losses = []
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            samples = target.shape[0]
            total_samples += samples
            data, target = torch.tensor(data).to(device, dtype=torch.float32), \
                torch.tensor(target).to(device, dtype=torch.float32)
            output = unet_model(data)
            val = loss_fn(output, target)
            val_losses.append(val.detach().cpu().numpy())

    return {'val_loss': np.mean(val_losses)}


# create an experimnet in federation
experiment_name = 'brain_MRI_experiment'
fl_experiment = FLExperiment(federation=federation, experiment_name=experiment_name)

fl_experiment.start(model_provider=MI,
                    task_keeper=TI,
                    data_loader=brain_dataset,
                    rounds_to_train=50,
                    opt_treatment='CONTINUE_GLOBAL',
                    device_assignment_policy='CUDA_PREFERRED')

fl_experiment.stream_metrics()
