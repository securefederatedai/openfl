# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Layers for Unet model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def soft_dice_loss(output, target):
    """Calculate loss."""
    num = target.size(0)
    m1 = output.view(num, -1)
    m2 = target.view(num, -1)
    intersection = m1 * m2
    score = 2.0 * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score.sum() / num
    return score


def soft_dice_coef(output, target):
    """Calculate soft DICE coefficient."""
    num = target.size(0)
    m1 = output.view(num, -1)
    m2 = target.view(num, -1)
    intersection = m1 * m2
    score = 2.0 * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    return score.sum()


class DoubleConv(nn.Module):
    """Pytorch double conv class."""

    def __init__(self, in_ch, out_ch):
        """Initialize layer."""
        super(DoubleConv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Do forward pass."""
        x = self.conv(x)
        return x


class Down(nn.Module):
    """Pytorch nn module subclass."""

    def __init__(self, in_ch, out_ch):
        """Initialize layer."""
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        """Do forward pass."""
        x = self.mpconv(x)
        return x

class Up(nn.Module):
    """Pytorch nn module subclass."""

    def __init__(self, in_ch, out_ch, bilinear=False):
        """Initialize layer."""
        super(Up, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        """Do forward pass."""
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
        )

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# Install dependencies if not already installed
# pip install torchvision==0.8.1
from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface, FLExperiment
import os
import PIL
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms as tsf


class KvasirShardDataset(Dataset):
    
    def __init__(self, dataset):
        self._dataset = dataset
        
        # Prepare transforms
        self.img_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize((332, 332)),
            tsf.ToTensor(),
            tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.mask_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize((332, 332), interpolation=PIL.Image.NEAREST),
            tsf.ToTensor()])
        
    def __getitem__(self, index):
        img, mask = self._dataset[index]
        img = self.img_trans(img).numpy()
        mask = self.mask_trans(mask).numpy()
        return img, mask
    
    def __len__(self):
        return len(self._dataset)

    

# Now you can implement you data loaders using dummy_shard_desc
class KvasirSD(DataInterface):

    def __init__(self, validation_fraction=1/8, **kwargs):
        super().__init__(**kwargs)
        
        self.validation_fraction = validation_fraction
        
    @property
    def shard_descriptor(self):
        return self._shard_descriptor
        
    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        Local shard_descriptor  will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor
        self._shard_dataset = KvasirShardDataset(shard_descriptor.get_dataset('train'))
        
        validation_size = max(1, int(len(self._shard_dataset) * self.validation_fraction))
        
        self.train_indeces = np.arange(len(self._shard_dataset) - validation_size)
        self.val_indeces = np.arange(len(self._shard_dataset) - validation_size, len(self._shard_dataset))
    
    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        train_sampler = SubsetRandomSampler(self.train_indeces)
        return DataLoader(
            self._shard_dataset,
            num_workers=8,
            batch_size=self.kwargs['train_bs'],
            sampler=train_sampler
        )

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        val_sampler = SubsetRandomSampler(self.val_indeces)
        return DataLoader(
            self._shard_dataset,
            num_workers=8,
            batch_size=self.kwargs['valid_bs'],
            sampler=val_sampler
        )

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        return len(self.train_indeces)

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.val_indeces)

fed_dataset = KvasirSD(train_bs=4, valid_bs=8)

import torch
import torch.nn as nn
import torch.optim as optim
"""
UNet model definition
"""
from layers import soft_dice_coef, soft_dice_loss, DoubleConv, Down, Up


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x
    
model_unet = UNet()
optimizer_adam = optim.Adam(model_unet.parameters(), lr=1e-4)

from copy import deepcopy

framework_adapter = 'openfl.plugins.frameworks_adapters.pytorch_adapter.FrameworkAdapterPlugin'
MI = ModelInterface(model=model_unet, optimizer=optimizer_adam, framework_plugin=framework_adapter)

# Save the initial model state
initial_model = deepcopy(model_unet)
TI = TaskInterface()
import torch

import tqdm
from openfl.component.aggregation_functions import Median

# The Interactive API supports registering functions definied in main module or imported.
def function_defined_in_notebook(some_parameter):
    print(f'Also I accept a parameter and it is {some_parameter}')

#The Interactive API supports overriding of the aggregation function
aggregation_function = Median()

# Task interface currently supports only standalone functions.
@TI.add_kwargs(**{'some_parameter': 42})
@TI.register_fl_task(model='unet_model', data_loader='train_loader', \
                     device='device', optimizer='optimizer')     
@TI.set_aggregation_function(aggregation_function)
def train(unet_model, train_loader, optimizer, device, loss_fn=soft_dice_loss, some_parameter=None):
    
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
        data, target = torch.tensor(data).to(device), torch.tensor(
            target).to(device, dtype=torch.float32)
        optimizer.zero_grad()
        output = unet_model(data)
        loss = loss_fn(output=output, target=target)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        
    return {'train_loss': np.mean(losses),}


@TI.register_fl_task(model='unet_model', data_loader='val_loader', device='device')     
def validate(unet_model, val_loader, device):
    print(f'\n\n TASK VALIDATE GOT DEVICE {device}\n\n')
    
    unet_model.eval()
    unet_model.to(device)
    
    val_loader = tqdm.tqdm(val_loader, desc="validate")

    val_score = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            samples = target.shape[0]
            total_samples += samples
            data, target = torch.tensor(data).to(device), \
                torch.tensor(target).to(device, dtype=torch.int64)
            output = unet_model(data)
            val = soft_dice_coef(output, target)
            val_score += val.sum().cpu().numpy()
            
    return {'dice_coef': val_score / total_samples,}