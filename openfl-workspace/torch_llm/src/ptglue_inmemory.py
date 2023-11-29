# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""


import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import horovod.torch as hvd
import torch
from datasets import Dataset
from src.glue_utils import GlueMrpc, get_dataset
from torch.utils.data import DataLoader

from openfl.utilities.data_splitters import EqualNumPyDataSplitter


class GlueMrpcFederatedDataLoader(DataLoader):
    def __init__(self, data_path, batch_size, **kwargs):
        train_set, valid_set, data_collator = get_dataset()
        self.data_splitter = EqualNumPyDataSplitter(shuffle=False)
        if isinstance(train_set, Dataset):
            self.train_set = GlueMrpc.from_dict(train_set.to_dict())
        else:
            self.train_set = train_set

        if isinstance(valid_set, Dataset):
            self.valid_set = GlueMrpc.from_dict(valid_set.to_dict())
        else:
            self.valid_set = valid_set
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.data_collator = data_collator
        self.datasave_path = f'temp_dataset_{self.data_path}'
        if kwargs.get('collaborator_count'):
            self.collaborator_count = kwargs['collaborator_count']
            data_path = int(data_path)
            train_idx = self.data_splitter.split(self.train_set, kwargs['collaborator_count'])[data_path-1]
            valid_idx = self.data_splitter.split(self.valid_set, kwargs['collaborator_count'])[data_path-1]
            train_set = self.train_set.select(train_idx)
            valid_set = self.valid_set.select(valid_idx)
            self.train_set = GlueMrpc.from_dict(train_set.to_dict())
            self.valid_set = GlueMrpc.from_dict(valid_set.to_dict())
            self.train_set.save_to_disk(f'{self.datasave_path}_train')
            self.valid_set.save_to_disk(f'{self.datasave_path}_valid')
            self.train_sampler = None
            self.test_sampler = None

        if kwargs.get('use_horovod'):
            hvd.init()
            from datasets import load_from_disk
            train_set = load_from_disk(f'{self.datasave_path}_train')
            valid_set = load_from_disk(f'{self.datasave_path}_valid')
            self.train_set = GlueMrpc.from_dict(train_set.to_dict())
            self.valid_set = GlueMrpc.from_dict(valid_set.to_dict())
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_set, num_replicas=hvd.size(), rank=hvd.rank()
            )
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.valid_set, num_replicas=hvd.size(), rank=hvd.rank()
            )
        
        self.X_train = self.train_set['input_ids']
        self.y_train = self.train_set['labels']
        self.train_loader = self.get_train_loader()

        self.X_valid = self.valid_set['input_ids']
        self.y_valid = self.valid_set['labels']
        self.val_loader = self.get_valid_loader()

        self.num_classes = 2

    def get_feature_shape(self):
        return self.train_set.get_shape()

    def get_train_loader(self, num_batches=None):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, collate_fn=self.data_collator, sampler=self.train_sampler
        )

    def get_valid_loader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size, collate_fn=self.data_collator, sampler=self.test_sampler
        )

    def get_train_data_size(self):
        return len(self.train_set)

    def get_valid_data_size(self):
        return len(self.valid_set)
    
class InHorovodGlueMrpcFederatedDataLoader(GlueMrpcFederatedDataLoader):
    def __init__(self, data_path, batch_size, **kwargs):
        train_set, valid_set, data_collator = get_dataset()
        self.data_splitter = EqualNumPyDataSplitter(shuffle=False)
        if isinstance(train_set, Dataset):
            self.train_set = GlueMrpc.from_dict(train_set.to_dict())
        else:
            self.train_set = train_set

        if isinstance(valid_set, Dataset):
            self.valid_set = GlueMrpc.from_dict(valid_set.to_dict())
        else:
            self.valid_set = valid_set
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.data_collator = data_collator
        self.datasave_path = f'temp_dataset_{self.data_path}'
        hvd.init()
        from datasets import load_from_disk
        train_set = load_from_disk(f'{self.datasave_path}_train')
        valid_set = load_from_disk(f'{self.datasave_path}_valid')
        self.train_set = GlueMrpc.from_dict(train_set.to_dict())
        self.valid_set = GlueMrpc.from_dict(valid_set.to_dict())
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_set, num_replicas=hvd.size(), rank=hvd.rank()
        )
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(
            self.valid_set, num_replicas=hvd.size(), rank=hvd.rank()
        )
        
        self.X_train = self.train_set['input_ids']
        self.y_train = self.train_set['labels']
        self.train_loader = self.get_train_loader()

        self.X_valid = self.valid_set['input_ids']
        self.y_valid = self.valid_set['labels']
        self.val_loader = self.get_valid_loader()

        self.num_classes = 2