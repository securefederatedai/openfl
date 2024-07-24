# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""


import os
import sys
from logging import getLogger

from openfl.federated.data import DataLoader

from openfl.utilities.data_splitters import EqualNumPyDataSplitter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from src.dataset_utils import get_dataset  # noqa: E402

logger = getLogger(__name__)


class MedQuadDataLoader(DataLoader):
    def __init__(self, data_path, batch_size, **kwargs):
        logger.info("get dataset")
        self.sequence_max_length = kwargs.setdefault("sequence_max_length", 512)
        self.val_set_size = kwargs.setdefault("val_set_size", 2000)
        train_set, valid_set, data_collator, tokenizer = get_dataset(
            self.sequence_max_length, self.val_set_size
        )
        self.tokenizer = tokenizer
        self.data_splitter = EqualNumPyDataSplitter(shuffle=False)

        self.batch_size = batch_size
        self.data_collator = data_collator
        logger.info("load from disk")
        if kwargs.get("collaborator_count"):
            self.collaborator_count = kwargs["collaborator_count"]
            data_path = int(data_path)
            train_idx = self.data_splitter.split(
                train_set, kwargs["collaborator_count"]
            )[data_path - 1]
            valid_idx = self.data_splitter.split(
                valid_set, kwargs["collaborator_count"]
            )[data_path - 1]
            self.train_set = train_set.select(train_idx)
            self.valid_set = valid_set.select(valid_idx)

        self.X_train = self.train_set["text"]
        self.y_train = self.train_set["text"]
        self.train_loader = self.get_train_loader()

        self.X_valid = self.valid_set["text"]
        self.y_valid = self.valid_set["text"]
        self.val_loader = self.get_valid_loader()

        self.num_classes = 2

    def get_feature_shape(self):
        return self.sequence_max_length

    def get_train_loader(self):
        return self.train_set

    def get_valid_loader(self):
        return self.valid_set

    def get_train_data_size(self):
        return len(self.train_set)

    def get_valid_data_size(self):
        return len(self.valid_set)
