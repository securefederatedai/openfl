"""Copyright (C) 2020-2021 Intel Corporation
   SPDX-License-Identifier: Apache-2.0

Licensed subject to the terms of the separately executed evaluation
license agreement between Intel Corporation and you.
"""
from logging import getLogger

import numpy as np
import src.dataloader_utils as dlu

from openfl.federated import KerasDataLoader

logger = getLogger(__name__)


class NLPDataLoader(KerasDataLoader):
    """NLP Dataloader template."""

    def __init__(self, collaborator_count, split_ratio,
                 num_samples, data_path, batch_size, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data Returns:
            batch_size: The batch size of the data loader tuple: shape of an example feature array
            **kwargs: Additional arguments, passed to super init and load_mnist_shard

        Returns:
           none
        """
        self.shard_num = data_path
        self.data_path = dlu.download_data_()

        self.batch_size = batch_size

        train, valid, details = dlu.load_shard(collaborator_count, self.shard_num,
                                               self.data_path, num_samples, split_ratio)

        self.num_samples = details['num_samples']
        self.num_encoder_tokens = details['num_encoder_tokens']
        self.num_decoder_tokens = details['num_decoder_tokens']
        self.max_encoder_seq_length = details['max_encoder_seq_length']
        self.max_decoder_seq_length = details['max_decoder_seq_length']

        self.X_train = [train[0], train[1]]
        self.y_train = train[2]
        self.X_valid = [valid[0], valid[1]]
        self.y_valid = valid[2]

    def get_feature_shape(self):
        """Get the shape of an example feature array."""
        return self.X_train[0].shape

    def get_train_loader(self, batch_size=None):
        """
        Get training data loader.

        Returns
        -------
        loader object
        """
        return self._get_batch_generator(X1=self.X_train[0], X2=self.X_train[1],
                                         y=self.y_train, batch_size=batch_size)

    def get_valid_loader(self, batch_size=None):
        """
        Get validation data loader.

        Returns:
            loader object
        """
        return self._get_batch_generator(X1=self.X_valid[0], X2=self.X_valid[1],
                                         y=self.y_valid, batch_size=batch_size)

    def get_train_data_size(self):
        """
        Get total number of training samples.

        Returns:
            int: number of training samples
        """
        return self.X_train[0].shape[0]

    def get_valid_data_size(self):
        """
        Get total number of validation samples.

        Returns:
            int: number of validation samples
        """
        return self.X_valid[0].shape[0]

    @staticmethod
    def _batch_generator(X1, X2, y, idxs, batch_size, num_batches):
        """
        Generate batch of data.

        Args:
            X: input data
            y: label data
            idxs: The index of the dataset
            batch_size: The batch size for the data loader
            num_batches: The number of batches
        Yields:
            tuple: input data, label data
        """
        for i in range(num_batches):
            a = i * batch_size
            b = a + batch_size
            yield [X1[idxs[a:b]], X2[idxs[a:b]]], y[idxs[a:b]]

    def _get_batch_generator(self, X1, X2, y, batch_size):
        """
        Return the dataset generator.

        Args:
            X1: input data  (encoder)
            X2: input data  (decoder)
            y: label data
            batch_size: The batch size for the data loader
        """
        if batch_size is None:
            batch_size = self.batch_size
        # shuffle data indices
        idxs = np.random.permutation(np.arange(X1.shape[0]))
        # compute the number of batches
        num_batches = int(np.ceil(X1.shape[0] / batch_size))
        # build the generator and return it
        # TODO: due to _batch_generator(X1, ...) has first param X1, all params here will be moved,
        #       X1 -> X2, X2 -> y, y -> idxs, idxs -> batch_size, batch_size -> num_batches,
        #       and num_batches -> should be unexpected in this function
        return self._batch_generator(X1, X2, y, idxs, batch_size, num_batches)
