# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import FastEstimatorDataLoader

import fastestimator as fe
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop.univariate import Normalize


class FastEstimatorCifarInMemory(FastEstimatorDataLoader):
    """TensorFlow Data Loader for MNIST Dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """
        Initialize.

        Args:
            data_path: File path for the dataset
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and
             load_mnist_shard
        """
        # TODO: We should be downloading the dataset shard into a directory
        # TODO: There needs to be a method to ask how many collaborators and
        #  what index/rank is this collaborator.
        # Then we have a way to automatically shard based on rank and size
        # of collaborator list.

        train_data, eval_data = cifar10.load_data()
        test_data = eval_data.split(0.5)

        collaborator_count = kwargs['collaborator_count']

        train_data, eval_data, test_data = self.split_data(
            train_data,
            eval_data,
            test_data,
            int(data_path),
            collaborator_count
        )

        print(f"train_data = {train_data}")
        print(f"eval_data = {eval_data}")
        print(f"test_data = {test_data}")

        print(f"batch_size = {batch_size}")

        super().__init__(fe.Pipeline(
            train_data=train_data,
            eval_data=eval_data,
            test_data=test_data,
            batch_size=batch_size,
            ops=[
                Normalize(inputs="x", outputs="x",
                          mean=(0.4914, 0.4822, 0.4465),
                          std=(0.2471, 0.2435, 0.2616))
            ]), **kwargs)

    def split_data(self, train, eva, test, rank, collaborator_count):
        """Split data into N parts, where N is the collaborator count."""
        if collaborator_count == 1:
            return train, eva, test

        fraction = [1.0 / float(collaborator_count)]
        fraction *= (collaborator_count - 1)

        # Expand the split list into individual parameters
        train_split = train.split(*fraction)
        eva_split = eva.split(*fraction)
        test_split = test.split(*fraction)

        train = [train]
        eva = [eva]
        test = [test]

        if type(train_split) is not list:
            train.append(train_split)
            eva.append(eva_split)
            test.append(test_split)
        else:
            # Combine all partitions into a single list
            train = [train] + train_split
            eva = [eva] + eva_split
            test = [test] + test_split

        # Extract the right shard
        train = train[rank - 1]
        eva = eva[rank - 1]
        test = test[rank - 1]

        return train, eva, test
