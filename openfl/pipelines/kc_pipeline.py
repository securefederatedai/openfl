# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""KCPipeline module."""


import numpy as np
import gzip as gz
import copy as co

from sklearn import cluster

from .pipeline import TransformationPipeline, Transformer


class KmeansTransformer(Transformer):
    """A K-means transformer class to quantize input data."""

    def __init__(self, n_cluster=6):
        """Class initializer.

        Args:
            n_cluster (int): Number of clusters for the K-means
        """
        self.lossy = True
        self.n_cluster = n_cluster

    def forward(self, data, **kwargs):
        """
        Quantize data into n_cluster levels of values.

        Args:
            data: an numpy array from the model tensor_dict
            data: an numpy array being quantized
            **kwargs: Variable arguments to pass
        """
        metadata = {'int_list': list(data.shape)}
        # clustering
        k_means = cluster.KMeans(n_clusters=self.n_cluster, n_init=self.n_cluster)
        data = data.reshape((-1, 1))
        k_means.fit(data)
        quantized_values = k_means.cluster_centers_.squeeze()
        indices = k_means.labels_
        quant_array = np.choose(indices, quantized_values)
        int_array, int2float_map = self._float_to_int(quant_array)
        metadata['int_to_float'] = int2float_map

        return int_array, metadata

    def backward(self, data, metadata, **kwargs):
        """Recover data array back to the original numerical type and the shape.

        Args:
            data: an flattened numpy array
            metadata: dictionary to contain information for recovering ack to original data array

        Returns:
            data: Numpy array with original numerical type and shape
        """
        # convert back to float
        # TODO
        data = co.deepcopy(data)
        int2float_map = metadata['int_to_float']
        for key in int2float_map:
            indices = data == key
            data[indices] = int2float_map[key]
        data_shape = list(metadata['int_list'])
        data = data.reshape(data_shape)
        return data

    @staticmethod
    def _float_to_int(np_array):
        """Create look-up table for conversion between floating and integer types.

        Args:
            np_array: A Numpy array

        Returns:
            int_array: The input Numpy float array converted to an integer array
            int_to_float_map
        """
        flatten_array = np_array.reshape(-1)
        unique_value_array = np.unique(flatten_array)
        int_array = np.zeros(flatten_array.shape, dtype=np.int)
        int_to_float_map = {}
        float_to_int_map = {}
        # create table
        for idx, u_value in enumerate(unique_value_array):
            int_to_float_map.update({idx: u_value})
            float_to_int_map.update({u_value: idx})
            # assign to the integer array
            indices = np.where(flatten_array == u_value)
            int_array[indices] = idx
        int_array = int_array.reshape(np_array.shape)
        return int_array, int_to_float_map


class GZIPTransformer(Transformer):
    """A GZIP transformer class to losslessly compress data."""

    def __init__(self):
        """Initialize."""
        self.lossy = False

    def forward(self, data, **kwargs):
        """Compress data into bytes.

        Args:
            data: A Numpy array

        Returns:
            GZIP compressed data object
        """
        bytes_ = data.astype(np.float32).tobytes()
        compressed_bytes_ = gz.compress(bytes_)
        metadata = {}
        return compressed_bytes_, metadata

    def backward(self, data, metadata, **kwargs):
        """Decompress data into numpy of float32.

        Args:
            data: Compressed GZIP data
            metadata:
            **kwargs: Additional parameters to pass to the function

        Returns:
            data: Numpy array
        """
        decompressed_bytes_ = gz.decompress(data)
        data = np.frombuffer(decompressed_bytes_, dtype=np.float32)
        return data


class KCPipeline(TransformationPipeline):
    """A pipeline class to compress data lossly using k-means and GZIP methods."""

    def __init__(self, p_sparsity=0.01, n_clusters=6, **kwargs):
        """Initialize a pipeline of transformers.

        Args:
            p_sparsity (float): Amount of sparsity for compression (Default = 0.01)
            n_clusters (int): Number of K-mean cluster

        Return:
            Transformer class object
        """
        # instantiate each transformer
        self.p = p_sparsity
        self.n_cluster = n_clusters
        transformers = [KmeansTransformer(self.n_cluster), GZIPTransformer()]
        super(KCPipeline, self).__init__(transformers=transformers, **kwargs)
