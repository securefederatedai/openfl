# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""KCPipeline module."""

import copy as co
import gzip as gz

import numpy as np
from sklearn import cluster

from openfl.pipelines.pipeline import TransformationPipeline, Transformer


class KmeansTransformer(Transformer):
    """K-means transformer class for quantizing input data.

    This class is a transformer that uses the K-means method for quantization.

    Attributes:
        n_cluster (int): The number of clusters for the K-means.
        lossy (bool): Indicates if the transformer is lossy.
    """

    def __init__(self, n_cluster=6):
        """Initialize KmeansTransformer.

        Args:
            n_cluster (int, optional): The number of clusters for the K-means.
                Defaults to 6.
        """
        self.lossy = True
        self.n_cluster = n_cluster

    def forward(self, data, **kwargs):
        """Quantize data into n_cluster levels of values.

        Args:
            data: The data to be quantized.
            **kwargs: Variable arguments to pass.

        Returns:
            int_array: The quantized data.
            metadata: The metadata for the quantization.
        """
        metadata = {"int_list": list(data.shape)}
        # clustering
        k_means = cluster.KMeans(n_clusters=self.n_cluster, n_init=self.n_cluster)
        data = data.reshape((-1, 1))
        if data.shape[0] >= self.n_cluster:
            k_means = cluster.KMeans(n_clusters=self.n_cluster, n_init=self.n_cluster)
            k_means.fit(data)
            quantized_values = k_means.cluster_centers_.squeeze()
            indices = k_means.labels_
            quant_array = np.choose(indices, quantized_values)
        else:
            quant_array = data

        int_array, int2float_map = self._float_to_int(quant_array)
        metadata["int_to_float"] = int2float_map

        return int_array, metadata

    def backward(self, data, metadata, **kwargs):
        """Recover data array back to the original numerical type and the
        shape.

        Args:
            data: The flattened numpy array.
            metadata: The dictionary containing information for recovering to
                original data array.

        Returns:
            data: The numpy array with original numerical type and shape.
        """
        # convert back to float
        # TODO
        data = co.deepcopy(data)
        int2float_map = metadata["int_to_float"]
        for key in int2float_map:
            indices = data == key
            data[indices] = int2float_map[key]
        data_shape = list(metadata["int_list"])
        data = data.reshape(data_shape)
        return data

    @staticmethod
    def _float_to_int(np_array):
        """Create look-up table for conversion between floating and integer
        types.

        Args:
            np_array: A Numpy array.

        Returns:
            int_array: The input Numpy float array converted to an integer
                array.
            int_to_float_map: The dictionary mapping integers to floats.
        """
        flatten_array = np_array.reshape(-1)
        unique_value_array = np.unique(flatten_array)
        int_array = np.zeros(flatten_array.shape, dtype=np.int32)
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
    """GZIP transformer class for losslessly compressing data.

    Attributes:
        lossy (bool): Indicates if the transformer is lossy.
    """

    def __init__(self):
        """Initialize GZIPTransformer."""
        self.lossy = False

    def forward(self, data, **kwargs):
        """Compress data into bytes.

        Args:
            data: A Numpy array.

        Returns:
            compressed_bytes_: The GZIP compressed data object.
            metadata: An empty dictionary.
        """
        bytes_ = data.astype(np.float32).tobytes()
        compressed_bytes_ = gz.compress(bytes_)
        metadata = {}
        return compressed_bytes_, metadata

    def backward(self, data, metadata, **kwargs):
        """Decompress data into numpy of float32.

        Args:
            data: Compressed GZIP data
            metadata: An empty dictionary.
            **kwargs: Additional parameters to pass to the function

        Returns:
            data: The decompressed data as a numpy array.
        """
        decompressed_bytes_ = gz.decompress(data)
        data = np.frombuffer(decompressed_bytes_, dtype=np.float32)
        return data


class KCPipeline(TransformationPipeline):
    """A pipeline class to compress data lossly using k-means and GZIP methods.

    Attributes:
        p (float): The amount of sparsity for compression.
        n_cluster (int): The number of K-mean clusters.
    """

    def __init__(self, p_sparsity=0.01, n_clusters=6, **kwargs):
        """Initialize a pipeline of transformers.

        Args:
            p_sparsity (float, optional): The amount of sparsity for
                compression. Defaults to 0.01.
            n_clusters (int, optional): The number of K-mean clusters.
                Defaults to 6.
            **kwargs: Additional keyword arguments.
        """
        # instantiate each transformer
        self.p = p_sparsity
        self.n_cluster = n_clusters
        transformers = [KmeansTransformer(self.n_cluster), GZIPTransformer()]
        super().__init__(transformers=transformers, **kwargs)
