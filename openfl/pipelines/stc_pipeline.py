# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""STCPipelinemodule."""

import numpy as np
import gzip as gz

from .pipeline import TransformationPipeline, Transformer


class SparsityTransformer(Transformer):
    """A transformer class to sparsify input data."""

    def __init__(self, p=0.01):
        """Initialize.

        Args:
            p (float): sparsity ratio (Default=0.01)
        """
        self.lossy = True
        self.p = p

    def forward(self, data, **kwargs):
        """Sparsify data and pass over only non-sparsified elements by reducing the array size.

        Args:
            data: an numpy array from the model tensor_dict

        Returns:
            condensed_data: an numpy array being sparsified.
            metadata: dictionary to store a list of meta information.
        """
        metadata = {'int_list': list(data.shape)}
        # sparsification
        data = data.astype(np.float32)
        flatten_data = data.flatten()
        n_elements = flatten_data.shape[0]
        k_op = int(np.ceil(n_elements * self.p))
        topk, topk_indices = self._topk_func(flatten_data, k_op)
        #
        condensed_data = topk
        sparse_data = np.zeros(flatten_data.shape)
        sparse_data[topk_indices] = topk
        nonzero_element_bool_indices = sparse_data != 0.0
        metadata['bool_list'] = list(nonzero_element_bool_indices)
        return condensed_data, metadata
        # return sparse_data, metadata

    def backward(self, data, metadata, **kwargs):
        """Recover data array with the right shape and numerical type.

        Args:
            data: an numpy array with non-zero values.
            metadata: dictionary to contain information for recovering back to original data array.

        Returns:
            recovered_data: an numpy array with original shape.
        """
        data = data.astype(np.float32)
        data_shape = metadata['int_list']
        nonzero_element_bool_indices = list(metadata['bool_list'])
        recovered_data = np.zeros(data_shape).reshape(-1).astype(np.float32)
        recovered_data[nonzero_element_bool_indices] = data
        recovered_data = recovered_data.reshape(data_shape)
        return recovered_data

    @staticmethod
    def _topk_func(x, k):
        """Select top k values.

        Args:
            x: an numpy array to be sorted out for top-k components.
            k: k most maximum values.

        Returns:
            topk_mag: components with top-k values.
            indices: indices of the top-k components.
        """
        # quick sort as default on magnitude
        idx = np.argsort(np.abs(x))
        # sorted order, the right most is the largest magnitude
        length = x.shape[0]
        start_idx = length - k
        # get the top k magnitude
        topk_mag = np.asarray(x[idx[start_idx:]])
        indices = np.asarray(idx[start_idx:])
        if min(topk_mag) - 0 < 10e-8:  # avoid zeros
            topk_mag = topk_mag + 10e-8
        return topk_mag, indices


class TernaryTransformer(Transformer):
    """A transformer class to ternerize input data."""

    def __init__(self):
        """Initialize."""
        self.lossy = True

    def forward(self, data, **kwargs):
        """Ternerize data into positive mean value, negative mean value and zero value.

        Args:
            data: an flattened numpy array

        Returns:
            int_data: an numpy array being terneraized.
            metadata: dictionary to store a list of meta information.
        """
        # ternarization, data is sparse and flattened
        mean_topk = np.mean(np.abs(data))
        out_ = np.where(data > 0.0, mean_topk, 0.0)
        out = np.where(data < 0.0, -mean_topk, out_)
        int_array, int2float_map = self._float_to_int(out)
        metadata = {'int_to_float': int2float_map}
        return int_array, metadata

    def backward(self, data, metadata, **kwargs):
        """Recover data array back to the original numerical type.

        Args:
            data: an numpy array with non-zero values.

        Returns:
            metadata: dictionary to contain information for recovering back to original data array.
            data (return): an numpy array with original numerical type.
        """
        # TODO
        import copy
        data = copy.deepcopy(data)
        int2float_map = metadata['int_to_float']
        for key in int2float_map:
            indices = data == key
            data[indices] = int2float_map[key]
        return data

    @staticmethod
    def _float_to_int(np_array):
        """Create look-up table for conversion between floating and integer types.

        Args:
            np_array:

        Returns:
            int_array:
            int_to_float_map:

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
    """A transformer class to losslessly compress data."""

    def __init__(self):
        """Initialize."""
        self.lossy = False

    def forward(self, data, **kwargs):
        """Compress data into numpy of float32.

        Args:
            data: an numpy array with non-zero values

        Returns:
            compressed_bytes :
            metadata: dictionary to contain information for recovering back to original data array

        """
        bytes_ = data.astype(np.float32).tobytes()
        compressed_bytes = gz.compress(bytes_)
        metadata = {}
        return compressed_bytes, metadata

    def backward(self, data, metadata, **kwargs):
        """Decompress data into numpy of float32.

        Args:
            data: an numpy array with non-zero values
            metadata: dictionary to contain information for recovering back to original data array

        Returns:
            data:
        """
        decompressed_bytes_ = gz.decompress(data)
        data = np.frombuffer(decompressed_bytes_, dtype=np.float32)
        return data


class STCPipeline(TransformationPipeline):
    """A pipeline class to compress data lossly using sparsity and ternerization methods."""

    def __init__(self, p_sparsity=0.01, n_clusters=6, **kwargs):
        """Initialize a pipeline of transformers.

        Args:
            p_sparsity (float): Sparsity factor (Default=0.01)
            n_cluster (int): Number of K-Means clusters (Default=6)

        Returns:
            Data compression transformer pipeline object
        """
        # instantiate each transformer
        self.p = p_sparsity
        transformers = [SparsityTransformer(self.p), TernaryTransformer(), GZIPTransformer()]
        super(STCPipeline, self).__init__(transformers=transformers, **kwargs)
