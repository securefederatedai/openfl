.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _data_splitting:
===============================
Specifying custom data splits
===============================

-------------------------------
Usage
-------------------------------
|productName| allows developers to use custom data splits **for single-node simulation**.
In order to do this, you should:

Python API
==========

Choose from predefined |productName| aggregation functions:

- ``openfl.plugins.data_splitters.EqualNumPyDataSplitter`` (default)
- ``openfl.plugins.data_splitters.RandomNumPyDataSplitter``
- ``openfl.component.aggregation_functions.LogNormalNumPyDataSplitter`` - assumes ``data`` argument as ``np.ndarray`` of integers (labels)
- ``openfl.component.aggregation_functions.DirichletNumPyDataSplitter`` - assumes ``data`` argument as ``np.ndarray`` of integers (labels)
Or create an implementation of :class:`openfl.plugins.data_splitters.NumPyDataSplitter`
and pass it to FederatedDataset constructor as either ``train_splitter`` or ``valid_splitter`` keyword argument.


CLI
====

Choose from predefined |productName| aggregation functions:

- ``openfl.plugins.data_splitters.EqualNumPyDataSplitter`` (default)
- ``openfl.plugins.data_splitters.RandomNumPyDataSplitter``
- ``openfl.component.aggregation_functions.LogNormalNumPyDataSplitter`` - assumes ``data`` argument as np.ndarray of integers (labels)
- ``openfl.component.aggregation_functions.DirichletNumPyDataSplitter`` - assumes ``data`` argument as np.ndarray of integers (labels)
Or create your own implementation of :class:`openfl.component.aggregation_functions.AggregationFunctionInterface`.
After defining the splitting behavior, you need to use it on your data to perform a simulation. 

``NumPyDataSplitter`` requires a single ``split`` function.
This function receives ``data`` - NumPy array required to build the subsets of data indices (see definition of :meth:`openfl.plugins.data_splitters.NumPyDataSplitter.split`). It could be the whole dataset, or labels only, or anything else.
``split`` function returns a list of lists of indices which represent the collaborator-wise indices groups.

    .. code-block:: python
        X_train, y_train = ... # train set
        X_valid, y_valid = ... # valid set
        train_splitter = RandomNumPyDataSplitter()
        valid_splitter = RandomNumPyDataSplitter()
        # collaborator_count value is passed to DataLoader constructor
        # shard_num can be evaluated from data_path
        train_idx = train_splitter.split(y_train, collaborator_count)[shard_num]
        valid_idx = valid_splitter.split(y_valid, collaborator_count)[shard_num]
        X_train_shard = X_train[train_idx]
        X_valid_shard = X_valid[valid_idx]

.. note::
    By default, we shuffle the data and perform equal split (see :class:`openfl.plugins.data_splitters.EqualNumPyDataSplitter`).
