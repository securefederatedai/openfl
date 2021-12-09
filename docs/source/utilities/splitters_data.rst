.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _data_splitting:

************************************
Dataset Splitters
************************************


|productName| allows developers to use specify custom data splits **for simulation runs on a single dataset**.

You may apply data splitters differently depending on |productName| workflow that you follow. 

Native Python API
==================

Choose from predefined |productName| data splitters functions:

- ``openfl.utilities.data_splitters.EqualNumPyDataSplitter`` (default)
- ``openfl.utilities.data_splitters.RandomNumPyDataSplitter``
- ``openfl.component.aggregation_functions.LogNormalNumPyDataSplitter`` - assumes ``data`` argument as ``np.ndarray`` of integers (labels)
- ``openfl.component.aggregation_functions.DirichletNumPyDataSplitter`` - assumes ``data`` argument as ``np.ndarray`` of integers (labels)

Or create an implementation of :class:`openfl.utilities.data_splitters.NumPyDataSplitter`
and pass it to FederatedDataset constructor as either ``train_splitter`` or ``valid_splitter`` keyword argument.


Using in Shard Descriptor
=========================

Choose from predefined |productName| data splitters functions:

- ``openfl.utilities.data_splitters.EqualNumPyDataSplitter`` (default)
- ``openfl.utilities.data_splitters.RandomNumPyDataSplitter``
- ``openfl.component.aggregation_functions.LogNormalNumPyDataSplitter`` - assumes ``data`` argument as np.ndarray of integers (labels)
- ``openfl.component.aggregation_functions.DirichletNumPyDataSplitter`` - assumes ``data`` argument as np.ndarray of integers (labels)

Or create your own implementation of :class:`openfl.component.aggregation_functions.AggregationFunction`.
After defining the splitting behavior, you need to use it on your data to perform a simulation. 

``NumPyDataSplitter`` requires a single ``split`` function.
This function receives ``data`` - NumPy array required to build the subsets of data indices (see definition of :meth:`openfl.utilities.data_splitters.NumPyDataSplitter.split`). It could be the whole dataset, or labels only, or anything else.
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

    By default, we shuffle the data and perform equal split (see :class:`openfl.utilities.data_splitters.EqualNumPyDataSplitter`).
