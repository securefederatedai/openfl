.. # Copyright (C) 2025 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

*************************************
Verifiable Datasets and Data Sources
*************************************

.. _verifiable_datasets_overview:

To accommodate for the proliferation of data sources and the need for trusted datasets, OpenFL provides a hierarchy of utility classes to build and verify datasets. 
This includes an extensible class hierarchy that enables the creation of datasets from various data sources, such as local file system, object storage and others.

The central abstraction is the :code:`VerifiableDatasetInfo` class that encapsulates the dataset's metadata and provides a method for verifying the integrity of the dataset.
A dataset can be built from multiple data sources (not necessarily of the same type):

.. mermaid:: ../../mermaid/verifiable_dataset_info.mmd
    :caption: Verifiable Dataset with Multiple Data Sources
    :align: center

The :code:`VerifiableDatasetInfo` class can then be used to create higher-order dataset classes that enable iterating through multiple data sources, while verifying integrity if required.
The :code:`root_hash` is used as a reference for integrity when loading items from the the data sources in the :code:`VerifiableDatasetInfo` object.

OpenFL comes with a toolbox of dataset layout classes per ML framework. For PyTorch's :code:`torch.utils.data.Dataset` OpenFL curently provides: 

- :code:`FolderDataset` - represents an iterable folder-layout dataset from a single data source, by implementing the :code:`__getitem__` method.
- :code:`ImageFolder` - a specialization of the :code:`FolderDataset` that is able to load binary images from a foler-like structure
- :code:`VerifiableMapStyleDataset` - a base class for map-style datasets that can be built from multiple data sources (as specified by a :code:`VerifiableDatasetInfo` object), including integrity checks.
- :code:`VerifiableImageFolder` - a specialization of the :code:`VerifiableMapStyleDataset` encapsulating a collection of :code:`ImageFolder` datasets

Note that the all those classes (directly or indirectly) extend :code:`torch.utils.data.DataLoader`, and are therefore compatible with all PyTorch utilities for pre-processing data sets.
A similar class hierarchy can be created for other ML frameworks that offer dataset utilities, such as TensorFlow.

.. mermaid:: ../../mermaid/verifiable_image_folder.mmd
    :caption: Dataset hierarchy
    :align: center

A practical example for the :code:`VerifiableImageFolder` backed by :code:`S3DataSource` is provided in the `s3_histology <https://github.com/securefederatedai/openfl/tree/develop/openfl-workspace/torch/histology_s3>`_ workspace template.
