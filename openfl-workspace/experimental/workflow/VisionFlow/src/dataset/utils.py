from datasets import Dataset, DatasetDict, concatenate_datasets
from typing import Dict, List
import numpy as np
np.random.seed(1000)

def split_dataset_dict(
    datasetdict: DatasetDict | Dict,
    collaborator_count: int,
    non_iid: bool = False,
    label_column: str = "label",
) -> list:
    """
    Splits a dataset dictionary into multiple smaller dictionaries for collaborators.

    Args:
        datasetdict (DatasetDict): A DatasetDict containing 'train' and 'test' datasets.
        collaborator_count (int): Number of collaborators to split the data for.
        non_iid (bool): Whether to split data in a non-IID manner.

    Returns:
        list: A list of dataset dictionaries, one for each collaborator.
    """
    keys = datasetdict.keys()
    if collaborator_count <= 0:
        raise ValueError("collaborator_count must be a positive integer.")

    data_dict = {}
    for key in keys:
        data = datasetdict[key]
        if non_iid:
            # Non-IID splitting using Dirichlet partitioning
            splits = partition_data_dirichlet(collaborator_count, label_column, data)
        else:
            # IID splitting
            splits = partition_dataset_iid(collaborator_count, data)
        data_dict[key] = splits

    # Create a dataset dictionary for each collaborator
    collaborator_datasets = build_collaborator_dataset_dicts(collaborator_count, keys, data_dict)

    return collaborator_datasets


def partition_data_dirichlet(collaborator_count, label_column, data):
    label_groups = group_by_label(data, label_column)  # Group data by labels
    label_counts = [len(group) for group in label_groups]
    proportions = dirichlet_partition(label_counts, collaborator_count)
    splits = [[] for _ in range(collaborator_count)]
    for label_group, proportion in zip(label_groups, proportions):
        for i, fraction in enumerate(proportion):
            split_size = int(fraction * len(label_group))
            splits[i].append(Dataset.from_dict(label_group[:split_size]))
            label_group = Dataset.from_dict(label_group[split_size:])
            # Concatenate the splits for each collaborator
    splits = [concatenate_datasets(split) for split in splits]
    return splits


def partition_dataset_iid(collaborator_count, data):
    splits = []
    for i in range(collaborator_count):
        splits.append(Dataset.from_dict(data[i : len(data) : collaborator_count]))
    return splits


def build_collaborator_dataset_dicts(collaborator_count, keys, data_dict):
    collaborator_datasets = []
    for i in range(collaborator_count):
        collaborator_datasets.append(DatasetDict({key: data_dict[key][i] for key in keys}))

    return collaborator_datasets


def group_by_label(data: Dataset, label_column):
    """
    Groups data by labels for non-IID splitting using Dataset's API.

    Args:
        data (Dataset): The dataset to group.

    Returns:
        list: A list of grouped data based on labels.
    """
    data = data.to_pandas()  # Convert to pandas DataFrame for easier manipulation
    grouped = data.groupby(label_column)
    return [Dataset.from_pandas(group) for _, group in grouped]


def dirichlet_partition(label_counts, collaborator_count, alpha=0.5):
    """
    Generates Dirichlet proportions for non-IID partitioning.

    Args:
        label_counts (list): List of counts for each label group.
        collaborator_count (int): Number of collaborators.
        alpha (float): Dirichlet concentration parameter.

    Returns:
        list: A list of proportions for each collaborator.
    """
    proportions = []
    for count in label_counts:
        dirichlet_sample = np.random.dirichlet([alpha] * collaborator_count)
        proportions.append(dirichlet_sample)
    return proportions


def apply_transforms(
    data: List[DatasetDict[str, Dataset]] | DatasetDict[str, Dataset] | Dataset,
    transform_fn,
    **kwargs,
):
    def process_dataset(dataset):
        return transform_fn(dataset, **kwargs)

    if isinstance(data, list):  # Handle List[DatasetDict[str, Dataset]]
        for dataset_dict in data:
            for key, split in dataset_dict.items():
                dataset_dict[key] = process_dataset(split)
    elif isinstance(data, DatasetDict):  # Handle DatasetDict[str, Dataset]
        for key, split in data.items():
            data[key] = process_dataset(split)
    elif isinstance(data, Dataset):  # Handle single Dataset
        return process_dataset(data)
    else:
        raise TypeError(
            "Unsupported data type. Expected List[DatasetDict[str, Dataset]], DatasetDict[str, Dataset], or Dataset."
        )
