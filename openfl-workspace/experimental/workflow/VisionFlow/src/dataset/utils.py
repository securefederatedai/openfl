from datasets import Dataset, DatasetDict, concatenate_datasets
from typing import Dict, List, Any, Callable, Union, Optional
import numpy as np


def split_dataset_dict(
    datasetdict: DatasetDict,
    collaborator_count: int,
    non_iid: bool = False,
    label_column: str = "label",
    seed: Optional[int] = None,
) -> List[DatasetDict]:
    """
    Splits a dataset dictionary into multiple smaller dictionaries for collaborators.

    Args:
        datasetdict (DatasetDict): A DatasetDict to be split.
        collaborator_count (int): Number of collaborators to split the data for.
        non_iid (bool): Whether to split data in a non-IID manner.
        label_column (str): The column name for labels.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        List[DatasetDict]: A list of dataset dictionaries, one for each collaborator.
    """
    if seed is not None:
        np.random.seed(seed)
    if collaborator_count <= 0:
        raise ValueError("collaborator_count must be a positive integer.")

    data_dict = {}
    for key in datasetdict.keys():
        data = datasetdict[key]
        if non_iid:
            # Non-IID splitting using Dirichlet partitioning
            splits = partition_data_dirichlet(collaborator_count, label_column, data, seed=seed)
        else:
            # IID splitting
            splits = partition_dataset_iid(collaborator_count, data, seed=seed)
        data_dict[key] = splits

    # Create a dataset dictionary for each collaborator
    collaborator_datasets = build_collaborator_dataset_dicts(collaborator_count, data_dict)

    return collaborator_datasets


def partition_data_dirichlet(
    collaborator_count: int,
    label_column: str,
    data: Dataset,
    seed: Optional[int] = 1000,
) -> List[Dataset]:
    """
    Partition dataset in a non-IID fashion using Dirichlet distribution.

    Args:
        collaborator_count (int): Number of collaborators.
        label_column (str): The column name for labels.
        data (Dataset): The dataset to partition.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        List[Dataset]: List of datasets, one per collaborator.
    """
    if seed is not None:
        np.random.seed(seed)
    label_groups = group_by_label(data, label_column)  # Group data by labels
    label_counts = [len(group) for group in label_groups]
    proportions = dirichlet_partition(label_counts, collaborator_count, seed=seed)
    splits = [[] for _ in range(collaborator_count)]
    for label_group, proportion in zip(label_groups, proportions):
        for i, fraction in enumerate(proportion):
            split_size = int(fraction * len(label_group))
            splits[i].append(Dataset.from_dict(label_group[:split_size]))
            label_group = Dataset.from_dict(label_group[split_size:])
            # Concatenate the splits for each collaborator
    splits = [concatenate_datasets(split) for split in splits]
    return splits


def dirichlet_partition(
    label_counts: List[int],
    collaborator_count: int,
    alpha: float = 0.5,
    seed: Optional[int] = 1000,
) -> List[np.ndarray]:
    """
    Generates Dirichlet proportions for non-IID partitioning.

    Args:
        label_counts (List[int]): List of counts for each label group.
        collaborator_count (int): Number of collaborators.
        alpha (float): Dirichlet concentration parameter.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        List[np.ndarray]: A list of proportions for each collaborator.
    """
    if seed is not None:
        np.random.seed(seed)
    proportions = []
    for count in label_counts:
        dirichlet_sample = np.random.dirichlet([alpha] * collaborator_count)
        proportions.append(dirichlet_sample)
    return proportions


def partition_dataset_iid(
    collaborator_count: int,
    data: Dataset,
    seed: Optional[int] = 1000,
) -> List[Dataset]:
    """
    Partition dataset in an IID fashion among collaborators.

    Args:
        collaborator_count (int): Number of collaborators.
        data (Dataset): The dataset to partition.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        List[Dataset]: List of datasets, one per collaborator.
    """
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    splits = []
    for i in range(collaborator_count):
        split_indices = indices[i:len(indices):collaborator_count]
        splits.append(data.select(split_indices))
    return splits


def build_collaborator_dataset_dicts(
    collaborator_count: int,
    data_dict: Dict[str, List[Dataset]],
) -> List[DatasetDict]:
    """
    Build a list of DatasetDicts, one for each collaborator, from split data.

    Args:
        collaborator_count (int): Number of collaborators.
        data_dict (Dict[str, List[Dataset]]): Dictionary mapping split names to lists of datasets.

    Returns:
        List[DatasetDict]: List of DatasetDicts, one per collaborator.
    """
    collaborator_datasets = []
    for i in range(collaborator_count):
        collaborator_datasets.append(
            DatasetDict({key: data_dict[key][i] for key in data_dict.keys()})
        )

    return collaborator_datasets


def group_by_label(
    data: Dataset,
    label_column: str,
) -> List[Dataset]:
    """
    Groups data by labels for non-IID splitting using Dataset's API.

    Args:
        data (Dataset): The dataset to group.
        label_column (str): The column name for labels.

    Returns:
        List[Dataset]: A list of grouped data based on labels.
    """
    data = data.to_pandas()  # Convert to pandas DataFrame for easier manipulation
    grouped = data.groupby(label_column)
    return [Dataset.from_pandas(group) for _, group in grouped]


def apply_transforms(
    data: Union[List[DatasetDict[str, Dataset]], DatasetDict[str, Dataset], Dataset],
    transform_fn: Callable[[Dataset, Any], Dataset],
    **kwargs,
) -> Union[List[DatasetDict[str, Dataset]], DatasetDict[str, Dataset], Dataset]:
    """
    Apply a transformation function to a dataset or collection of datasets.

    Args:
        data (List[DatasetDict[str, Dataset]] | DatasetDict[str, Dataset] | Dataset): The data to transform.
        transform_fn (Callable): The transformation function to apply.
        **kwargs: Additional keyword arguments for the transformation function.

    Returns:
        Same type as input data, with transformations applied.
    """
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
        process_dataset(data)
    else:
        raise TypeError(
            "Unsupported data type. Expected List[DatasetDict[str, Dataset]], DatasetDict[str, Dataset], or Dataset."
        )
