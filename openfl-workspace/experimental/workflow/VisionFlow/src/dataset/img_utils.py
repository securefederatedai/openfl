"""
Image dataset utility functions for data preparation and splitting.

This module provides helper functions for preparing image datasets,
including splitting datasets for federated learning scenarios,
debugging with smaller dataset sizes, and subsetting datasets by percentage.
"""

from src.dataset.utils import split_dataset_dict

from datasets import Dataset, DatasetDict
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)

ADE_MEAN = (0.485, 0.456, 0.406)  # Mean for ADE20K dataset normalization
ADE_STD = (0.229, 0.224, 0.225)  # Std for ADE20K dataset normalization

DEFAULT_IMAGE_FEATURE = "image"  # Default image feature key
DEFAULT_PROCESSED_IMAGE_FEATURE = "pixel_values"  # Default processed image feature key


def prepare_image_data(
    data: Union[DatasetDict, Dataset],
    collaborator_count: Optional[int] = None,
    non_iid: bool = False,
    debug_size: bool = False,
    percentage: float = 1.0,
    seed: Optional[int] = 1000,
) -> List[DatasetDict]:
    """
    Prepare image data for training or federated learning.

    Args:
        data (DatasetDict | Dataset): The input Dataset or DatasetDict.
        collaborator_count (Optional[int], optional): Number of collaborators to split the data for.
            If None, no splitting is performed. Defaults to None.
        non_iid (bool, optional): Whether to split data in a non-IID fashion. Defaults to False.
        debug_size (bool, optional): If True, limit each dataset split to 10 samples for debugging.
            Defaults to False.
        percentage (float, optional): Fraction of the dataset to use (between 0 and 1).
            Defaults to 1.0.
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to 1000.

    Returns:
        List[DatasetDict]: List of dataset dictionaries, one for each collaborator,
            or a single-item list if not splitting.

    Raises:
        ValueError: If percentage is not in (0, 1].
    """
    if not (0 < percentage <= 1.0):
        raise ValueError("percentage must be between 0 (exclusive) and 1.0 (inclusive)")

    if isinstance(data, Dataset):
        data = DatasetDict({"train": data})

    if collaborator_count is not None:
        if collaborator_count <= 0:
            raise ValueError("collaborator_count must be greater than 0")
        dataset_dict_list = split_dataset_dict(
            data, collaborator_count=collaborator_count, non_iid=non_iid, seed=seed
        )
        if debug_size:
            # Limit dataset size for debugging purposes
            for dataset_dict in dataset_dict_list:
                for key in dataset_dict.keys():
                    dataset_dict[key] = dataset_dict[key].select(range(min(10, len(dataset_dict[key]))))
    else:
        dataset_dict_list = [data]
        if debug_size:
            # Limit dataset size for debugging purposes
            for key in data.keys():
                data[key] = data[key].select(range(min(10, len(data[key]))))

    if percentage < 1.0:
        # Limit dataset size based on the percentage
        for dataset_dict in dataset_dict_list:
            for key in dataset_dict.keys():
                dataset_dict[key] = (
                    dataset_dict[key]
                    .shuffle(seed=seed)
                    .select(range(max(int(len(dataset_dict[key]) * percentage), 1)))
                )
                if len(dataset_dict[key]) == 0:
                    logger.warning(f"After applying percentage, split '{key}' is empty.")

    if percentage < 1.0 and percentage * 10 < 1:
        logger.warning("Selected percentage may result in very small datasets.")

    logger.info(
        f"Prepared image data: "
        f"split into {collaborator_count if collaborator_count else 1} collaborator(s), "
        f"{'debug size' if debug_size else 'full size'}, "
        f"using {percentage*100:.1f}% of data."
    )

    return dataset_dict_list
