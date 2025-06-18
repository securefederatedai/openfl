from typing import List

import albumentations
import numpy as np
import torch
from src.dataset.utils import apply_transforms, split_dataset_dict

from datasets import Dataset, DatasetDict, Image
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load CIFAR-10 dataset
ADE_MEAN = (0.485, 0.456, 0.406)
ADE_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_FEATURE = "image"
DEFAULT_PROCESSED_IMAGE_FEATURE = "pixel_values"
DEFAULT_LABEL_FEATURE = "labels"


def image_classification_collate_fn(batch):
    """
    Custom collate function for image classification tasks.
    It stacks pixel values and labels from the batch.
    """
    pixel_values = torch.stack([item[DEFAULT_PROCESSED_IMAGE_FEATURE] for item in batch], dim=0)
    labels = torch.stack([item[DEFAULT_LABEL_FEATURE] for item in batch], dim=0)
    return {DEFAULT_PROCESSED_IMAGE_FEATURE: pixel_values, DEFAULT_LABEL_FEATURE: labels}


def prepare_data_for_image_classification(
    data: DatasetDict[str, Dataset] | Dataset,
    collaborator_count=None,
    non_iid=False,
    image_feature=DEFAULT_IMAGE_FEATURE,
    label_feature=DEFAULT_LABEL_FEATURE,
    number_of_labels=None,
    image_size: int = 448,
    debug_size=False,
    percentage: float = 1.0,
):
    if isinstance(data, Dataset):
        data = DatasetDict({"train": data})

    if number_of_labels is None:
        number_of_labels = extract_number_of_labels(data, label_feature)

    if collaborator_count is not None:
        assert collaborator_count > 0, "collaborator_count must be greater than 0"
        dataset_dict_list = split_dataset_dict(
            data, collaborator_count=collaborator_count, non_iid=non_iid
        )
        if debug_size:
            # Limit dataset size for debugging purposes
            for dataset_dict in dataset_dict_list:
                for key in dataset_dict.keys():
                    dataset_dict[key] = dataset_dict[key].select(range(10))
    else:
        dataset_dict_list = data
        if debug_size:
            # Limit dataset size for debugging purposes
            for key in dataset_dict.keys():
                dataset_dict[key] = dataset_dict[key].select(range(10))

    if percentage < 1.0:
        # Limit dataset size based on the percentage
        for dataset_dict in dataset_dict_list:
            for key in dataset_dict.keys():
                dataset_dict[key] = (
                    dataset_dict[key]
                    .shuffle()
                    .select(range(max(int(len(dataset_dict[key]) * percentage), 1)))
                )

    apply_classification_transforms(
        dataset_dict_list,
        image_feature=image_feature,
        label_feature=label_feature,
        number_of_labels=number_of_labels,
        image_size=image_size,  # Assuming full size images are preferred
    )
    return dataset_dict_list, number_of_labels


def extract_number_of_labels(data, label_feature):
    try:
        if isinstance(data, DatasetDict):  # Handle DatasetDict[str, Dataset]
            number_of_labels = 0
            for dataset in data.values():
                partial_number_of_labels = max(
                    number_of_labels, _get_number_of_labels(dataset, label_feature)
                )
                if partial_number_of_labels > number_of_labels:
                    number_of_labels = partial_number_of_labels
        elif isinstance(data, Dataset):  # Handle single Dataset
            dataset = data
            number_of_labels = _get_number_of_labels(dataset, label_feature)
        else:
            raise TypeError(
                "Unsupported data type. Expected DatasetDict[str, Dataset], or Dataset."
            )
        logger.info(f"Automatically detected number of labels: {number_of_labels}")
    except Exception as e:
        raise RuntimeError(
            f"Error in getting number of labels: you can manualy specify number_of_labels {e} "
        )
    return number_of_labels


def _get_number_of_labels(dataset, label_feature):
    split_max = max(dataset[label_feature]) + 1
    return split_max


def default_classification_transforms(examples, number_of_labels=2, image_size=448):
    transform = albumentations.Compose(
        [
            albumentations.Resize(image_size, image_size),
            albumentations.ToRGB(),
            albumentations.HorizontalFlip(p=0.25),
            albumentations.VerticalFlip(p=0.25),
            albumentations.Rotate(p=0.25, limit=(-90, 90)),
            albumentations.Normalize(
                mean=ADE_MEAN,
                std=ADE_STD,
            ),
            albumentations.ToTensorV2(),
        ]
    )
    transformed_images, transformed_labels = [], []

    for image, label in zip(examples[DEFAULT_IMAGE_FEATURE], examples[DEFAULT_LABEL_FEATURE]):
        image = np.array(image)
        if image.shape[-1] == 4:
            image = image[:, :, :3]  # Ensure image is in HWC format and RGB
        transformed = transform(image=image)
        transformed_images.append(transformed[DEFAULT_IMAGE_FEATURE])
        transformed_labels.append(
            torch.nn.functional.one_hot(torch.tensor(label), num_classes=number_of_labels).float()
        )

    examples["pixel_values"] = transformed_images
    examples[DEFAULT_LABEL_FEATURE] = transformed_labels
    return examples


def apply_classification_transforms(
    data: List[DatasetDict[str, Dataset]] | DatasetDict[str, Dataset] | Dataset,
    use_shuffle=True,
    image_feature: str = "image",
    label_feature: str = "labels",
    image_size: int = 448,
    number_of_labels: int = 2,
):
    apply_transforms(
        data,
        _apply_classification_transforms,
        use_shuffle=use_shuffle,
        image_feature=image_feature,
        label_feature=label_feature,
        image_size=image_size,
        number_of_labels=number_of_labels,
    )


def _apply_classification_transforms(
    dataset: Dataset,
    use_shuffle=True,
    image_feature: str = "image",
    label_feature: str = "labels",
    image_size: int = 448,
    number_of_labels: int = 2,
):
    if image_feature not in dataset.features:
        raise ValueError(f"Image feature '{image_feature}' not found in dataset.")

    if label_feature not in dataset.features:
        raise ValueError(f"Label feature '{label_feature}' not found in dataset.")

    if image_feature != DEFAULT_IMAGE_FEATURE:
        dataset = dataset.rename_column(image_feature, DEFAULT_IMAGE_FEATURE)
    dataset = dataset.cast_column(DEFAULT_IMAGE_FEATURE, Image())

    if label_feature != DEFAULT_LABEL_FEATURE:
        dataset = dataset.rename_column(label_feature, DEFAULT_LABEL_FEATURE)

    def transform(examples):
        return default_classification_transforms(
            examples,
            number_of_labels=number_of_labels,
            image_size=image_size,
        )

    dataset.set_transform(transform)
    if use_shuffle:
        dataset = dataset.shuffle()
    return dataset


# %%
