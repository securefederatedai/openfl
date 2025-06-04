# %%
# %%
import os

os.chdir(os.path.dirname(__file__))

from datasets import load_dataset
from src.dataset.img_classification import (
    prepare_data_for_image_classification,
)

# %%

dataset_name = "cifar10"
dataset = load_dataset(dataset_name)

label_feature = "label"
image_feature = "img"
dataset_dicts, number_of_labels = prepare_data_for_image_classification(
    dataset,
    collaborator_count=5,
    non_iid=True,
    image_feature=image_feature,
    label_feature=label_feature,
    debug_size=True,
)
dataset_dicts[0]["train"][0]["image"]
# %%
dataset_name = "zh-plus/tiny-imagenet"
dataset = load_dataset(dataset_name)

label_feature = "label"
image_feature = "image"
dataset_dicts, number_of_labels = prepare_data_for_image_classification(
    dataset["train"],
    collaborator_count=5,
    non_iid=True,
    image_feature=image_feature,
    label_feature=label_feature,
    debug_size=True,
)
dataset_dicts[0]["train"][0]["image"]
# %%
dataset_name = "Falah/Alzheimer_MRI"
dataset = load_dataset(dataset_name)

label_feature = "label"
image_feature = "image"

dataset_dicts, number_of_labels = prepare_data_for_image_classification(
    dataset,
    collaborator_count=5,
    non_iid=True,
    image_feature=image_feature,
    label_feature=label_feature,
)
dataset_dicts[0]["train"][0]["image"]
# %%
