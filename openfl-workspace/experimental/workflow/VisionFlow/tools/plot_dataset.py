# %%
import os

os.chdir(os.path.dirname(__file__) + "/..")
import sys
sys.path.append(os.getcwd())
from datasets import load_dataset
from src.dataset.img_classification import prepare_data_for_image_classification

non_iid = True
debug_size = False
percentage = 0.8
collaborator_names = ["Portland", "Seattle", "Chandler", "Phoenix"]

dataset_name = "Falah/Alzheimer_MRI"
dataset = load_dataset(dataset_name)
global_validation_dataset = dataset["test"]
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

label_feature = "label"
image_feature = "image"
image_size = 224

dataset_dicts, number_of_labels = prepare_data_for_image_classification(
    dataset,
    collaborator_count=len(collaborator_names),
    non_iid=non_iid,
    image_feature=image_feature,
    label_feature=label_feature,
    image_size=image_size,
    debug_size=debug_size,
    percentage=percentage,
)
global_validation_dataset, _ = prepare_data_for_image_classification(
    global_validation_dataset,
    collaborator_count=None,
    image_feature=image_feature,
    label_feature=label_feature,
    image_size=image_size,
    debug_size=debug_size,
)
# %%
from collections import defaultdict
import numpy as np

plot_dict = defaultdict(list)
for dataset in dataset_dicts:
    for split, ds in dataset.items():
        labels = [i["labels"] for i in ds]
        unique_labels, counts = np.unique(labels, return_counts=True, axis=0)
        # Fill missing labels with 0 counts
        full_counts = np.zeros(number_of_labels, dtype=int)
        full_counts[np.argmax(unique_labels, axis=1)] = counts
        plot_dict[split].append((np.arange(number_of_labels), full_counts))
# %%
import matplotlib.pyplot as plt

splits = list(plot_dict.keys())
num_collaborators = len(collaborator_names)
num_classes = number_of_labels

for split in splits:
    plt.figure(figsize=(10, 6))
    width = 0.75  # width of each bar
    x = np.arange(num_classes)  # the label locations

    for idx, (unique_labels, counts) in enumerate(plot_dict[split]):
        plt.bar(
            idx * num_classes + x,
            counts,
            width=width,
            alpha=0.7,
            label=collaborator_names[idx],
        )
    # Set x-ticks only once per plot, after all bars are drawn
    if idx == num_collaborators - 1:
        tick_positions = []
        tick_labels = []
        for c_idx in range(num_collaborators):
            tick_positions.extend(c_idx * num_classes + x)
            tick_labels.extend([f"Class {i}\n{collaborator_names[c_idx]}" for i in x])
        plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
    plt.title(f"Class Distribution per Collaborator - {split.capitalize()} Split")
    plt.xlabel("Class (Collaborator)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()
# %%
from collections import defaultdict
import numpy as np

global_plot_dict = defaultdict(list)
for split, ds in global_validation_dataset.items():
    labels = [i["labels"] for i in ds]
    unique_labels, counts = np.unique(labels, return_counts=True, axis=0)
    # Fill missing labels with 0 counts
    full_counts = np.zeros(number_of_labels, dtype=int)
    full_counts[np.argmax(unique_labels, axis=1)] = counts
    global_plot_dict[split].append((np.arange(number_of_labels), full_counts))
# %%
import matplotlib.pyplot as plt

splits = list(global_plot_dict.keys())
num_classes = number_of_labels

for split in splits:
    plt.figure(figsize=(10, 6))
    width = 0.75  # width of each bar
    x = np.arange(num_classes)  # the label locations

    for idx, (unique_labels, counts) in enumerate(global_plot_dict[split]):
        plt.bar(
            idx * num_classes + x,
            counts,
            width=width,
            alpha=0.7,
            label='Global Validation',
        )
    # Set x-ticks only once per plot, after all bars are drawn
    if idx == 0:
        tick_positions = []
        tick_labels = []
        for c_idx in range(1):
            tick_positions.extend(c_idx * num_classes + x)
            tick_labels.extend([f"Class {i}" for i in x])
        plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
    plt.title(f"Class Distribution per Collaborator - {split.capitalize()} Split")
    plt.xlabel("Class (Collaborator)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()
# %%
