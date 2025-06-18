# %%
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_and_plot_tensorboard_logs(log_dir):
    all_metrics = {}
    for root, dirs, files in os.walk(log_dir):
        print(f"Checking files: {files}")
        good_path = None
        newnes = 0
        for file in files:
            if file.startswith("events.out.tfevents"):
                file_path = os.path.join(root, file)
                file_newness = os.path.getmtime(file_path)
                if file_newness > newnes:
                    newnes = file_newness
                    good_path = file_path
        if good_path:
            parent_dir = os.path.basename(root)
            all_metrics[f"{parent_dir}"] = extract_metrics(file_path)
            print(f"Extracted metrics from {file_path}")
    return all_metrics


# Specify the directory containing TensorBoard logs
log_directory = "/home/omar/Documents/mine/INTEL/openfl/openfl-workspace/experimental/workflow/VisionFlow/output/federated_tensorboard/"


def extract_metrics(single_dir):
    event_acc = EventAccumulator(single_dir)
    event_acc.Reload()
    metrics = {}
    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        metrics[tag]  = (steps, values)

    return metrics


# Load and plot the metrics
all_metrics = load_and_plot_tensorboard_logs(log_directory)

# %%
metrics_by_split = []
for full_model_name, value in all_metrics.items():
    split = ".".join(full_model_name.split("_peft")[-1].split("_")[1:3])
    base_model_name = full_model_name.split(":")[0]
    non_iid = "non_iid" in full_model_name
    peft = "peft" in full_model_name
    classification = "classification" in full_model_name
    with_pretrained = "with_pretrained" in full_model_name
    head = full_model_name.split(":")[1].split("_")[0]
    row = {
        "full_model_name": full_model_name,
        "split": split,
        "model_name": base_model_name,
        "non_iid": non_iid,
        "head": head,
        "peft": peft,
        "classification": classification,
        "with_pretrained": with_pretrained,
    }
    row.update(value)
    metrics_by_split.append(row)
# %%
df = pd.DataFrame(metrics_by_split)
# %%
df.dropna(subset=["losses/train_dict"], inplace=True)
df.reset_index(drop=True, inplace=True)
# %%
df
# %%
classification_models = df[df["classification"] == True]
value_dict = defaultdict(dict)
for base_model_name in classification_models["model_name"].unique():
    for pre_training in [True, False]:
        for head in classification_models["head"].unique():
            if 'MLP' in head:
                continue
            partial_df = classification_models[
                (classification_models["model_name"] == base_model_name)
                & (classification_models["with_pretrained"] == pre_training)
                & (classification_models["head"] == head)
            ]
            for split in partial_df["split"].unique():
                value_dict[base_model_name + ("_pretrained" if pre_training else "") + "_" + head][
                    split
                ] = max(
                    partial_df[partial_df["split"] == split][
                        "global_eval_metrics/eval_train_accuracy"
                        # "agg_validation_dict/eval_accuracy"
                        # "local_validation_dict/eval_accuracy"
                    ].iloc()[0][1]
                )
# %%
plt.figure(figsize=(12, 6))
for base_model_name, values in value_dict.items():
    sorted_values = dict(sorted(values.items(), key=lambda item: item[0]))  # Sort by split
    plt.plot(
        list(sorted_values.keys()),
        list(sorted_values.values()),
        label=base_model_name,
        marker="o" if "dino" in base_model_name else None,
        linestyle="--" if "pretrained" in base_model_name else "-",
    )
plt.legend()
plt.grid()
# %%
# %%
classification_models = df[df["classification"] == True]
value_dict = defaultdict(dict)
for base_model_name in classification_models["model_name"].unique():
    if "dino" not in base_model_name:
        continue  # Skip DINO models for this analysis
    for pre_training in [True, False]:
        for head in classification_models["head"].unique():
            if 'MLP' in head:
                continue
            partial_df = classification_models[
                (classification_models["model_name"] == base_model_name)
                & (classification_models["with_pretrained"] == pre_training)
                & (classification_models["head"] == head)
            ]
            for split in partial_df["split"].unique():
                value_dict[base_model_name + ("_pretrained" if pre_training else "") + "_" + head][
                    split
                ] = max(
                    partial_df[partial_df["split"] == split][
                        "global_eval_metrics/eval_train_accuracy"
                        # "agg_validation_dict/eval_accuracy"
                        # "local_validation_dict/eval_accuracy"
                    ].iloc()[0][1]
                )
# %%
plt.figure(figsize=(12, 6))
for base_model_name, values in value_dict.items():
    sorted_values = dict(sorted(values.items(), key=lambda item: item[0]))  # Sort by split
    plt.plot(
        list(sorted_values.keys()),
        list(sorted_values.values()),
        label=base_model_name,
        marker="o" if "dino" in base_model_name else None,
        linestyle="--" if "pretrained" in base_model_name else "-",
    )
plt.legend()
plt.grid()
# %%
