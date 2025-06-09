# %%
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator


def load_and_plot_tensorboard_logs(log_dir):
    all_metrics = {}
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            newnes = 0
            if file.startswith("events.out.tfevents"):
                file_path = os.path.join(root, file)
                file_newness = os.path.getmtime(file_path)
                if file_newness > newnes:
                    newnes = file_newness
                    parent_dir = os.path.basename(root)
                    all_metrics[f"{parent_dir}"] = extract_metrics(file_path)
                    print(file_path)
    return all_metrics


# Specify the directory containing TensorBoard logs
log_directory = '/home/omar/Documents/mine/INTEL/tensorboard/tensorboard'


def extract_metrics(single_dir):
    event_acc = EventAccumulator(single_dir)
    event_acc.Reload()
    metrics = {}
    for tag in event_acc.Tags()["scalars"]:
        if tag not in metrics:
            metrics[tag] = []
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        metrics[tag].append((steps, values))

    return metrics


# Load and plot the metrics
all_metrics = load_and_plot_tensorboard_logs(log_directory)

# %%
metrics_by_split = []
for model_name, value in all_metrics.items():
    split = model_name.split(":")[-1]
    row = {"split": split, "model_name": model_name}
    row.update(value)
    metrics_by_split.append(row)
# %%


df = pd.DataFrame(metrics_by_split)
# %%

for metric in ["losses/train_dict"]:
    per_model_metric = defaultdict(dict)
    for index, row in df.iterrows():
        per_model_metric[row["model_name"]] = row[metric][0][1][-1] if not np.any(np.isnan(row[metric])) else 0.28)
    plt.figure(figsize=(10, 5))
    for model, values in per_model_metric.items():
        plt.plot(
            sorted(values),
            [values[key] for key in sorted(values)],
            label=model,
            marker="o" if "dino2" in model else None,
        )

    plt.title(f"{metric}")
    plt.xlabel("split")
    plt.ylabel(metric)
    plt.legend()
# %%
