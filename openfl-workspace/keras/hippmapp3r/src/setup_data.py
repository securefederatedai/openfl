# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""You may copy this file as the starting point of your own model."""
import os
import numpy as np
from tqdm import tqdm
import argparse


def create_data(collaborators, dst, total_dataset_size_per_col):
    num_samples = 1
    input_shape = (160, 160, 128)
    X_train = np.random.random((num_samples,) + input_shape)
    y_train = np.random.random((num_samples,) + input_shape)

    # Save the arrays
    os.makedirs(dst + "/1", exist_ok=True)
    np.save(dst + "/1/X_train.npy", X_train)
    np.save(dst + "/1/y_train.npy", y_train)
    single_file_size_MB = os.path.getsize(dst + "/1/X_train.npy") / 1024 / 1024
    single_file_size_MB += os.path.getsize(dst + "/1/y_train.npy") / 1024 / 1024
    required_file_size_MB = total_dataset_size_per_col * collaborators
    files_to_make = required_file_size_MB // single_file_size_MB
    files_per_collaborator = int(files_to_make // collaborators)
    for col in range(1, collaborators + 1):
        print("making files for collaborator", col)
        os.makedirs(dst + f"/{col}", exist_ok=True)
        for i in tqdm(range(files_per_collaborator)):
            X_train = np.random.random((num_samples,) + input_shape)
            y_train = np.random.random((num_samples,) + input_shape)
            np.save(dst + f"/{col}/X_train_{i}.npy", X_train)
            np.save(dst + f"/{col}/y_train_{i}.npy", y_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_collaborators", type=int, help="Number of collaborators")
    parser.add_argument(
        "--total_dataset_size_per_col_MB", type=int, help="Total dataset size per collaborator"
    )
    args = parser.parse_args()

    num_collaborators = args.num_collaborators
    total_dataset_size_per_col = args.total_dataset_size_per_col_MB

    dst = "data"
    create_data(num_collaborators, dst, total_dataset_size_per_col)
