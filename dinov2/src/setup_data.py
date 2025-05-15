# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm


def save_slices(image, output_path, dir_name, prefix, axis, slice_range):
    axes = {"x": 0, "y": 1, "z": 2}
    axis_index = axes[axis]
    slices = np.take(
        image,
        range(
            image.shape[axis_index] // 2 - slice_range // 2,
            image.shape[axis_index] // 2 + slice_range // 2,
        ),
        axis=axis_index,
    )

    for i in range(slices.shape[axis_index]):
        slice_data = np.take(slices, i, axis=axis_index)
        slice_image = Image.fromarray(slice_data)
        slice_image = slice_image.convert("I")
        slice_image.save(
            os.path.join(output_path, f"{dir_name}_{prefix}_{axis}_{str(i).zfill(3)}.png"),
        )


def process_directory(args):
    dir_path, output_path, slice_range_x, slice_range_y, slice_range_z = args
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    slice_list = [slice_range_x, slice_range_y, slice_range_z]

    for name in ["flair", "t1ce", "seg", "t2", "t1"]:
        file_path = os.path.join(dir_path, os.path.basename(dir_path) + f"_{name}.nii")

        if os.path.exists(file_path):
            img = nib.load(file_path).get_fdata()

            for n, dim in enumerate(["x", "y", "z"]):
                if slice_list[n] > 0:
                    save_slices(
                        img, output_path, os.path.basename(dir_path), name, dim, slice_list[n]
                    )


def save_middle_slices(
    dataset_path, output_path, slice_range_x=80, slice_range_y=80, slice_range_z=80
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    directories = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
    args = [
        (
            dir_path,
            os.path.join(output_path, os.path.basename(dir_path)),
            slice_range_x,
            slice_range_y,
            slice_range_z,
        )
        for dir_path in directories
    ]
    with Pool() as pool:
        list(tqdm(pool.imap(process_directory, args), total=len(directories)))


def main():
    parser = argparse.ArgumentParser(description="Process MRI slices.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--slice_range_x", type=int, default=240, help="Slice range for x-axis")
    parser.add_argument("--slice_range_y", type=int, default=240, help="Slice range for y-axis")
    parser.add_argument("--slice_range_z", type=int, default=155, help="Slice range for z-axis")
    args = parser.parse_args()

    TRAIN_DATASET_PATH = os.path.join(
        args.dataset_path, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData"
    )
    OUTPUT_TRAIN_PATH = os.path.join(args.dataset_path, "Processed_TrainingData")
    if os.path.exists(OUTPUT_TRAIN_PATH):
        os.system(f"rm -rf {OUTPUT_TRAIN_PATH}")

    print("dataset_path:", args.dataset_path, "output_train_path:", OUTPUT_TRAIN_PATH)

    save_middle_slices(
        TRAIN_DATASET_PATH,
        OUTPUT_TRAIN_PATH,
        slice_range_x=args.slice_range_x,
        slice_range_y=args.slice_range_y,
        slice_range_z=args.slice_range_z,
    )


if __name__ == "__main__":
    main()
