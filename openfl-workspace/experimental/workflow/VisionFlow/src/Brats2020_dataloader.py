import glob
import multiprocessing as mp
import os
import random

import albumentations
import numpy as np
import torch
from datasets import Dataset, DatasetDict, Image

ORIGNINAL_SEGMENT_CLASSES = {
    0: "NOT tumor",
    1: "NECROTIC/CORE",  # label for non enhancing tumor
    2: "EDEMA",
    3: "ENHANCING",
}

SEGMENT_CLASSES = {
    0: "NOT tumor",
    1: "tumor",
}


IMAGE_TYPES = ["flair", "t1ce", "seg", "t2", "t1"]


def get_all_patients(path):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def get_images_for_dir(dir_path, im_type="flair"):
    outputs = sorted(glob.glob(os.path.join(dir_path, f"*{im_type}*.png")))
    return outputs


def process_patient(
    patient, dataset_path, image_types, x_slice_range, y_slice_range, z_slice_range, sample_size
):
    samples_per_patient = []
    patient_path = os.path.join(dataset_path, patient)
    outputs = get_images_for_dir(patient_path, image_types[0])

    for name in outputs:
        if "x" in name.split(image_types[0])[1]:
            x = int(name.split("_")[-1].split(".")[0])
            if not (x_slice_range[0] <= x < x_slice_range[1]):
                continue
        if "y" in name.split(image_types[0])[1]:
            y = int(name.split("_")[-1].split(".")[0])
            if not (y_slice_range[0] <= y < y_slice_range[1]):
                continue
        if "z" in name.split(image_types[0])[1]:
            z = int(name.split("_")[-1].split(".")[0])
            if not (z_slice_range[0] <= z < z_slice_range[1]):
                continue
        sample = [name]
        for im_type in image_types[1:]:
            if os.path.isfile(name.replace(image_types[0], im_type)):
                sample.append(name.replace(image_types[0], im_type))
        if len(sample) == sample_size:
            samples_per_patient.append(sample)
        else:
            # print(name, "is missing images")
            pass
    return samples_per_patient


def get_images_per_patient(
    dataset_path="Processed_TrainingData/",
    image_types=IMAGE_TYPES,
    x_slice_range=(0, 200),
    y_slice_range=(0, 0),
    z_slice_range=(0, 0),
):
    patient_list = get_all_patients(dataset_path)
    samples = []
    sample_size = len(image_types)

    with mp.Pool() as pool:
        samples = pool.starmap(
            process_patient,
            [
                (
                    patient,
                    dataset_path,
                    image_types,
                    x_slice_range,
                    y_slice_range,
                    z_slice_range,
                    sample_size,
                )
                for patient in patient_list
            ],
        )
    samples = [patient for patient in samples if len(patient) > 0]
    return samples


def create_dataset_dict(
    dataset_path,
    test_ratio=0.1,
    seed=42,
    number_of_patients_per_collaborator=None,
    collaborator_count=1,
):
    images_per_patient = get_images_per_patient(dataset_path=dataset_path, image_types=IMAGE_TYPES)
    random.seed(seed)
    random.shuffle(images_per_patient)
    test_size = int(len(images_per_patient) * test_ratio)

    global_test_patients = images_per_patient[:test_size]

    test_patients = images_per_patient[test_size : test_size + test_size]

    train_patients = images_per_patient[test_size + test_size :]
    train_patients = random.choices(
        train_patients, k=int(number_of_patients_per_collaborator) * collaborator_count
    )

    collaborator_train_split = [
        flatten_samples(train_patients[i::collaborator_count]) for i in range(collaborator_count)
    ]
    collaborator_test_split = [
        flatten_samples(test_patients[i::collaborator_count]) for i in range(collaborator_count)
    ]

    data_dict = []
    for train_samples, test_samples in zip(collaborator_train_split, collaborator_test_split):
        data_dict.append(
            DatasetDict(
                {
                    "train": create_dataset(train_samples),
                    "test": create_dataset(test_samples),
                }
            )
        )

    apply_transforms(data_dict)

    global_test_patients = create_dataset(flatten_samples(global_test_patients))
    for key in IMAGE_TYPES:
        global_test_patients = global_test_patients.cast_column(key, Image())
    global_test_patients.set_transform(transforms)

    return data_dict, global_test_patients


def apply_transforms(data_dict, image_types=IMAGE_TYPES):
    for col_dataset in data_dict:
        for split in col_dataset.keys():
            for key in image_types:
                col_dataset[split] = col_dataset[split].cast_column(key, Image())
            col_dataset[split].set_transform(transforms)
            col_dataset[split] = col_dataset[split].shuffle()


def flatten_samples(samples):
    return [i for patients in samples for i in patients]


def create_dataset(samples):
    return Dataset.from_dict(
        {key: [sample[n] for sample in samples] for n, key in enumerate(IMAGE_TYPES)}
    )


ADE_MEAN = np.array([128, 128, 128]) / 255
ADE_STD = np.array([128, 128, 128]) / 255

ADE_MEAN = (0.485, 0.456, 0.406)
ADE_STD = (0.229, 0.224, 0.225)


def transforms(examples):
    transform = albumentations.Compose(
        [
            albumentations.Resize(448 // 2, 448 // 2),
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
    transformed_images, transformed_masks = [], []

    for image, seg_mask in zip(examples["flair"], examples["seg"]):
        image, seg_mask = np.array(image), np.array(seg_mask, np.int32)
        max_value = np.percentile(image, 95)
        min_value = np.percentile(image, 5)
        image = np.where(image <= max_value, image, max_value)
        image = np.where(image <= min_value, 0.0, image)
        div = (max_value - min_value) if max_value != min_value else 1.0
        image = (image - min_value) / div * 255

        transformed = transform(image=image, mask=seg_mask)

        transformed_images.append(transformed["image"])
        # transformed_masks.append(transformed["mask"].long())
        transformed["mask"][transformed["mask"] != 0] = 1
        transformed_masks.append(
            torch.nn.functional.one_hot(
                transformed["mask"].long(), num_classes=len(SEGMENT_CLASSES)
            ).permute(2, 0, 1)
        )

    examples["pixel_values"] = transformed_images
    examples["labels"] = transformed_masks
    return examples




