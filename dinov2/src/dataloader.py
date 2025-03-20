from datasets import Dataset, DatasetDict, Image
import os
import random
import glob
import albumentations
import numpy as np
import multiprocessing as mp
import torch

SEGMENT_CLASSES = {
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


def get_samples(
    dataset_path="PATH/Processed_TrainingData/",
    image_types=IMAGE_TYPES,
    # x_slice_range=(60, 180),
    # y_slice_range=(40, 200),
    x_slice_range=(0, 0),
    y_slice_range=(0, 0),
    z_slice_range=(37, 117),
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


def create_dataset_dict(test_ratio=0.1, seed=42, patient_percentage=None, collaborator_count=1):
    samples = get_samples(image_types=IMAGE_TYPES)
    random.seed(seed)
    random.shuffle(samples)
    test_size = int(len(samples) * test_ratio)

    def flatten_samples(samples):
        return [i for patients in samples for i in patients]

    g_test_samples = samples[:test_size]
    test_samples = samples[test_size : test_size + test_size]
    train_samples = samples[test_size + test_size :]

    if patient_percentage is not None:
        if isinstance(patient_percentage, float):
            train_samples = random.sample(
                train_samples, int(len(train_samples) * patient_percentage)
            )
        else:
            train_samples = random.choices(
                train_samples, k=int(patient_percentage) * collaborator_count
            )
    split_train_samples = [
        flatten_samples(train_samples[i::collaborator_count]) for i in range(collaborator_count)
    ]

    split_test_samples = [
        flatten_samples(test_samples[i::collaborator_count]) for i in range(collaborator_count)
    ]

    def create_dataset(samples):
        return Dataset.from_dict(
            {key: [sample[n] for sample in samples] for n, key in enumerate(IMAGE_TYPES)}
        )

    data_dict = [
        DatasetDict(
            {
                "train": create_dataset(tr_samples),
                "test": create_dataset(te_samples),
            }
        )
        for tr_samples, te_samples in zip(split_train_samples, split_test_samples)
    ]

    for col_dataset in data_dict:
        for split in ["train", "test"]:
            for key in IMAGE_TYPES:
                col_dataset[split] = col_dataset[split].cast_column(key, Image())
            col_dataset[split].set_transform(transforms)
            col_dataset[split] = col_dataset[split].shuffle()

    g_test_samples = create_dataset(flatten_samples(g_test_samples))
    for key in IMAGE_TYPES:
        g_test_samples = g_test_samples.cast_column(key, Image())
    g_test_samples.set_transform(transforms)

    return data_dict, g_test_samples


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
        image, seg_mask = np.array(image), np.array(seg_mask)
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


def collate_fn(inputs):
    pixel_values = torch.stack([i["pixel_values"] for i in inputs], dim=0)
    labels = torch.stack(
        [i["labels"] for i in inputs],
        dim=0,
    )
    return {"pixel_values": pixel_values, "labels": labels}
