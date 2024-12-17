# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision
import numpy as np
import pathlib
import os
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image as Image
import imagen as ig
import numbergen as ng


watermark_dir = "./files/watermark-dataset/MWAFFLE/"


def generate_watermark(
    x_size=28, y_size=28, num_class=10, num_samples_per_class=10, img_dir=watermark_dir
):
    """
    Generate Watermark by superimposing a pattern on noisy background.

    Parameters
    ----------
    x_size: x dimension of the image
    y_size: y dimension of the image
    num_class: number of classes in the original dataset
    num_samples_per_class: number of samples to be generated per class
    img_dir: directory for saving watermark dataset

    Reference
    ---------
    WAFFLE: Watermarking in Federated Learning (https://arxiv.org/abs/2008.07298)

    """
    x_pattern = int(x_size * 2 / 3.0 - 1)
    y_pattern = int(y_size * 2 / 3.0 - 1)

    np.random.seed(0)
    for cls in range(num_class):
        patterns = []
        random_seed = 10 + cls
        patterns.append(
            ig.Line(
                xdensity=x_pattern,
                ydensity=y_pattern,
                thickness=0.001,
                orientation=np.pi * ng.UniformRandom(seed=random_seed),
                x=ng.UniformRandom(seed=random_seed) - 0.5,
                y=ng.UniformRandom(seed=random_seed) - 0.5,
                scale=0.8,
            )
        )
        patterns.append(
            ig.Arc(
                xdensity=x_pattern,
                ydensity=y_pattern,
                thickness=0.001,
                orientation=np.pi * ng.UniformRandom(seed=random_seed),
                x=ng.UniformRandom(seed=random_seed) - 0.5,
                y=ng.UniformRandom(seed=random_seed) - 0.5,
                size=0.33,
            )
        )

        pat = np.zeros((x_pattern, y_pattern))
        for i in range(6):
            j = np.random.randint(len(patterns))
            pat += patterns[j]()
        res = pat > 0.5
        pat = res.astype(int)

        x_offset = np.random.randint(x_size - x_pattern + 1)
        y_offset = np.random.randint(y_size - y_pattern + 1)

        for i in range(num_samples_per_class):
            base = np.random.rand(x_size, y_size)
            base[
                x_offset: x_offset + pat.shape[0],
                y_offset: y_offset + pat.shape[1],
            ] += pat
            d = np.ones((x_size, x_size))
            img = np.minimum(base, d)
            if not os.path.exists(img_dir + str(cls) + "/"):
                os.makedirs(img_dir + str(cls) + "/")
            plt.imsave(
                img_dir + str(cls) + "/wm_" + str(i + 1) + ".png",
                img,
                cmap=matplotlib.cm.gray,
            )


# If the Watermark dataset does not exist, generate and save the Watermark images
watermark_path = pathlib.Path(watermark_dir)
if watermark_path.exists() and watermark_path.is_dir():
    print(
        f"Watermark dataset already exists at: {watermark_path}. Proceeding to next step ... "
    )
    pass
else:
    print("Generating Watermark dataset... ")
    generate_watermark()


class WatermarkDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, label_dir=None, transforms=None):
        self.images_dir = os.path.abspath(images_dir)
        self.image_paths = [
            os.path.join(self.images_dir, d) for d in os.listdir(self.images_dir)
        ]
        self.label_paths = label_dir
        self.transform = transforms
        temp = []

        # Recursively counting total number of images in the directory
        for image_path in self.image_paths:
            for path in os.walk(image_path):
                if len(path) <= 1:
                    continue
                path = path[2]
                for im_n in [image_path + "/" + p for p in path]:
                    temp.append(im_n)
        self.image_paths = temp

        if len(self.image_paths) == 0:
            raise Exception(f"No file(s) found under {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = image.convert("RGB")
        image = self.transform(image)
        label = int(image_filepath.split("/")[-2])

        return image, label


def get_watermark_transforms():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize(28),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize
        ]
    )


watermark_data = WatermarkDataset(
    images_dir=watermark_dir,
    transforms=get_watermark_transforms(),
)


def aggregator_attrs(watermark_data, batch_size):
    return {
        "watermark_data_loader": torch.utils.data.DataLoader(
            watermark_data, batch_size=batch_size, shuffle=True
        ),
        "pretrain_epochs": 25,
        "retrain_epochs": 25,
        "watermark_acc_threshold": 0.98,
        "watermark_pretraining_completed": False,
    }
