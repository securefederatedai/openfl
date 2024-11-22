# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import shutil
import torch
import torchvision.transforms as T

from pathlib import Path
from copy import deepcopy
from PIL import Image
from torch.utils.data import Dataset, random_split


common_data_folder = os.path.join(os.getcwd(), 'data')
zip_file_path = os.path.join(common_data_folder, 'tiny-imagenet-200.zip')
os.makedirs(common_data_folder, exist_ok=True)
os.system(f'wget --no-clobber http://cs231n.stanford.edu/tiny-imagenet-200.zip'
          f' -O {zip_file_path}')
print('Unpacking tiny-imagenet-200.zip')
shutil.unpack_archive(str(zip_file_path), str(common_data_folder))

normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

augmentation = T.RandomApply(
    [T.RandomHorizontalFlip(),
     T.RandomRotation(10),
     T.RandomResizedCrop(64)],
    p=.8
)

training_transform = T.Compose(
    [T.Lambda(lambda x: x.convert('RGB')),
     T.ToTensor(),
     augmentation,
     normalize]
)

valid_transform = T.Compose(
    [T.Lambda(lambda x: x.convert('RGB')),
     T.ToTensor(),
     normalize]
)


class TinyImageNetDataset(Dataset):
    """TinyImageNet shard dataset class."""

    NUM_IMAGES_PER_CLASS = 500

    def __init__(self, data_folder: Path, data_type='train', transform=None):
        """Initialize TinyImageNetDataset."""
        super(TinyImageNetDataset, self).__init__()
        self.data_type = data_type
        self._common_data_folder = data_folder
        self._data_folder = os.path.join(data_folder, data_type)
        self.labels = {}  # fname - label number mapping
        self.image_paths = sorted(
            glob.iglob(
                os.path.join(self._data_folder, '**', '*.JPEG'),
                recursive=True
            )
        )
        with open(os.path.join(self._common_data_folder, 'wnids.txt'), 'r') as fp:
            self.label_texts = sorted(
                [text.strip() for text in fp.readlines()]
            )
        self.label_text_to_number = {
            text: i for i, text in enumerate(self.label_texts)
        }
        self.fill_labels()
        self.transform = transform

    def __len__(self) -> int:
        """Return the len of the shard dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return an item by the index."""
        file_path = self.image_paths[index]
        sample = self.read_image(file_path)
        if self.transform:
            sample = self.transform(sample)
        label = self.labels[os.path.basename(file_path)]
        return sample, label

    def read_image(self, path: Path):
        """Read the image."""
        img = Image.open(path)
        return img

    def fill_labels(self) -> None:
        """Fill labels."""
        if self.data_type == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels[f'{label_text}_{cnt}.JPEG'] = i
        elif self.data_type == 'val':
            with open(os.path.join(self._data_folder, 'val_annotations.txt'), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[
                        label_text
                    ]


train_dataset = TinyImageNetDataset(
    os.path.join(common_data_folder, 'tiny-imagenet-200'),
    transform=training_transform
)
test_dataset = TinyImageNetDataset(
    os.path.join(common_data_folder, 'tiny-imagenet-200'),
    data_type='val',
    transform=valid_transform
)


def collaborator_private_attrs(
    index, n_collaborators, batch_size, train_dataset, test_dataset
):

    train = deepcopy(train_dataset)
    test = deepcopy(test_dataset)

    train = random_split(
        train, [len(train) // n_collaborators] * n_collaborators
    )[index]
    test = random_split(
        test, [len(test) // n_collaborators] * n_collaborators
    )[index]

    return {
        'train_loader': torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True
        ),
        'test_loader': torch.utils.data.DataLoader(
            test, batch_size=batch_size,
        ),
    }
