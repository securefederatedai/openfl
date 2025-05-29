# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch, torchvision, random
import numpy as np
from torch.utils.data import TensorDataset

RANDOM_SEED = 10
STD_DEV = torch.from_numpy(np.array([0.2023, 0.1994, 0.2010]))
MEAN = torch.from_numpy(np.array([0.4914, 0.4822, 0.4465]))
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 1000

def seed_random_generators(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def trigger_single_image(image):
    """
    Adds a red square with a height/width of 6 pixels into
    the upper left corner of the given image.
    @param image tensor, containing the normalized pixel values of the image.
    The image will be modified in-place.
    @return given image
    """
    color = (torch.Tensor((1, 0, 0)) - MEAN) / STD_DEV
    image[:, 0:6, 0:6] = color.repeat((6, 6, 1)).permute(2, 1, 0)
    return image

def poison_data(samples_to_poison, labels_to_poison, pdr=0.5):
    """
    poisons a given local dataset, consisting of samples and labels, s.t.,
    the given ratio of this image consists of samples for the backdoor behavior
    :param samples_to_poison tensor containing all samples of the local dataset
    :param labels_to_poison tensor containing all labels
    :param pdr poisoned data rate
    :return poisoned local dataset (samples, labels)
    """
    if pdr == 0:
        return samples_to_poison, labels_to_poison

    assert 0 < pdr <= 1.0
    samples_to_poison = samples_to_poison.clone()
    labels_to_poison = labels_to_poison.clone()

    dataset_size = samples_to_poison.shape[0]
    num_samples_to_poison = int(dataset_size * pdr)
    if num_samples_to_poison == 0:
        # corner case for tiny pdrs
        assert pdr > 0  # Already checked above
        assert dataset_size > 1
        num_samples_to_poison += 1

    indices = np.random.choice(dataset_size, size=num_samples_to_poison, replace=False)
    for image_index in indices:
        image = trigger_single_image(samples_to_poison[image_index])
        samples_to_poison[image_index] = image
    labels_to_poison[indices] = 2
    return samples_to_poison, labels_to_poison.long()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(MEAN, STD_DEV), ])
cifar_train = torchvision.datasets.CIFAR10(root="../files", train=True, download=True, transform=transform)
cifar_train = list(cifar_train)
cifar_test = torchvision.datasets.CIFAR10(root="../files", train=False, download=True, transform=transform)
cifar_test = list(cifar_test)


def envoy_attrs(index, malicious, number_of_collaborators, train_dataset_ratio, test_dataset_ratio, cifar_train, cifar_test):
    # split the dataset
    seed_random_generators(RANDOM_SEED)
    X = torch.stack([x[0] for x in cifar_train] + [x[0] for x in cifar_test])
    Y = torch.LongTensor(np.stack(np.array([x[1] for x in cifar_train] + [x[1] for x in cifar_test])))
    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)

    N_total_samples = len(cifar_test) + len(cifar_train)
    train_dataset_size = int(N_total_samples * train_dataset_ratio)
    test_dataset_size = int(N_total_samples * test_dataset_ratio)
    X = X[shuffled_indices]
    Y = Y[shuffled_indices]

    train_dataset_data = X[:train_dataset_size]
    train_dataset_targets = Y[:train_dataset_size]

    test_dataset_data = X[train_dataset_size:train_dataset_size + test_dataset_size]
    test_dataset_targets = Y[train_dataset_size:train_dataset_size + test_dataset_size]
    
    benign_training_x = train_dataset_data[index::number_of_collaborators]
    benign_training_y = train_dataset_targets[index::number_of_collaborators]

    if malicious:
        local_train_data, local_train_targets = poison_data(benign_training_x,
                                                            benign_training_y)
    else:
        local_train_data, local_train_targets = benign_training_x, benign_training_y

    local_test_data = test_dataset_data[index::number_of_collaborators]
    local_test_targets = test_dataset_targets[index::number_of_collaborators]

    poison_test_data, poison_test_targets = poison_data(local_test_data, local_test_targets,
                                                        pdr=1.0)
    
    return {
        "train_loader": torch.utils.data.DataLoader(
            TensorDataset(local_train_data, local_train_targets),
            batch_size=BATCH_SIZE_TRAIN, shuffle=True
        ),
        "test_loader": torch.utils.data.DataLoader(
            TensorDataset(local_test_data, local_test_targets),
            batch_size=BATCH_SIZE_TEST, shuffle=False
        ),
        "backdoor_test_loader": torch.utils.data.DataLoader(
            TensorDataset(poison_test_data, poison_test_targets),
            batch_size=BATCH_SIZE_TEST, shuffle=False
        ),
    }
