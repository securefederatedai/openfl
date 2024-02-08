#!/usr/bin/env python
# coding: utf-8

# Copyright (C) 2022-2024 TU Darmstadt
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------
# Primary author: Phillip Rieger <phillip.rieger@trust.tu-darmstadt.de>
# Co-authored-by: Torsten Krauss <torsten.krauss@uni-wuerzburg.de>
# ------------------------------------------------------------

import argparse
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.optim as optim
from torchvision import transforms, datasets
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from CrowdGuardClientValidation import CrowdGuardClientValidation
from openfl.experimental.interface import Aggregator, Collaborator, FLSpec
from openfl.experimental.placement import aggregator, collaborator
from openfl.experimental.runtime import LocalRuntime
from urllib.request import urlretrieve

warnings.filterwarnings("ignore")

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.00075
MOMENTUM = 0.9
LOG_INTERVAL = 10
TOTAL_CLIENT_NUMBER = 4
PMR = 0.25
NUMBER_OF_MALICIOUS_CLIENTS = max(1, int(TOTAL_CLIENT_NUMBER * PMR)) if PMR > 0 else 0
NUMBER_OF_BENIGN_CLIENTS = TOTAL_CLIENT_NUMBER - NUMBER_OF_MALICIOUS_CLIENTS
PRETRAINED_MODEL_FILE = 'pretrained_cifar.pt'

# set the random seed for repeatable results
RANDOM_SEED = 10

VOTE_FOR_BENIGN = 1
VOTE_FOR_POISONED = 0
STD_DEV = torch.from_numpy(np.array([0.2023, 0.1994, 0.2010]))
MEAN = torch.from_numpy(np.array([0.4914, 0.4822, 0.4465]))


def download_pretrained_model():
    urlretrieve('https://huggingface.co/prieger/cifar10/resolve/main/pretrained_cifar.pt?'
                'download=true', PRETRAINED_MODEL_FILE)


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


class SequentialWithInternalStatePrediction(nn.Sequential):
    """
    Adapted version of Sequential that implements the function predict_internal_states
    """

    def predict_internal_states(self, x):
        """
        applies the submodules on the input. Compared to forward, this function also returns
        all intermediate outputs
        """
        result = []
        for module in self:
            x = module(x)
            # We can define our layer as we want. We selected Convolutional and
            # Linear Modules as layers here.
            # Differs for every model architecture.
            # Can be defined by the defender.
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                result.append(x)
        return result, x


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = SequentialWithInternalStatePrediction(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = SequentialWithInternalStatePrediction(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

    def predict_internal_states(self, x):
        result, x = self.features.predict_internal_states(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        result += self.classifier.predict_internal_states(x)[0]
        return result


def default_optimizer(model, optimizer_type=None, optimizer_like=None):
    """
    Return a new optimizer based on the optimizer_type or the optimizer template

    Args:
        model:   NN model architected from nn.module class
        optimizer_type: "SGD" or "Adam"
        optimizer_like: "torch.optim.SGD" or "torch.optim.Adam" optimizer
    """
    if optimizer_type == "SGD" or isinstance(optimizer_like, optim.SGD):
        return optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    elif optimizer_type == "Adam" or isinstance(optimizer_like, optim.Adam):
        return optim.Adam(model.parameters())


def test(network, test_loader, device, mode='Benign', move_to_cpu_afterward=True,
         test_train='Test'):
    network.eval()
    network.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader)
    accuracy = float(correct / len(test_loader.dataset))
    print(
        (
            f"{mode} {test_train} set: Avg. loss: {test_loss}, "
            f"Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * accuracy:5.03f}%)"
        )
    )
    if move_to_cpu_afterward:
        network.to("cpu")
    return accuracy


def FedAvg(models):  # NOQA: N802
    """
    Return a Federated average model based on Fedavg algorithm: H. B. Mcmahan,
    E. Moore, D. Ramage, S. Hampson, and B. A. Y.Arcas,
    “Communication-efficient learning of deep networks from decentralized data,” 2017.

    Args:
        models: Python list of locally trained models by each collaborator
    """
    new_model = models[0]
    if len(models) > 1:
        state_dicts = [model.state_dict() for model in models]
        state_dict = new_model.state_dict()
        for key in models[1].state_dict():
            state_dict[key] = np.sum(
                [state[key] for state in state_dicts], axis=0
            ) / len(models)
        new_model.load_state_dict(state_dict)
    return new_model


def scale_update_of_model(to_scale, global_model, scaling_factor):
    """
    Scales the update of a local model (thus the difference between global and local model)
    :param to_scale: local model as state dict
    :pram global_model
    :param scaling factor
    :return scaled local model as state dict
    """
    print(f'Scale Model by {scaling_factor}')
    result = {}
    for name, data in to_scale.items():
        if not (name.endswith('.bias') or name.endswith('.weight')):
            result[name] = data
        else:
            update = data - global_model[name]
            scaled = scaling_factor * update
            result[name] = scaled + global_model[name]
    return result


def create_cluster_map_from_labels(expected_number_of_labels, clustering_labels):
    """
    Converts a list of labels into a dictionary where each label is the key and
    the values are lists/np arrays of the indices from the samples that received
    the respective label
    :param expected_number_of_labels number of samples whose labels are contained in
    clustering_labels
    :param clustering_labels list containing the labels of each sample
    :return dictionary of clusters
    """
    assert len(clustering_labels) == expected_number_of_labels

    clusters = {}
    for i, cluster in enumerate(clustering_labels):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(i)
    return {index: np.array(cluster) for index, cluster in clusters.items()}


def determine_biggest_cluster(clustering):
    """
    Given a clustering, given as dictionary of the form {cluster_id: [items in cluster]}, the
    function returns the id of the biggest cluster
    """
    biggest_cluster_id = None
    biggest_cluster_size = None
    for cluster_id, cluster in clustering.items():
        size_of_current_cluster = np.array(cluster).shape[0]
        if biggest_cluster_id is None or size_of_current_cluster > biggest_cluster_size:
            biggest_cluster_id = cluster_id
            biggest_cluster_size = size_of_current_cluster
    return biggest_cluster_id


class FederatedFlow(FLSpec):
    def __init__(self, model, optimizers, device="cpu", total_rounds=10, top_model_accuracy=0,
                 pmr=0.25, aggregation_algorithm='FedAVG', **kwargs, ):
        if aggregation_algorithm not in ['FedAVG', 'CrowdGuard']:
            raise Exception(f'Unsupported Aggregation Algorithm: {aggregation_algorithm}')
        super().__init__(**kwargs)
        self.aggregation_algorithm = aggregation_algorithm
        self.model = model
        self.global_model = Net()
        self.pmr = pmr
        self.start_time = None
        self.collaborators = None
        self.private = None
        self.optimizers = optimizers
        self.total_rounds = total_rounds
        self.top_model_accuracy = top_model_accuracy
        self.device = device
        self.round_num = 0  # starting round
        print(20 * "#")
        print(f"Round {self.round_num}...")
        print(20 * "#")

    @aggregator
    def start(self):
        self.start_time = time.time()
        print("Performing initialization for model")
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.next(
            self.train,
            foreach="collaborators",
            exclude=["private"],
        )

    # @collaborator  # Uncomment if you want ro run on CPU
    @collaborator(num_gpus=1)  # Assuming GPU(s) is available on the machine
    def train(self):
        self.collaborator_name = self.input
        print(20 * "#")
        print(f"Performing model training for collaborator {self.input} in round {self.round_num}")

        self.model.to(self.device)
        original_model = {n: d.clone() for n, d in self.model.state_dict().items()}
        test(self.model, self.train_loader, self.device, move_to_cpu_afterward=False,
             test_train='Train')
        test(self.model, self.test_loader, self.device, move_to_cpu_afterward=False)
        test(self.model, self.backdoor_test_loader, self.device, mode='Backdoor',
             move_to_cpu_afterward=False)
        self.optimizer = default_optimizer(self.model, optimizer_like=self.optimizers[self.input])

        self.model.train()
        train_losses = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target).to(self.device)
            loss.backward()
            self.optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                train_losses.append(loss.item())

        self.loss = np.mean(train_losses)
        self.training_completed = True

        test(self.model, self.train_loader, self.device, move_to_cpu_afterward=False,
             test_train='Train')
        test(self.model, self.test_loader, self.device, move_to_cpu_afterward=False)
        test(self.model, self.backdoor_test_loader, self.device, mode='Backdoor',
             move_to_cpu_afterward=False)
        if 'malicious' in self.input:
            weights = self.model.state_dict()
            scaled = scale_update_of_model(weights, original_model, 1 / self.pmr)
            self.model.load_state_dict(scaled)
        self.model.to("cpu")
        torch.cuda.empty_cache()
        if self.aggregation_algorithm == 'FedAVG':
            self.next(self.fed_avg_aggregation, exclude=["training_completed"])
        else:
            self.next(self.collect_models, exclude=["training_completed"])

    @aggregator
    def fed_avg_aggregation(self, inputs):
        self.all_models = {input.collaborator_name: input.model.cpu() for input in inputs}
        self.model = FedAvg([m.cpu() for m in self.all_models.values()])
        self.round_num += 1
        if self.round_num + 1 < self.total_rounds:
            self.next(self.train, foreach="collaborators")
        else:
            self.next(self.end)

    @aggregator
    def collect_models(self, inputs):
        # Following the CrowdGuard paper, this should be executed within SGX

        self.all_models = {i.collaborator_name: i.model.cpu() for i in inputs}
        self.next(self.local_validation, foreach="collaborators")

    @collaborator
    def local_validation(self):
        # Following the CrowdGuard paper, this should be executed within SGX
        print(
            f"Performing model validation for collaborator {self.input} in round {self.round_num}"
        )
        self.collaborator_name = self.input
        all_names = list(self.all_models.keys())
        all_models = [self.all_models[n] for n in all_names]
        own_client_index = all_names.index(self.collaborator_name)
        detected_suspicious_models = CrowdGuardClientValidation.validate_models(self.global_model,
                                                                                all_models,
                                                                                own_client_index,
                                                                                self.train_loader,
                                                                                self.device)
        detected_suspicious_models = sorted(detected_suspicious_models)
        print(
            f'Suspicious Models detected by {own_client_index}: {detected_suspicious_models}')

        votes_of_this_client = []
        for c in range(len(all_models)):
            if c == own_client_index:
                votes_of_this_client.append(VOTE_FOR_BENIGN)
            elif c in detected_suspicious_models:
                votes_of_this_client.append(VOTE_FOR_POISONED)
            else:
                votes_of_this_client.append(VOTE_FOR_BENIGN)
        self.votes_of_this_client = {}
        for name, vote in zip(all_names, votes_of_this_client):
            self.votes_of_this_client[name] = vote

        self.next(self.defend)

    @aggregator
    def defend(self, inputs):
        # Following the CrowdGuard paper, this should be executed within SGX

        all_names = list(self.all_models.keys())
        all_votes_by_name = {i.collaborator_name: i.votes_of_this_client for i in inputs}

        all_models = [self.all_models[name] for name in all_names]
        binary_votes = [[all_votes_by_name[own_name][val_name] for val_name in all_names] for
                        own_name in all_names]

        ac_e = AgglomerativeClustering(n_clusters=2, distance_threshold=None,
                                       compute_full_tree=True,
                                       affinity="euclidean", memory=None, connectivity=None,
                                       linkage='single',
                                       compute_distances=True).fit(binary_votes)
        ac_e_labels: list = ac_e.labels_.tolist()
        agglomerative_result = create_cluster_map_from_labels(len(all_names), ac_e_labels)
        print(f'Agglomerative Clustering: {agglomerative_result}')
        agglomerative_negative_cluster = agglomerative_result[
            determine_biggest_cluster(agglomerative_result)]

        db_scan_input_idx_list = agglomerative_negative_cluster
        print(f'DBScan Input: {db_scan_input_idx_list}')
        db_scan_input_list = [binary_votes[vote_id] for vote_id in db_scan_input_idx_list]

        db = DBSCAN(eps=0.5, min_samples=1).fit(db_scan_input_list)
        dbscan_clusters = create_cluster_map_from_labels(len(agglomerative_negative_cluster),
                                                         db.labels_.tolist())
        biggest_dbscan_cluster = dbscan_clusters[determine_biggest_cluster(dbscan_clusters)]
        print(f'DBScan Clustering: {biggest_dbscan_cluster}')

        single_sample_of_biggest_cluster = biggest_dbscan_cluster[0]
        final_voting = db_scan_input_list[single_sample_of_biggest_cluster]
        negatives = [i for i, vote in enumerate(final_voting) if vote == VOTE_FOR_BENIGN]
        recognized_benign_models = [all_models[n] for n in negatives]

        print(f'Negatives: {negatives}')

        self.model = FedAvg([m.cpu() for m in recognized_benign_models])
        del inputs
        self.round_num += 1
        if self.round_num < self.total_rounds:
            print(f'Finished round {self.round_num}/{self.total_rounds}')
            self.next(self.train, foreach="collaborators")
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(20 * "#")
        print("All rounds completed successfully")
        print(20 * "#")
        print("This is the end of the flow")
        print(20 * "#")


def seed_random_generators(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    seed_random_generators(RANDOM_SEED)

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--test_dataset_ratio",
        type=float,
        default=0.4,
        help="Indicate the what fraction of the sample will be used for testing",
    )
    argparser.add_argument(
        "--train_dataset_ratio",
        type=float,
        default=0.4,
        help="Indicate the what fraction of the sample will be used for training",
    )

    argparser.add_argument(
        "--log_dir",
        type=str,
        default="test_debug",
        help="Indicate where to save the privacy loss profile and log files during the training",
    )
    argparser.add_argument(
        "--comm_round",
        type=int,
        default=30,
        help="Indicate the communication round of FL",
    )
    argparser.add_argument(
        "--optimizer_type",
        type=str,
        default="SGD",
        help="Indicate optimizer to use for training",
    )

    args = argparser.parse_args()

    download_pretrained_model()

    # Setup participants
    aggregator_object = Aggregator()
    aggregator_object.private_attributes = {}
    collaborator_names = [f'benign_{i:02d}' for i in range(NUMBER_OF_BENIGN_CLIENTS)] + [
        f'malicious_{i:02d}' for i in range(NUMBER_OF_MALICIOUS_CLIENTS)]
    collaborators = [Collaborator(name=name) for name in collaborator_names]
    if torch.cuda.is_available():
        device = torch.device(
            "cuda:1"
        )  # This will enable Ray library to reserve available GPU(s) for the task
    else:
        device = torch.device("cpu")

    # Prepare local datasets
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD_DEV), ])
    cifar_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    cifar_train = list(cifar_train)
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    cifar_test = list(cifar_test)
    X = torch.stack([x[0] for x in cifar_train] + [x[0] for x in cifar_test])
    Y = torch.LongTensor(
        np.stack(np.array([x[1] for x in cifar_train] + [x[1] for x in cifar_test])))

    # split the dataset
    seed_random_generators(RANDOM_SEED)
    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)

    N_total_samples = len(cifar_test) + len(cifar_train)
    train_dataset_size = int(N_total_samples * args.train_dataset_ratio)
    test_dataset_size = int(N_total_samples * args.test_dataset_ratio)
    X = X[shuffled_indices]
    Y = Y[shuffled_indices]

    train_dataset_data = X[:train_dataset_size]
    train_dataset_targets = Y[:train_dataset_size]

    test_dataset_data = X[train_dataset_size:train_dataset_size + test_dataset_size]
    test_dataset_targets = Y[train_dataset_size:train_dataset_size + test_dataset_size]
    print(f"Dataset info (total {N_total_samples}): train - {test_dataset_targets.shape[0]}, "
          f"test - {test_dataset_targets.shape[0]}, ")

    # partition the dataset for clients

    for idx, collab in enumerate(collaborators):
        # construct the training and test and population dataset
        benign_training_x = train_dataset_data[idx::len(collaborators)]
        benign_training_y = train_dataset_targets[idx::len(collaborators)]

        if 'malicious' in collab.name:
            local_train_data, local_train_targets = poison_data(benign_training_x,
                                                                benign_training_y)
        else:
            local_train_data, local_train_targets = benign_training_x, benign_training_y

        local_test_data = test_dataset_data[idx::len(collaborators)]
        local_test_targets = test_dataset_targets[idx::len(collaborators)]

        poison_test_data, poison_test_targets = poison_data(local_test_data, local_test_targets,
                                                            pdr=1.0)

        collab.private_attributes = {
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

    local_runtime = LocalRuntime(aggregator=aggregator_object, collaborators=collaborators)

    print(f"Local runtime collaborators = {local_runtime.collaborators}")

    # change to the internal flow loop
    model = Net()
    top_model_accuracy = 0
    optimizers = {
        collaborator.name: default_optimizer(model, optimizer_type=args.optimizer_type)
        for collaborator in collaborators
    }
    flflow = FederatedFlow(
        model,
        optimizers,
        device,
        args.comm_round,
        top_model_accuracy,
        NUMBER_OF_MALICIOUS_CLIENTS / TOTAL_CLIENT_NUMBER,
        'CrowdGuard'
    )
    flflow.runtime = local_runtime
    flflow.run()
