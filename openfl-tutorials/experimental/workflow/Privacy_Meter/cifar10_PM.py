# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------
# Primary author: Hongyan Chang <hongyan.chang@intel.com>
# Co-authored-by: Anindya S. Paul <anindya.s.paul@intel.com>
# Co-authored-by: Brandon Edwards <brandon.edwards@intel.com>
# ------------------------------------------------------------

from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
from openfl.experimental.workflow.placement import aggregator, collaborator
import torchvision.transforms as transforms
import pickle
from pathlib import Path

from privacy_meter.model import PytorchModelTensor
import copy
from auditor import (
    PopulationAuditor,
    plot_auc_history,
    plot_tpr_history,
    plot_roc_history,
    PM_report,
)

import time
import os
import argparse
from cifar10_loader import CIFAR10
import warnings

warnings.filterwarnings("ignore")

batch_size_train = 32
batch_size_test = 1000
learning_rate = 0.005
momentum = 0.9
log_interval = 10

# set the random seed for repeatable results
random_seed = 10
torch.manual_seed(random_seed)


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
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
        self.classifier = nn.Sequential(
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


def default_optimizer(model, optimizer_type=None, optimizer_like=None):
    """
    Return a new optimizer based on the optimizer_type or the optimizer template

    Args:
        model:   NN model architected from nn.module class
        optimizer_type: "SGD" or "Adam"
        optimizer_like: "torch.optim.SGD" or "torch.optim.Adam" optimizer
    """
    if optimizer_type == "SGD" or isinstance(optimizer_like, optim.SGD):
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_type == "Adam" or isinstance(optimizer_like, optim.Adam):
        return optim.Adam(model.parameters())


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
        # Convert numpy arrays within the state dictionary to PyTorch tensors
        state_dict_tensors = {key: torch.tensor(value, dtype=torch.float32).cpu() for key, value in state_dict.items()}

        # Load the converted state dictionary into the model
        new_model.load_state_dict(state_dict_tensors)
    return new_model


def inference(network, test_loader, device):
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
            f"Test set: Avg. loss: {test_loss}, "
            f"Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * accuracy}%)"
        )
    )
    network.to("cpu")
    return accuracy


def optimizer_to_device(optimizer, device):
    """
    Sending the "torch.optim.Optimizer" object into the specified device
    for model training and inference

    Args:
        optimizer: torch.optim.Optimizer from "default_optimizer" function
        device: CUDA device id or "cpu"
    """
    if optimizer.state_dict()["state"] != {}:
        if isinstance(optimizer, optim.SGD):
            for param in optimizer.param_groups[0]["params"]:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        elif isinstance(optimizer, optim.Adam):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    else:
        raise (ValueError("No dict keys in optimizer state: please check"))


def load_previous_round_model_and_optimizer_and_perform_testing(
    model, global_model, optimizer, collaborator_name, round_num, device
):
    """
    Load pickle file to retrieve the model and optimizer state dictionary
    from the previous round for each collaborator
    and perform several validation routines with current
    round state dictionaries to test the flow loop.
    Note: this functionality can be enabled through the command line argument
    by setting "--flow_internal_loop_test=True".

    Args:
        model: local collaborator model at the current round
        global_model: Federated averaged model at the aggregator
        optimizer: local collaborator optimizer at the current round
        collaborator_name: name of the collaborator (Type:string)
        round_num: current round (Type:int)
        device: CUDA device id or "cpu"
    """
    print(f"Loading model and optimizer state dict for round {round_num-1}")
    model_prevround = Net()  # instantiate a new model
    model_prevround = model_prevround.to(device)
    optimizer_prevround = default_optimizer(model_prevround, optimizer_like=optimizer)
    if os.path.isfile(
        f"Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}.pickle"
    ):
        with open(
            f"Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}.pickle",
            "rb",
        ) as f:
            model_prevround_config = pickle.load(f)
            model_prevround.load_state_dict(model_prevround_config["model_state_dict"])
            optimizer_prevround.load_state_dict(
                model_prevround_config["optim_state_dict"]
            )

            for param_tensor in model.state_dict():
                for tensor_1, tensor_2 in zip(
                    model.state_dict()[param_tensor],
                    global_model.state_dict()[param_tensor],
                ):
                    if (
                        torch.equal(tensor_1.to(device), tensor_2.to(device))
                        is not True
                    ):
                        raise (
                            ValueError(
                                (
                                    "local and global model differ: "
                                    f"{collaborator_name} at round {round_num-1}."
                                )
                            )
                        )

                if isinstance(optimizer, optim.SGD):
                    if optimizer.state_dict()["state"] != {}:
                        for param_idx in optimizer.state_dict()["param_groups"][0][
                            "params"
                        ]:
                            for tensor_1, tensor_2 in zip(
                                optimizer.state_dict()["state"][param_idx][
                                    "momentum_buffer"
                                ],
                                optimizer_prevround.state_dict()["state"][param_idx][
                                    "momentum_buffer"
                                ],
                            ):
                                if (
                                    torch.equal(
                                        tensor_1.to(device), tensor_2.to(device)
                                    )
                                    is not True
                                ):
                                    raise (
                                        ValueError(
                                            (
                                                "Momentum buffer data differ: "
                                                f"{collaborator_name} at round {round_num-1}"
                                            )
                                        )
                                    )
                    else:
                        raise (ValueError("Current optimizer state is empty"))

                model_params = [
                    model.state_dict()[param_tensor]
                    for param_tensor in model.state_dict()
                ]
                for idx, param in enumerate(optimizer.param_groups[0]["params"]):
                    for tensor_1, tensor_2 in zip(param.data, model_params[idx]):
                        if (
                            torch.equal(tensor_1.to(device), tensor_2.to(device))
                            is not True
                        ):
                            raise (
                                ValueError(
                                    (
                                        "Model and optimizer do not point "
                                        "to the same params for collaborator: "
                                        f"{collaborator_name} at round {round_num-1}."
                                    )
                                )
                            )

    else:
        raise (ValueError("No such name of pickle file exists"))


def save_current_round_model_and_optimizer_for_next_round_testing(
    model, optimizer, collaborator_name, round_num
):
    """
    Save the model and optimizer state dictionary
    of a collaboartor ("collaborator_name")
    in a given round ("round_num") into a pickle file
    for later retrieving and verifying its correctness.
    This provide the user the ability to verify the fields
    in the model and optimizer state dictionary and
    may provide confidence on the results of privacy auditing.
    Note: this functionality can be enabled through the command line
    argument by setting "--flow_internal_loop_test=True".

    Args:
        model: local collaborator model at the current round
        optimizer: local collaborator optimizer at the current round
        collaborator_name: name of the collaborator (Type:string)
        round_num: current round (Type:int)
    """
    model_config = {
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
    }
    with open(
        f"Collaborator_{collaborator_name}_model_config_roundnumber_{round_num}.pickle",
        "wb",
    ) as f:
        pickle.dump(model_config, f)


class FederatedFlow(FLSpec):
    def __init__(
        self,
        model,
        optimizers,
        device="cpu",
        total_rounds=10,
        top_model_accuracy=0,
        flow_internal_loop_test=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.global_model = Net()
        self.optimizers = optimizers
        self.total_rounds = total_rounds
        self.top_model_accuracy = top_model_accuracy
        self.device = device
        self.flow_internal_loop_test = flow_internal_loop_test
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
            self.aggregated_model_validation,
            foreach="collaborators",
            exclude=["private"],
        )

    @collaborator
    def aggregated_model_validation(self):
        print(
            (
                "Performing aggregated model validation for collaborator: "
                f"{self.input} in round {self.round_num}"
            )
        )
        self.agg_validation_score = inference(self.model, self.test_loader, self.device)
        print(f"{self.input} value of {self.agg_validation_score}")
        self.collaborator_name = self.input
        self.next(self.train)

    @collaborator
    def train(self):
        print(20 * "#")
        print(
            f"Performing model training for collaborator {self.input} in round {self.round_num}"
        )

        self.model.to(self.device)
        self.optimizer = default_optimizer(
            self.model, optimizer_like=self.optimizers[self.input]
        )

        if self.round_num > 0:
            self.optimizer.load_state_dict(
                deepcopy(self.optimizers[self.input].state_dict())
            )
            optimizer_to_device(optimizer=self.optimizer, device=self.device)

            if self.flow_internal_loop_test:
                load_previous_round_model_and_optimizer_and_perform_testing(
                    self.model,
                    self.global_model,
                    self.optimizer,
                    self.collaborator_name,
                    self.round_num,
                    self.device,
                )

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
            if batch_idx % log_interval == 0:
                train_losses.append(loss.item())

        self.loss = np.mean(train_losses)
        self.training_completed = True

        if self.flow_internal_loop_test:
            save_current_round_model_and_optimizer_for_next_round_testing(
                self.model, self.optimizer, self.collaborator_name, self.round_num
            )

        self.model.to("cpu")
        tmp_opt = deepcopy(self.optimizers[self.input])
        tmp_opt.load_state_dict(self.optimizer.state_dict())
        self.optimizer = tmp_opt
        torch.cuda.empty_cache()
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        print(
            (
                "Performing local model validation for collaborator: "
                f"{self.input} in round {self.round_num}"
            )
        )
        print(self.device)
        start_time = time.time()

        print("Test dataset performance")
        self.local_validation_score = inference(
            self.model, self.test_loader, self.device
        )
        print("Train dataset performance")
        self.local_validation_score_train = inference(
            self.model, self.train_loader, self.device
        )

        print(
            (
                "Doing local model validation for collaborator: "
                f"{self.input}: {self.local_validation_score}"
            )
        )
        print(f"local validation time cost {(time.time() - start_time)}")

        if (
            self.round_num == 0
            or self.round_num % self.local_pm_info.interval == 0
            or self.round_num == self.total_rounds
        ):
            print("Performing Auditing")
            self.next(self.audit)
        else:
            self.next(self.join, exclude=["training_completed"])

    @collaborator
    def audit(self):
        print(
            (
                "Performing local and global model auditing for collaborator: "
                f"{self.input} in round {self.round_num}"
            )
        )
        begin_time = time.time()

        datasets = {
            "train": self.train_dataset,
            "test": self.test_dataset,
            "audit": self.population_dataset,
        }

        start_time = time.time()
        # batch_size for the PytorchModelTensor indicates batch size for computing the signals.
        # for computing loss and logits, it can be large, e.g., 1000.
        # for computing the signal_norm, it should be around 25.
        # Otherwise, one may get OOM depending on the GPU memory.

        target_model = PytorchModelTensor(
            copy.deepcopy(self.model), nn.CrossEntropyLoss(), self.device
        )
        self.local_pm_info = PopulationAuditor(
            target_model, datasets, self.local_pm_info
        )
        target_model.model_obj.to("cpu")
        self.local_pm_info.update_history("round", self.round_num)

        print(f"population attack for the local model uses {time.time() - start_time}")

        start_time = time.time()
        target_model = PytorchModelTensor(
            copy.deepcopy(self.global_model), nn.CrossEntropyLoss(), self.device
        )
        self.global_pm_info = PopulationAuditor(
            target_model, datasets, self.global_pm_info
        )
        self.global_pm_info.update_history("round", self.round_num)
        target_model.model_obj.to("cpu")
        print(f"population attack for the global model uses {time.time() - start_time}")

        start_time = time.time()

        history_dict = {
            "PM Result (Local)": self.local_pm_info,
            "PM Result (Global)": self.global_pm_info,
        }

        # # generate the plot for the privacy loss
        plot_tpr_history(history_dict, self.input, self.local_pm_info.fpr_tolerance)
        plot_auc_history(history_dict, self.input)
        plot_roc_history(history_dict, self.input)

        # save the privacy report
        saving_path = f"{self.local_pm_info.log_dir}/{self.input}.pkl"
        Path(self.local_pm_info.log_dir).mkdir(parents=True, exist_ok=True)
        with open(saving_path, "wb") as handle:
            pickle.dump(history_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"auditing time: {time.time() - begin_time}")

        # Clean up state before transitioning to collaborator
        delattr(self, "train_dataset")
        delattr(self, "train_loader")
        delattr(self, "test_dataset")
        delattr(self, "test_loader")
        delattr(self, "population_dataset")
        self.next(self.join, exclude=["training_completed"])

    @aggregator
    def join(self, inputs):
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs
        ) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs
        ) / len(inputs)
        print(
            f"Average aggregated model validation values = {self.aggregated_model_accuracy}"
        )
        print(f"Average training loss = {self.average_loss}")
        print(f"Average local model validation values = {self.local_model_accuracy}")

        self.model = FedAvg([input.model.cpu() for input in inputs])
        self.global_model.load_state_dict(deepcopy(self.model.state_dict()))
        self.optimizers.update(
            {input.collaborator_name: input.optimizer for input in inputs}
        )

        del inputs
        self.next(self.check_round_completion)

    @aggregator
    def check_round_completion(self):
        if self.round_num != self.total_rounds:
            if self.aggregated_model_accuracy > self.top_model_accuracy:
                print(
                    (
                        "Accuracy improved to "
                        f"{self.aggregated_model_accuracy} for round {self.round_num}"
                    )
                )
                self.top_model_accuracy = self.aggregated_model_accuracy
            self.round_num += 1
            print(20 * "#")
            print(f"Round {self.round_num}...")
            print(20 * "#")
            self.next(
                self.aggregated_model_validation,
                foreach="collaborators",
                exclude=["private"],
            )
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(20 * "#")
        print("All rounds completed successfully")
        print(20 * "#")
        print("This is the end of the flow")
        print(20 * "#")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--audit_dataset_ratio",
        type=float,
        default=0.2,
        help="Indicate the what fraction of the sample will be used for auditing",
    )
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
        "--signals",
        nargs="*",
        type=str,
        default=["loss", "gradient_norm", "logits"],
        help="Indicate which signal to use for membership inference attack",
    )
    argparser.add_argument(
        "--fpr_tolerance",
        nargs="*",
        type=float,
        default=[0.1, 0.5, 0.9],
        help="Indicate false positive tolerance rates in which users are interested",
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
        "--auditing_interval", type=int, default=1, help="Indicate auditing interval"
    )
    argparser.add_argument(
        "--is_features",
        type=bool,
        default=True,
        help="Indicate whether to use the gradient norm with respect to the features as a signal",
    )
    argparser.add_argument(
        "--layer_number",
        type=int,
        default=10,
        help="Indicate whether layer to compute the gradient or gradient norm",
    )
    argparser.add_argument(
        "--flow_internal_loop_test",
        type=bool,
        default=False,
        help="Indicate enabling of internal loop testing of Federated Flow",
    )
    argparser.add_argument(
        "--optimizer_type",
        type=str,
        default="SGD",
        help="Indicate optimizer to use for training",
    )

    args = argparser.parse_args()

    # Setup participants
    # If running with GPU and 1 GPU is available then
    # Set `num_gpus=0.3` to run on GPU
    aggregator = Aggregator()

    collaborator_names = ["Portland", "Seattle"]

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Download and setup the train, and test dataset
    transform = transforms.Compose([transforms.ToTensor()])

    cifar_train = CIFAR10(root="./data", train=True, download=True, transform=transform)

    cifar_test = CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Split the dataset in train, test, and population dataset
    N_total_samples = len(cifar_test) + len(cifar_train)
    train_dataset_size = int(N_total_samples * args.train_dataset_ratio)
    test_dataset_size = int(N_total_samples * args.test_dataset_ratio)
    audit_dataset_size = min(
        int(N_total_samples * args.audit_dataset_ratio),
        N_total_samples - test_dataset_size - train_dataset_size,
    )

    X = np.concatenate([cifar_test.data, cifar_train.data])
    Y = np.concatenate([cifar_test.targets, cifar_train.targets]).tolist()

    train_dataset = deepcopy(cifar_train)
    train_dataset.data = X[:train_dataset_size]
    train_dataset.targets = Y[:train_dataset_size]

    test_dataset = deepcopy(cifar_test)
    test_dataset.data = X[train_dataset_size: train_dataset_size + test_dataset_size]
    test_dataset.targets = Y[
        train_dataset_size: train_dataset_size + test_dataset_size
    ]

    population_dataset = deepcopy(cifar_test)
    population_dataset.data = X[-audit_dataset_size:]
    population_dataset.targets = Y[-audit_dataset_size:]

    print(
        (
            f"Dataset info (total {N_total_samples}): "
            f"train - {len(train_dataset)}, "
            f"test - {len(test_dataset)}, "
            f"audit - {len(population_dataset)}"
        )
    )

    # Split train, test, and population dataset among collaborators
    # this function will be called before executing collaborator steps
    # which will return private attributes dictionary for each collaborator
    def callable_to_initialize_collaborator_private_attributes(
        index, n_collaborators, train_ds, test_ds, population_ds, args
    ):
        # construct the training and test and population dataset
        local_train = deepcopy(train_ds)
        local_test = deepcopy(test_ds)
        local_population = deepcopy(population_ds)

        local_train.data = train_ds.data[index::n_collaborators]
        local_train.targets = train_ds.targets[index::n_collaborators]

        local_test.data = test_ds.data[index::n_collaborators]
        local_test.targets = test_ds.targets[index::n_collaborators]

        local_population.data = population_ds.data[index::n_collaborators]
        local_population.targets = population_ds.targets[index::n_collaborators]

        # initialize pm report to track the privacy loss during the training
        local_pm_info = PM_report(
            fpr_tolerance_list=args.fpr_tolerance,
            is_report_roc=True,
            level="local",
            signals=args.signals,
            log_dir=args.log_dir,
            interval=args.auditing_interval,
            other_info={
                "is_features": args.is_features,
                "layer_number": args.layer_number,
            },
        )
        global_pm_info = PM_report(
            fpr_tolerance_list=args.fpr_tolerance,
            is_report_roc=True,
            level="global",
            signals=args.signals,
            log_dir=args.log_dir,
            interval=args.auditing_interval,
            other_info={
                "is_features": args.is_features,
                "layer_number": args.layer_number,
            },
        )

        Path(local_pm_info.log_dir).mkdir(parents=True, exist_ok=True)
        Path(global_pm_info.log_dir).mkdir(parents=True, exist_ok=True)

        return {
            "local_pm_info": local_pm_info,
            "global_pm_info": global_pm_info,
            "train_dataset": local_train,
            "test_dataset": local_test,  # provide the dataset obj for the auditing purpose
            "population_dataset": local_population,
            "train_loader": torch.utils.data.DataLoader(
                local_train, batch_size=batch_size_train, shuffle=True
            ),
            "test_loader": torch.utils.data.DataLoader(
                local_test, batch_size=batch_size_test, shuffle=False
            ),
        }

    collaborators = []
    for idx, collab_name in enumerate(collaborator_names):
        collaborators.append(
            Collaborator(
                name=collab_name,
                private_attributes_callable=callable_to_initialize_collaborator_private_attributes,
                # If 1 GPU is available in the machine
                # Set `num_gpus=0.0` to `num_gpus=0.3` to run on GPU
                # with ray backend with 2 collaborators
                num_cpus=0.0,
                num_gpus=0.0,
                index=idx,
                n_collaborators=len(collaborator_names),
                train_ds=train_dataset,
                test_ds=test_dataset,
                population_ds=population_dataset,
                args=args,
            )
        )

    # Set backend='ray' to use ray-backend
    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators, backend="single_process"
    )

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
        args.flow_internal_loop_test,
    )

    flflow.runtime = local_runtime
    flflow.run()
