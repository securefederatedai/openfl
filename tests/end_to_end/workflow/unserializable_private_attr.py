# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.placement import aggregator, collaborator
from openfl.databases import TensorDB
from openfl.utilities import TensorKey
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def FedAvg(models, weights=None):
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():
        state_dict[key] = torch.from_numpy(
            np.average(
                [state[key].numpy() for state in state_dicts], axis=0, weights=weights
            )
        )
    new_model.load_state_dict(state_dict)
    return new_model


class TestFlowUnserializablePrivateAttributes(FLSpec):
    """
    Testflow to validate handling of unserializable private attributes.
    """
    __test__ = False # to prevent pytest from trying to discover tests in the class

    def __init__(self, rounds=5, **kwargs):
        super().__init__(**kwargs)
        self.model = Net()
        self.current_round = 0
        self.n_rounds = rounds

    @aggregator
    def start(self):
        self.collaborators = self.runtime.collaborators
        self.next(self.aggregated_model_validation, foreach="collaborators")

    @collaborator
    def aggregated_model_validation(self):
        log.info(f"Performing aggregated model validation for collaborator {self.input}")
        self.next(self.train)

    @collaborator
    def train(self):
        # Save trained models to Collaborator's Tensor DB
        self.save_model_to_tensordb(
            model=self.model,
            tensordb=self.col_tensor_db,
            origin=self.input,
            round=self.current_round,
            report=False,
            tags="Trained_Tensor",
        )
        log.info(self.col_tensor_db)
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.next(self.join)

    @aggregator
    def join(self, inputs):

        # Update agg_tensor_db with each collaborator's model weights
        for input in inputs:
            # Save model to Aggregator's Tensor DB
            self.save_model_to_tensordb(
                model=input.model,
                tensordb=self.agg_tensor_db,
                origin=input.input,
                round=self.current_round,
                report=False,
                tags="Trained",
            )

        self.model = FedAvg([input.model for input in inputs])

        # Save model to Aggregator's Tensor DB
        self.save_model_to_tensordb(
            model=self.model,
            tensordb=self.agg_tensor_db,
            origin="Agg",
            round=self.current_round,
            report=False,
            tags="Agg_Tensor",
        )
        print(self.agg_tensor_db)

        self.current_round += 1
        if self.current_round < self.n_rounds:
            self.next(self.aggregated_model_validation, foreach="collaborators")
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(f"This is the end of the flow")

    def save_model_to_tensordb(
        self, model=None, tensordb=None, origin=None, round=0, report=False, tags=("")
    ):
        # Update tensor_db
        tensor_key_dict = {}
        for name, param in model.named_parameters():
            tensor_key = TensorKey(
                tensor_name=name,
                origin=origin,
                round_number=round,
                report=False,
                tags=tags,
            )
            tensor_key_dict[tensor_key] = param.detach().cpu().numpy()
        tensordb.cache_tensor(tensor_key_dict)
