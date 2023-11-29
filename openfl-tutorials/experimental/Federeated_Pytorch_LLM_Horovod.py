import argparse
from typing import Any, Mapping

import horovod.torch as hvd
import numpy as np
import torch
import torch as pt
import torch.nn as nn
from datasets import Dataset, load_dataset, load_metric
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import openfl.native as fx
from openfl.federated import PyTorchTaskRunner
from openfl.federated.task.runner_pt import change_tags
from openfl.utilities import Metric, TensorKey
from openfl.utilities.data_splitters import EqualNumPyDataSplitter
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_scheduler)
from transformers.trainer_pt_utils import get_parameter_names
import sys
sys.path.append('openfl/openfl-workspace/torch_llm')
from src.pt_model import LLMTaskRunner
from src.ptglue_inmemory import GlueMrpcFederatedDataLoader
import openfl.interface.workspace as workspace
import os
WORKSPACE_PREFIX = os.path.join(os.path.expanduser('~'), '.local', 'workspace')


def main():
    log_level = 'INFO'
    log_file = None
    workspace.create(WORKSPACE_PREFIX, 'torch_llm')
    os.chdir(WORKSPACE_PREFIX)
    fx.setup_logging(level=log_level, log_file=log_file)
    num_collaborators = 1

    collaborator_models = [
        LLMTaskRunner(data_loader=GlueMrpcFederatedDataLoader(data_slice, 32, collaborator_count=num_collaborators))
        for data_slice in range(num_collaborators)
    ]
    collaborators = {
        "one": collaborator_models[0],
    }

    # Collaborator one's data
    for i, model in enumerate(collaborator_models):
        print(
            f"Collaborator {i}'s training data size: {len(model.data_loader.train_set)}"
        )
        print(
            f"Collaborator {i}'s validation data size: {len(model.data_loader.valid_set)}\n"
        )
    final_fl_model = fx.run_experiment(
        collaborators,
        {"aggregator.settings.rounds_to_train": 5, "tasks.train.kwargs.epochs": 1},
    )


if __name__ == "__main__":
    main()
