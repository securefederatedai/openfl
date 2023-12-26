# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
from typing import Any, Mapping

import numpy as np
import torch
import torch as pt
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.optim import AdamW
from openfl.utilities.split import split_tensor_dict_for_holdouts
from openfl.federated import PyTorchTaskRunner
from openfl.federated.task.runner_pt import change_tags
from openfl.utilities import Metric, TensorKey
from transformers import AutoModelForSequenceClassification, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from datasets import load_metric
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import subprocess
import json
from logging import getLogger

logger = getLogger(__name__)

import horovod.torch as hvd
import numpy as np
import torch
import torch as pt
import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from tqdm import tqdm
from openfl.utilities import Metric
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def _init_model(base_model_name = "roberta-base", device=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, return_dict=True
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="lora_only",
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    return model
    
def _init_optimizer(model, num_training_steps):
    ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=0.001)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps * 5,
    )
    return optimizer, lr_scheduler
    