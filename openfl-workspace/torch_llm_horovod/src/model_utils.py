# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import os
import sys
from logging import getLogger

import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

logger = getLogger(__name__)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
ALL_LAYERNORM_LAYERS = [nn.LayerNorm]


def _init_model(base_model_name="roberta-base", device=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, return_dict=True, num_labels=6,
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
