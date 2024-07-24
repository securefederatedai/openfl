# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import os
import sys
from logging import getLogger

import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import (
    get_scheduler,
    AutoModelForCausalLM,
)
from transformers.trainer_pt_utils import get_parameter_names
import torch

logger = getLogger(__name__)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
ALL_LAYERNORM_LAYERS = [nn.LayerNorm]


def _init_model(
    base_model_name="microsoft/Phi-3-mini-4k-instruct", device=None
):
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
        device_map=None,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, **model_kwargs
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules="all-linear",
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    return model, peft_config


def _init_optimizer(opt_model, training_args):
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in opt_model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in opt_model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "lr": training_args.learning_rate,
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_cls = AdamW
    # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
    # e.g. for GaLore optimizer.
    if "params" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("params")

    # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
    # e.g. for LOMO optimizer.
    if "model" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("model")

    # For layer-wise dummy optimizers we overwrite
    # optimizer_grouped_parameters with `optimizer_dict`
    # to avoid arguments conflicts.
    if "optimizer_dict" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
    )
    return optimizer, lr_scheduler


def _init_configs():
    training_config = {
        "bf16": True,
        "use_ipex": False,
        "use_cpu": True,
        "do_eval": False,
        "learning_rate": 5.0e-06,
        "log_level": "info",
        "logging_steps": 20,
        "logging_strategy": "steps",
        "lr_scheduler_type": "constant",
        "num_train_epochs": 1,
        "max_steps": -1,
        "output_dir": "./checkpoint_dir",
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": 1,
        "per_device_train_batch_size": 1,
        "remove_unused_columns": True,
        "save_steps": 100,
        "save_total_limit": 1,
        "seed": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.2,
    }

    return training_config
