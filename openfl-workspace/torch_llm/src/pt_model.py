# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import argparse
from typing import Any, Mapping

import horovod.torch as hvd
import numpy as np
import torch
import torch as pt
import torch.nn as nn
import tqdm
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from tqdm import tqdm

from openfl.federated import PyTorchTaskRunner
from openfl.federated.task.runner_pt import change_tags
from openfl.utilities import Metric, TensorKey
from transformers import (AutoModelForSequenceClassification, get_scheduler)
from transformers.trainer_pt_utils import get_parameter_names
from datasets import Dataset, load_dataset, load_metric


class LLMTaskRunner(PyTorchTaskRunner):
    def __init__(
        self,
        data_loader,
        base_model_name="roberta-base",
        device=None,
        metric=None,
        args=None,
        **kwargs,
    ):
        kwargs["data_loader"] = data_loader
        super().__init__(device, **kwargs)
        self.base_model_name = base_model_name
        self.kwargs = kwargs
        self.metric = load_metric("glue", "mrpc")
        self._init_model()
        self._init_optimizer()
        

        self.save_models = []

    def _init_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name, return_dict=True
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="lora_only",
        )
        self.model = get_peft_model(model, peft_config)
        self.model.to(self.device)
        if self.kwargs.get('use_horovod'):
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)

    def _init_optimizer(self):
        ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=0.001)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.data_loader.train_set) * 5,
        )
        if self.kwargs.get('use_horovod'):
            self.optimizer = hvd.DistributedOptimizer(optimizer)
        self.initialize_tensorkeys_for_functions()

    def train(self):
        return self.model.train()

    def state_dict(self):
        return get_peft_model_state_dict(self.model)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return set_peft_model_state_dict(self.model, state_dict)

    def validate(
        self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs
    ):
        """Validate.

        Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB

        """
        self.save_models.append(input_tensor_dict.copy())
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.model.eval()

        self.model.to(self.device)
        val_score = 0
        total_samples = 0

        loader = self.data_loader.get_valid_loader()
        if use_tqdm:
            loader = tqdm(loader, desc="validate")

        with pt.no_grad():
            for sample in loader:
                samples = sample["input_ids"].shape[0]
                total_samples += samples
                output = self.model(**sample)
                # get the index of the max log-probability
                logits = output.logits
                predictions = torch.argmax(logits, dim=-1)
                self.metric.add_batch(
                    predictions=predictions, references=sample["labels"]
                )
        val_score = self.metric.compute()["accuracy"]

        origin = col_name
        suffix = "validate"
        if kwargs["apply"] == "local":
            suffix += "_local"
        else:
            suffix += "_agg"
        tags = ("metric",)
        tags = change_tags(tags, add_field=suffix)
        # TODO figure out a better way to pass in metric for this pytorch
        #  validate function
        output_tensor_dict = {
            TensorKey("acc", origin, round_num, True, tags): np.array(val_score)
        }

        # Empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def train_epoch(self, batch_generator) -> Metric:
        """Train single epoch.

        Override this function in order to use custom training.

        Args:
            batch_generator: Train dataset batch generator. Yields (samples, targets) tuples of
            size = `self.data_loader.batch_size`.
        Returns:
            Metric: An object containing name and np.ndarray value.
        """
        losses = []
        for sample in batch_generator:
            self.model.zero_grad()
            output = self.model(**sample)
            loss = output.loss
            self.model.backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.model.step()
            losses.append(loss.detach().cpu().numpy())
        loss = np.mean(losses)
        if self.model.config.problem_type == "regression":
            loss_fct = MSELoss()
        elif self.model.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
        elif self.model.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
        return Metric(name=loss_fct._get_name(), value=np.array(loss))

    def save_native(
        self,
        filepath,
        model_state_dict_key="model_state_dict",
        optimizer_state_dict_key="optimizer_state_dict",
        **kwargs,
    ):
        """
        Save model and optimizer states in a picked file specified by the \
        filepath. model_/optimizer_state_dicts are stored in the keys provided. \
        Uses pt.save().

        Args:
            filepath (string)                 : Path to pickle file to be
                                                created by pt.save().
            model_state_dict_key (string)     : key for model state dict
                                                in pickled file.
            optimizer_state_dict_key (string) : key for optimizer state
                                                dict in picked file.
            kwargs                            : unused

        Returns:
            None
        """
        pickle_dict = {
            model_state_dict_key: get_peft_model_state_dict(self.model),
            optimizer_state_dict_key: self.optimizer.state_dict(),
        }
        pt.save(pickle_dict, filepath)