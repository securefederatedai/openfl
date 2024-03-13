# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import os
import sys
from typing import Any, Mapping

import horovod.torch as hvd
import numpy as np
import torch
import torch as pt
import torch.nn as nn
import tqdm
import datasets
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from openfl.utilities import Metric

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.model_utils import _init_model, _init_optimizer  # noqa: E402


def simple_accuracy(preds, labels):
    return float((preds == labels).mean())


class SimpleAcc(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="_DESCRIPTION",
            citation="_CITATION",
            inputs_description="_KWARGS_DESCRIPTION",
            features=datasets.Features(
                {
                    "predictions": datasets.Value(
                        "int64" if self.config_name != "stsb" else "float32"
                    ),
                    "references": datasets.Value(
                        "int64" if self.config_name != "stsb" else "float32"
                    ),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, predictions, references):
        return {"accuracy": simple_accuracy(predictions, references)}


class LLMTrainer(nn.Module):
    def __init__(
        self,
        data_loader,
        base_model_name="roberta-base",
        device=None,
        metric=None,
        args=None,
        **kwargs,
    ):
        super().__init__()
        self.data_loader = data_loader
        self.base_model_name = base_model_name
        self.kwargs = kwargs
        self.device = device

        self.metric = SimpleAcc()
        self.model = _init_model(base_model_name, device)
        self.optimizer, self.lr_scheduler = _init_optimizer(
            self.model, len(self.data_loader.train_set)
        )

    def train(self):
        return self.model.train()

    def state_dict(self):
        return get_peft_model_state_dict(self.model)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return set_peft_model_state_dict(self.model, state_dict)

    def load_state(self, kwargs):
        print("loading data", os.getcwd())
        if hvd.rank() == 0:
            checkpoint = torch.load(kwargs["state_path"])
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            kwargs.update(checkpoint["kwargs"])
            print("loaded")
        print("kwags broadcast")
        kwargs = hvd.broadcast_object(kwargs, root_rank=0)
        print("optimizer broadcast")
        optim_state = hvd.broadcast_object(self.optimizer.state_dict(), root_rank=0)
        print("model broadcast")
        state_dict = hvd.broadcast_object(self.state_dict(), root_rank=0)
        print("scheduler broadcast")
        lr_scheduler_state_dict = hvd.broadcast_object(
            self.lr_scheduler.state_dict(), root_rank=0
        )
        if hvd.rank() > 0:
            self.load_state_dict(state_dict)
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
            self.optimizer.load_state_dict(optim_state)

    def train_batches(self, round_num, use_tqdm=False, epochs=1, **kwargs):
        """Train batches.

        Train the model on the requested number of batches.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)
            epochs:              The number of epochs to train

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """
        self.load_state(kwargs)

        self.train()
        self.to(self.device)
        for epoch in range(epochs):
            loader = self.data_loader.get_train_loader()
            if use_tqdm:
                loader = tqdm.tqdm(loader, desc="train epoch")
            metric = self.train_epoch(loader)
        metric = hvd.allreduce(torch.from_numpy(metric))
        if hvd.rank() == 0:
            if self.model.config.problem_type == "regression":
                loss_fct = MSELoss()
            elif self.model.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
            elif self.model.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
            torch.save(
                {
                    "output": metric,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss_fct_name": loss_fct._get_name(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                },
                kwargs["out_path"],
            )

    def validate(self, round_num, use_tqdm=False, **kwargs):
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
        self.load_state(kwargs)

        self.model.eval()
        self.model.to(self.device)
        val_score = 0
        total_samples = 0
        loader = self.data_loader.get_valid_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="validate")
        samples_run = 0
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
                samples_run += len(sample)
        val_score = np.asarray(self.metric.compute()["accuracy"])
        result = hvd.allreduce(torch.from_numpy(val_score))
        if hvd.rank() == 0:
            torch.save({"output": result}, kwargs["out_path"])
        hvd.join()

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
            loss.backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            losses.append(loss.detach().cpu().numpy())
        loss = np.mean(losses)
        return np.array(loss)
