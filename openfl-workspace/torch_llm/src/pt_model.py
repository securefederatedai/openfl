# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import os
import sys
from logging import getLogger
from typing import Any, Mapping
from typing import Iterator
from typing import Tuple

import numpy as np
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import Metric
from trl import SFTTrainer, SFTConfig
from .dataset_inmemory import MedQuadDataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.model_utils import (  # noqa: E402
    _init_model,
    _init_optimizer,
    _init_configs,
)

logger = getLogger(__name__)


class LLMTaskRunner(PyTorchTaskRunner):
    def __init__(
        self,
        data_loader: MedQuadDataLoader,
        base_model_name="microsoft/Phi-3-mini-4k-instruct",
        device=None,
        **kwargs,
    ):
        kwargs["data_loader"] = data_loader
        super().__init__(device, **kwargs)
        self.base_model_name = base_model_name
        self.kwargs = kwargs
        self.model, self.peft_config = _init_model(base_model_name, device)
        self.training_config = _init_configs()
        self.optimizer, self.lr_scheduler = _init_optimizer(
            self.model,
            training_args=SFTConfig(**self.training_config),
        )
        self.sftconfig = SFTConfig(
            **self.training_config
        )
        self.initialize_tensorkeys_for_functions()
        self._data_loader: MedQuadDataLoader = self.data_loader

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def state_dict(self):
        return get_peft_model_state_dict(self.model)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ):
        return set_peft_model_state_dict(self.model, state_dict)

    def train_(
        self, train_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]
    ) -> Metric:
        self._data_loader.padding_side = 'right'
        trainer: SFTTrainer = SFTTrainer(
            model=self.model,
            args=self.sftconfig,
            train_dataset=train_dataloader,
            tokenizer=self._data_loader.tokenizer,
            optimizers=(self.optimizer, self.lr_scheduler),
            max_seq_length=2048,
            dataset_text_field="text",
            packing=False
        )
        train_out = trainer.train()
        self.optimizer, self.lr_scheduler = trainer.optimizer, trainer.lr_scheduler
        return Metric(
            name="CrossEntropyLoss",
            value=np.array(train_out.metrics["train_loss"]),
        )

    def validate_(
        self, validation_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]
    ) -> Metric:
        """
        Perform validation on PyTorch Model

        Override this function for your own custom validation function

        Args:
            validation_data_loader: Validation dataset batch generator.
                                    Yields (samples, targets) tuples.
        Returns:
            Metric: An object containing name and np.ndarray value
        """

        trainer = SFTTrainer(
            model=self.model,
            args=self.sftconfig,
            train_dataset=validation_dataloader,
            eval_dataset=validation_dataloader,
            tokenizer=self._data_loader.tokenizer,
            # optimizers=(self.optimizer, self.lr_scheduler),
            max_seq_length=2048,
            dataset_text_field="text",
            packing=False
        )
        self._data_loader.padding_side = 'left'
        evaluation = trainer.evaluate()
        return Metric(name="eval_loss", value=np.array(evaluation["eval_loss"]))
