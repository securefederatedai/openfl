# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import json
import os
import subprocess
import sys
from logging import getLogger
from typing import Any, Mapping

import numpy as np
import torch
import torch as pt
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from openfl.federated import PyTorchTaskRunner
from openfl.federated.task.runner_pt import change_tags
from openfl.utilities import Metric, TensorKey
from openfl.utilities.split import split_tensor_dict_for_holdouts

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.model_utils import _init_model, _init_optimizer  # noqa: E402

logger = getLogger(__name__)

NP = os.environ.get("OPENFL_HOROVOD_DEMO_NP", "4")
NETWORK_INTERFACES = os.environ.get("OPENFL_HOROVOD_DEMO_NICS", "localhost")
LOCAL_HOST = os.environ.get("OPENFL_HOROVOD_DEMO_LOCALHOSTIP", "localhost")
HOSTS = os.environ.get("OPENFL_HOROVOD_DEMO_HOSTS", "localhost:4")

print("NP:", NP)
print("NETWORK_INTERFACES:", NETWORK_INTERFACES)
print("LOCAL_HOST:", LOCAL_HOST)
print("HOSTS:", HOSTS)


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
        self.model = _init_model(base_model_name, device)
        self.optimizer, self.lr_scheduler = _init_optimizer(
            self.model, len(self.data_loader.train_set)
        )
        self.initialize_tensorkeys_for_functions()

    def train(self):
        return self.model.train()

    def state_dict(self):
        return get_peft_model_state_dict(self.model)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return set_peft_model_state_dict(self.model, state_dict)

    def save_modelstate(self, col_name, round_num, func_name, kwargs):
        state_path = f"col-{col_name}_rnd-{round_num}_state_{func_name}.pt"
        state_path = os.path.join(os.getcwd(), state_path)
        out_path = f"col-{col_name}_rnd-{round_num}_out_{func_name}.pt"
        out_path = os.path.join(os.getcwd(), out_path)
        data_path = self.data_loader.data_path
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "kwargs": kwargs,
                "lr_scheduler": self.lr_scheduler.state_dict(),
            },
            state_path,
        )
        return state_path, out_path, data_path

    def launch_horovod(
        self, data_path, state_path, out_path, function_name, horovod_kwags
    ):
        arg_list = [
            "horovodrun",
            "-np",
            NP,
            "--network-interfaces",
            NETWORK_INTERFACES,
            "--verbose",
            "-H",
            HOSTS,
            "python",
            os.path.join(os.getcwd(), "src/InHorovodrun.py"),
            "--data_path",
            str(data_path),
            "--state_path",
            state_path,
            "--batch_size",
            str(self.data_loader.batch_size),
            "--kwargs",
            json.dumps(horovod_kwags),
            "--func",
            function_name,
            "--out_path",
            out_path,
        ]
        print(arg_list)
        result = subprocess.run(
            arg_list,
            capture_output=True,
        )
        return result

    def validate_task(
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
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        state_path, out_path, data_path = self.save_modelstate(
            col_name, round_num, "validate", kwargs
        )
        horovod_kwags = {
            "col_name": col_name,
            "round_num": round_num,
            "input_tensor_dict": None,
            "use_tqdm": use_tqdm,
        }
        result = self.launch_horovod(
            data_path,
            state_path,
            out_path,
            "validate",
            horovod_kwags,
        )

        if result.returncode != 0:
            self.logger.info(f"{result.stdout}")
            self.logger.info(f"{result.stderr}")
            raise RuntimeError(result.stderr)
        else:
            self.logger.info(f"{result.stdout}")
            self.logger.info(f"{result.stderr}")

        val_score = torch.load(out_path)["output"]

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

    def train_task(
        self, col_name, round_num, input_tensor_dict, use_tqdm=False, epochs=1, **kwargs
    ):
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
        self.rebuild_model(round_num, input_tensor_dict)
        state_path, out_path, data_path = self.save_modelstate(
            col_name, round_num, "train_batches", kwargs
        )
        horovod_kwags = {
            "col_name": col_name,
            "round_num": round_num,
            "input_tensor_dict": None,
            "use_tqdm": use_tqdm,
        }
        result = self.launch_horovod(
            data_path, state_path, out_path, "train_batches", horovod_kwags
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        checkpoint = torch.load(out_path)
        metric = checkpoint["output"]
        loss_fct_name = checkpoint["loss_fct_name"]
        metric = Metric(name=loss_fct_name, value=np.array(metric))
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # Output metric tensors (scalar)
        origin = col_name
        tags = ("trained",)
        output_metric_dict = {
            TensorKey(metric.name, origin, round_num, True, ("metric",)): metric.value
        }

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs
        )

        # Create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags): nparray
            for tensor_name, nparray in global_model_dict.items()
        }
        # Create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags): nparray
            for tensor_name, nparray in local_model_dict.items()
        }
        # The train/validate aggregated function of the next round will look
        # for the updated model parameters.
        # This ensures they will be resolved locally
        next_local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num + 1, False, ("model",)): nparray
            for tensor_name, nparray in local_model_dict.items()
        }

        global_tensor_dict = {**output_metric_dict, **global_tensorkey_model_dict}
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict,
        }

        # Update the required tensors if they need to be pulled from the
        # aggregator
        # TODO this logic can break if different collaborators have different
        # roles between rounds.
        # For example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator because
        # these are only created after training occurs. A work around could
        # involve doing a single epoch of training on random data to get the
        # optimizer names, and then throwing away the model.
        if self.opt_treatment == "CONTINUE_GLOBAL":
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        # This will signal that the optimizer values are now present,
        # and can be loaded when the model is rebuilt
        self.training_round_completed = True

        # Return global_tensor_dict, local_tensor_dict
        return global_tensor_dict, local_tensor_dict

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
