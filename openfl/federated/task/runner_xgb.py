# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""XGBoostTaskRunner module."""

from copy import deepcopy
from typing import Iterator, Tuple

import numpy as np
import json

from openfl.federated.task.runner import TaskRunner
from openfl.utilities import Metric, TensorKey, change_tags
from openfl.utilities.split import split_tensor_dict_for_holdouts

import xgboost as xgb
from openfl.utilities import LocalTensor
import json
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


class XGBoostTaskRunner(TaskRunner):
    def __init__(self, **kwargs):
        """Initializes the XGBoostTaskRunner object.

        Args:
            **kwargs: Additional parameters to pass to the functions.
        """
        super().__init__()
        TaskRunner.__init__(self, **kwargs)

        # This is a map of all the required tensors for each of the public
        # functions in XGBoostTaskRunner
        self.required_tensorkeys_for_function = {}
        self.training_round_completed = False


    def validate_task(self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs):
        """Validate Task.

        Run validation of the model on the local data.

        Args:
            col_name (str): Name of the collaborator.
            round_num (int): What round is it.
            input_tensor_dict (dict): Required input tensors (for model).
            use_tqdm (bool): Use tqdm to print a progress bar (Default=True).
            **kwargs: Additional parameters.

        Returns:
            global_output_dict (dict):  Tensors to send back to the aggregator.
            local_output_dict (dict):   Tensors to maintain in the local
                TensorDB.
        """
        if round_num != 0:
            self.model = bytearray(input_tensor_dict)

        loader = self.data_loader.get_valid_loader()

        metric = self.validate_(loader)

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
        output_tensor_dict = {TensorKey(metric.name, origin, round_num, True, tags): metric.value}

        # Empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def train_task(
        self,
        col_name,
        round_num,
        input_tensor_dict,
        use_tqdm=False,
        epochs=1,
        **kwargs,
    ):
        """Train batches task.

        Train the model on the requested number of batches.

        Args:
            col_name (str): Name of the collaborator.
            round_num (int): What round is it.
            input_tensor_dict (dict): Required input tensors (for model).
            use_tqdm (bool): Use tqdm to print a progress bar (Default=True).
            epochs (int): The number of epochs to train.
            **kwargs: Additional parameters.

        Returns:
            global_output_dict (dict):  Tensors to send back to the aggregator.
            local_output_dict (dict):   Tensors to maintain in the local
                TensorDB.
        """
        # self.rebuild_model(round_num, input_tensor_dict)
        # set to "training" mode
        if round_num != 0:
            self.model = bytearray(input_tensor_dict)
        loader = self.data_loader.get_train_loader()
        metric = self.train_(loader)
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

        global_tensor_dict = {
            **output_metric_dict,
            **global_tensorkey_model_dict,
        }
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
        

    def get_tensor_dict(self, with_opt_vars=False):
        """Return the tensor dictionary.

        Args:
            with_opt_vars (bool): Return the tensor dictionary including the
                optimizer tensors (Default=False)

        Returns:
            state (dict): Tensor dictionary {**dict, **optimizer_dict}
        """
        # Gets information regarding tensor model layers and optimizer state.
        # FIXME: self.parameters() instead? Unclear if load_state_dict() or
        # simple assignment is better
        # for now, state dict gives us names which is good
        # FIXME: do both and sanity check each time?

        state = to_cpu_numpy(self.state_dict())

        if with_opt_vars:
            opt_state = _get_optimizer_state(self.optimizer)
            state = {**state, **opt_state}

        return state


    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """Get the required tensors for specified function that could be called
        as part of a task. By default, this is just all of the layers and
        optimizer of the model.

        Args:
            func_name (str): The function name.

        Returns:
            list : [TensorKey].
        """
        if func_name == "validate_task":
            local_model = "apply=" + str(kwargs["apply"])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Set the required tensors for all publicly accessible task methods.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function.

        Args:
            with_opt_vars (bool): Flag to check if optimizer variables are
                included. Defaults to False.

        Returns:
            None
        """
        # TODO there should be a way to programmatically iterate through
        #  all of the methods in the class and declare the tensors.
        # For now this is done manually

        output_model_dict = self.get_tensor_dict(with_opt_vars=with_opt_vars)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs
        )
        if not with_opt_vars:
            global_model_dict_val = global_model_dict
            local_model_dict_val = local_model_dict
        else:
            output_model_dict = self.get_tensor_dict(with_opt_vars=False)
            global_model_dict_val, local_model_dict_val = split_tensor_dict_for_holdouts(
                self.logger,
                output_model_dict,
                **self.tensor_dict_split_fn_kwargs,
            )

        self.required_tensorkeys_for_function["train_task"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
            for tensor_name in global_model_dict
        ]
        self.required_tensorkeys_for_function["train_task"] += [
            TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
            for tensor_name in local_model_dict
        ]

        self.required_tensorkeys_for_function["train_task"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
            for tensor_name in global_model_dict
        ]
        self.required_tensorkeys_for_function["train_task"] += [
            TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
            for tensor_name in local_model_dict
        ]

        # Validation may be performed on local or aggregated (global) model,
        # so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function["validate_task"] = {}
        # TODO This is not stateless. The optimizer will not be
        self.required_tensorkeys_for_function["validate_task"]["apply=local"] = [
            TensorKey(tensor_name, "LOCAL", 0, False, ("trained",))
            for tensor_name in {**global_model_dict_val, **local_model_dict_val}
        ]
        self.required_tensorkeys_for_function["validate_task"]["apply=global"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
            for tensor_name in global_model_dict_val
        ]
        self.required_tensorkeys_for_function["validate_task"]["apply=global"] += [
            TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
            for tensor_name in local_model_dict_val
        ]

    def save_native(
        self,
        filepath,
        model_state_dict_key="model_state_dict",
        optimizer_state_dict_key="optimizer_state_dict",
        **kwargs,
    ):
        """Save model and optimizer states in a picked file specified by the
        filepath. model_/optimizer_state_dicts are stored in the keys provided.
        Uses pt.save().

        Args:
            filepath (str): Path to pickle file to be created by pt.save().
            model_state_dict_key (str): key for model state dict in pickled
                file.
            optimizer_state_dict_key (str): key for optimizer state dict in
                picked file.
            **kwargs: Additional parameters.

        Returns:
            None
        """
        pickle_dict = {
            model_state_dict_key: self.state_dict(),
            optimizer_state_dict_key: self.optimizer.state_dict(),
        }
        torch.save(pickle_dict, filepath)

    def train_(self, train_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]) -> Metric:
        """Train single epoch.

        Override this function in order to use custom training.

        Args:
            batch_generator (Iterator): Train dataset batch generator. Yields
                (samples, targets) tuples of
                size = `self.data_loader.batch_size`.

        Returns:
            Metric: An object containing name and np.ndarray value.
        """
        losses = []
        for data, target in train_dataloader:
            data, target = torch.tensor(data).to(self.device), torch.tensor(target).to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.loss_fn(output=output, target=target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        loss = np.mean(losses)
        return Metric(name=self.loss_fn.__name__, value=np.array(loss))

    def validate_(self, validation_dataloader) -> Metric:
        """Validate model."""

        dtest, y_test = validation_dataloader
        preds = bst.predict(dtest)
        rmse = root_mean_squared_error(y_test, preds)

        return Metric(name="accuracy", value=np.array(rmse))
