# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""XGBoostTaskRunner module."""

# from copy import deepcopy
# from typing import Iterator, Tuple

import json

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

from openfl.federated.task.runner import TaskRunner
from openfl.utilities import Metric, TensorKey, change_tags
from openfl.utilities.split import split_tensor_dict_for_holdouts


class XGBoostTaskRunner(TaskRunner):
    def __init__(self, **kwargs):
        """Initializes the XGBoostTaskRunner object.

        Args:
            **kwargs: Additional parameters to pass to the functions.
        """
        super().__init__(**kwargs)
        self.global_model = None
        self.required_tensorkeys_for_function = {}
        self.training_round_completed = False

    def rebuild_model(self, input_tensor_dict):
        """
        Rebuilds the model using the provided input tensor dictionary.

        This method checks if the 'local_tree' key in the input tensor dictionary is either a non-empty numpy array
        or a non-None value. If this condition is met, it updates the internal tensor dictionary with the provided input.

        Parameters:
        input_tensor_dict (dict): A dictionary containing tensor data. It must include the key 'local_tree', which can be:
                                - A non-empty numpy array
                                - Any non-None value

        Returns:
        None
        """
        if (
            isinstance(input_tensor_dict["local_tree"], np.ndarray)
            and input_tensor_dict["local_tree"].size != 0
        ) or (
            not isinstance(input_tensor_dict["local_tree"], np.ndarray)
            and input_tensor_dict["local_tree"] is not None
        ):
            self.set_tensor_dict(input_tensor_dict)

    def validate_task(self, col_name, round_num, input_tensor_dict, **kwargs):
        """Validate Task.

        Run validation of the model on the local data.

        Args:
            col_name (str): Name of the collaborator.
            round_num (int): What round is it.
            input_tensor_dict (dict): Required input tensors (for model).
            **kwargs: Additional parameters.

        Returns:
            global_output_dict (dict):  Tensors to send back to the aggregator.
            local_output_dict (dict):   Tensors to maintain in the local
                TensorDB.
        """
        loader = self.data_loader.get_valid_dmatrix()

        # during agg validation, self.bst will still be None. during local validation, it will have a value - no need to rebuild
        if self.bst is None:
            self.rebuild_model(input_tensor_dict)

        # if self.bst is still None after rebuilding, then there was no initial global model, so set metric to 0
        if self.bst is None:
            # for first round agg validation, there is no model so set metric to 0
            # TODO: this is not robust, especially if using a loss metric
            metric = Metric(name="accuracy", value=np.array(0))
        else:
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
        **kwargs,
    ):
        """Train batches task.

        Train the model on the requested number of batches.

        Args:
            col_name (str): Name of the collaborator.
            round_num (int): What round is it.
            input_tensor_dict (dict): Required input tensors (for model).
            **kwargs: Additional parameters.

        Returns:
            global_output_dict (dict):  Tensors to send back to the aggregator.
            local_output_dict (dict):   Tensors to maintain in the local
                TensorDB.
        """
        self.rebuild_model(input_tensor_dict)
        loader = self.data_loader.get_train_dmatrix()
        metric = self.train_(loader)
        # Output metric tensors (scalar)
        origin = col_name
        tags = ("trained",)
        output_metric_dict = {
            TensorKey(metric.name, origin, round_num, True, ("metric",)): metric.value
        }

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict()
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
            self.initialize_tensorkeys_for_functions()

        # This will signal that the optimizer values are now present,
        # and can be loaded when the model is rebuilt
        self.training_round_completed = True

        # Return global_tensor_dict, local_tensor_dict
        # import pdb; pdb.set_trace()
        # TODO it is still decodable from here with .tobytes().decode('utf-8')
        return global_tensor_dict, local_tensor_dict

    def get_tensor_dict(self, with_opt_vars=False):
        """
        Retrieves the tensor dictionary containing the model's tree structure.

        This method returns a dictionary with the key 'local_tree', which contains the model's tree structure as a numpy array.
        If the model has not been initialized (`self.bst` is None), it returns an empty numpy array.
        If the global model is not set or is empty, it returns the entire model as a numpy array.
        Otherwise, it returns only the trees added in the latest training session.

        Parameters:
        with_opt_vars (bool): N/A for XGBoost (Default=False).

        Returns:
        dict: A dictionary with the key 'local_tree' containing the model's tree structure as a numpy array.
        """

        if self.bst is None:
            # For initializing tensor dict
            return {"local_tree": np.array([], dtype=np.float32)}

        booster_array = self.bst.save_raw("json")
        booster_dict = json.loads(booster_array)

        if (
            isinstance(self.global_model, np.ndarray) and self.global_model.size == 0
        ) or self.global_model is None:
            booster_float32_array = np.frombuffer(booster_array, dtype=np.uint8).astype(np.float32)
            return {"local_tree": booster_float32_array}

        global_model_booster_dict = json.loads(self.global_model)
        num_global_trees = int(
            global_model_booster_dict["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_trees"
            ]
        )
        num_total_trees = int(
            booster_dict["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"]
        )

        # Calculate the number of trees added in the latest training
        num_latest_trees = num_total_trees - num_global_trees
        latest_trees = booster_dict["learner"]["gradient_booster"]["model"]["trees"][
            -num_latest_trees:
        ]

        latest_trees_json = json.dumps(latest_trees)
        latest_trees_bytes = latest_trees_json.encode("utf-8")
        latest_trees_float32_array = np.frombuffer(latest_trees_bytes, dtype=np.uint8).astype(
            np.float32
        )

        return {"local_tree": latest_trees_float32_array}

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

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False):
        """Set the tensor dictionary.

        Args:
            tensor_dict (dict): The tensor dictionary.
            with_opt_vars (bool): N/A for XGBoost (Default=False).
        """
        # The with_opt_vars argument is not used in this method
        self.global_model = bytearray(tensor_dict["local_tree"].astype(np.uint8).tobytes())
        self.bst = xgb.Booster()
        self.bst.load_model(self.global_model)

    def save_native(
        self,
        filepath,
        **kwargs,
    ):
        """Save XGB booster to file.

        Args:
            filepath (str): Path to pickle file to be created by booster.save_model().
            **kwargs: Additional parameters.

        Returns:
            None
        """
        self.bst.save_model(filepath)

    def train_(self, train_dataloader) -> Metric:
        """Train model."""
        dtrain = train_dataloader["dmatrix"]
        evals = [(dtrain, "train")]
        evals_result = {}

        self.bst = xgb.train(
            self.params,
            dtrain,
            self.num_rounds,
            xgb_model=self.bst,
            evals=evals,
            evals_result=evals_result,
            verbose_eval=False,
        )

        loss = evals_result["train"]["logloss"][-1]
        return Metric(name=self.loss_fn.__name__, value=np.array(loss))

    def validate_(self, validation_dataloader) -> Metric:
        """Validate model."""

        dtest = validation_dataloader["dmatrix"]
        y_test = validation_dataloader["labels"]
        preds = self.bst.predict(dtest)
        y_pred_binary = np.where(preds > 0.5, 1, 0)
        acc = accuracy_score(y_test, y_pred_binary)

        return Metric(name="accuracy", value=np.array(acc))
