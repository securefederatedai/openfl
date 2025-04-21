# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""GaNDLFTaskRunner module."""

import logging
import os
from copy import deepcopy
from typing import Union

import numpy as np
import torch as pt
import yaml
from GANDLF.compute.forward_pass import validate_network
from GANDLF.compute.generic import create_pytorch_objects
from GANDLF.compute.training_loop import train_network
from GANDLF.config_manager import ConfigManager

from openfl.federated.task.runner import TaskRunner
from openfl.utilities import TensorKey
from openfl.utilities.split import split_tensor_dict_for_holdouts

logger = logging.getLogger(__name__)


class GaNDLFTaskRunner(TaskRunner):
    """GaNDLF Model class for Federated Learning.

    This class provides methods to manage and manipulate GaNDLF models in a
    federated learning context.

    Attributes:
        build_model (function or class): Function or Class to build the model.
        lambda_opt (function): Lambda function for the optimizer.
        model (Model): The built model.
        optimizer (Optimizer): Optimizer for the model.
        scheduler (Scheduler): Scheduler for the model.
        params (Parameters): Parameters for the model.
        device (str): Device for the model.
        training_round_completed (bool): Whether the training round has been
            completed.
        required_tensorkeys_for_function (dict): Required tensorkeys for
            function.
        tensor_dict_split_fn_kwargs (dict): Keyword arguments for the tensor
            dict split function.
    """

    def __init__(
        self,
        gandlf_config: Union[str, dict] = None,
        device: str = None,
        **kwargs,
    ):
        """Initializes the GaNDLFTaskRunner object.

        Sets up the initial state of the GaNDLFTaskRunner object, initializing
        various components needed for the federated model.
        Args:
            gandlf_config (Union[str, dict], optional): GaNDLF configuration.
                Can be a string (file path) or a dictionary. Defaults to None.
            device (str, optional): Compute device. Defaults to None
                (default="cpu").
            **kwargs: Additional parameters to pass to the function.
        """
        super().__init__(**kwargs)

        assert bool(gandlf_config), "gandlf_config must be specified"

        # allow pass-through of a gandlf config as a file or a dict

        train_csv = self.data_loader.train_csv
        val_csv = self.data_loader.val_csv

        if isinstance(gandlf_config, str) and os.path.exists(gandlf_config):
            gandlf_config = yaml.safe_load(open(gandlf_config, "r"))

        gandlf_config = ConfigManager(gandlf_config)

        (
            model,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
            params,
        ) = create_pytorch_objects(
            gandlf_config, train_csv=train_csv, val_csv=val_csv, device=device
        )
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = params
        self.device = device

        # pass the actual dataloaders to the wrapper loader
        self.data_loader.set_dataloaders(train_loader, val_loader)

        self.training_round_completed = False

        self.required_tensorkeys_for_function = {}

        # FIXME: why isn't this initial call in runner_pt?
        self.initialize_tensorkeys_for_functions(with_opt_vars=False)

        # overwrite attribute to account for one optimizer param (in every
        # child model that does not overwrite get and set tensordict) that is
        # not a numpy array
        self.tensor_dict_split_fn_kwargs.update({"holdout_tensor_names": ["__opt_state_needed"]})

    def rebuild_model(self, round_num, input_tensor_dict, validation=False):
        """Parse tensor names and update weights of model. Handles the
        optimizer treatment.

        Args:
            round_num: The current round number.
            input_tensor_dict (dict): The input tensor dictionary used to
                update the weights of the model.
            validation (bool, optional): A flag indicating whether the model
                is in validation. Defaults to False.

        Returns:
            None
        """

        if self.opt_treatment == "RESET":
            self.reset_opt_vars()
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)
        elif (
            self.training_round_completed
            and self.opt_treatment == "CONTINUE_GLOBAL"
            and not validation
        ):
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=True)
        else:
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)

    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs):
        """Validate.
        Run validation of the model on the local data.
        Args:
            col_name (str): Name of the collaborator.
            round_num (int): Current round number.
            input_tensor_dict (dict): Required input tensors (for model).
            use_tqdm (bool, optional): Use tqdm to print a progress bar.
                Defaults to False.
            **kwargs: Key word arguments passed to GaNDLF main_run.

        Returns:
            output_tensor_dict (dict): Tensors to send back to the aggregator.
            {} (dict): Tensors to maintain in the local TensorDB.
        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.model.eval()

        epoch_valid_loss, epoch_valid_metric = validate_network(
            self.model,
            self.data_loader.val_dataloader,
            self.scheduler,
            self.params,
            round_num,
            mode="validation",
        )

        logger.info(epoch_valid_loss)
        logger.info(epoch_valid_metric)

        origin = col_name
        suffix = "validate"
        if kwargs["apply"] == "local":
            suffix += "_local"
        else:
            suffix += "_agg"
        tags = ("metric", suffix)

        output_tensor_dict = {}
        valid_loss_tensor_key = TensorKey("valid_loss", origin, round_num, True, tags)
        output_tensor_dict[valid_loss_tensor_key] = np.array(epoch_valid_loss)
        for k, v in epoch_valid_metric.items():
            tensor_key = TensorKey(f"valid_{k}", origin, round_num, True, tags)
            output_tensor_dict[tensor_key] = np.array(v)

        # Empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def train(
        self,
        col_name,
        round_num,
        input_tensor_dict,
        use_tqdm=False,
        epochs=1,
        **kwargs,
    ):
        """Train batches.
        Train the model on the requested number of batches.
        Args:
            col_name (str): Name of the collaborator.
            round_num (int): Current round number.
            input_tensor_dict (dict): Required input tensors (for model).
            use_tqdm (bool, optional): Use tqdm to print a progress bar.
                Defaults to False.
            epochs (int, optional): The number of epochs to train. Defaults to 1.
            **kwargs: Key word arguments passed to GaNDLF main_run.

        Returns:
            global_tensor_dict (dict): Tensors to send back to the aggregator.
            local_tensor_dict (dict): Tensors to maintain in the local
                TensorDB.
        """
        self.rebuild_model(round_num, input_tensor_dict)
        # set to "training" mode
        self.model.train()
        for epoch in range(epochs):
            logger.info("Run %s epoch of %s round", epoch, round_num)
            # FIXME: do we want to capture these in an array
            # rather than simply taking the last value?
            epoch_train_loss, epoch_train_metric = train_network(
                self.model,
                self.data_loader.train_dataloader,
                self.optimizer,
                self.params,
            )

        # output model tensors (Doesn't include TensorKey)
        tensor_dict = self.get_tensor_dict(with_opt_vars=True)

        metric_dict = {"loss": epoch_train_loss}
        for k, v in epoch_train_metric.items():
            metric_dict[f"train_{k}"] = v

        # Return global_tensor_dict, local_tensor_dict
        # is this even pt-specific really?
        global_tensor_dict, local_tensor_dict = create_tensorkey_dicts(
            tensor_dict,
            metric_dict,
            col_name,
            round_num,
            self.tensor_dict_split_fn_kwargs,
        )

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
                optimizer tensors (Default=False).

        Returns:
            state (dict): Tensor dictionary {**dict, **optimizer_dict}
        """
        # Gets information regarding tensor model layers and optimizer state.
        # FIXME: self.parameters() instead? Unclear if load_state_dict() or
        # simple assignment is better
        # for now, state dict gives us names which is good
        # FIXME: do both and sanity check each time?

        state = to_cpu_numpy(self.model.state_dict())

        if with_opt_vars:
            opt_state = _get_optimizer_state(self.optimizer)
            state = {**state, **opt_state}

        return state

    def _get_weights_names(self, with_opt_vars=False):
        """Get the names of the weights.

        Args:
            with_opt_vars (bool, optional): Include the optimizer variables.
                Defaults to False.

        Returns:
            list: List of weight names.
        """
        # Gets information regarding tensor model layers and optimizer state.
        # FIXME: self.parameters() instead? Unclear if load_state_dict() or
        # simple assignment is better
        # for now, state dict gives us names which is good
        # FIXME: do both and sanity check each time?

        state = self.model.state_dict().keys()

        if with_opt_vars:
            opt_state = _get_optimizer_state(self.model.optimizer)
            state += opt_state.keys()

        return state

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False):
        """Set the tensor dictionary.

        Args:
            tensor_dict (dict): The tensor dictionary.
            with_opt_vars (bool, optional): Include the optimizer tensors.
                Defaults to False.
        """
        set_pt_model_from_tensor_dict(self.model, tensor_dict, self.device, with_opt_vars)

    def get_optimizer(self):
        """Get the optimizer of this instance.

        Returns:
            Optimizer: The optimizer of this instance.
        """
        return self.optimizer

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """Get the required tensors for specified function that could be called
        as part of a task.

        By default, this is just all of the layers and optimizer of the model.

        Args:
            func_name (str): Function name.
            **kwargs: Additional keyword arguments.

        Returns:
            required_tensorkeys_for_function (list): List of required
                TensorKey.
        """
        if func_name == "validate":
            local_model = "apply=" + str(kwargs["apply"])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Set the required tensors for all publicly accessible task methods.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function.

        Args:
            with_opt_vars (bool, optional): Include the optimizer tensors.
                Defaults to False.
        """
        # TODO there should be a way to programmatically iterate through
        #  all of the methods in the class and declare the tensors.
        # For now this is done manually

        output_model_dict = self.get_tensor_dict(with_opt_vars=with_opt_vars)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            output_model_dict, **self.tensor_dict_split_fn_kwargs
        )
        if not with_opt_vars:
            global_model_dict_val = global_model_dict
            local_model_dict_val = local_model_dict
        else:
            output_model_dict = self.get_tensor_dict(with_opt_vars=False)
            global_model_dict_val, local_model_dict_val = split_tensor_dict_for_holdouts(
                output_model_dict,
                **self.tensor_dict_split_fn_kwargs,
            )

        self.required_tensorkeys_for_function["train"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
            for tensor_name in global_model_dict
        ]
        self.required_tensorkeys_for_function["train"] += [
            TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
            for tensor_name in local_model_dict
        ]

        # Validation may be performed on local or aggregated (global) model,
        # so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function["validate"] = {}
        # TODO This is not stateless. The optimizer will not be
        self.required_tensorkeys_for_function["validate"]["apply=local"] = [
            TensorKey(tensor_name, "LOCAL", 0, False, ("trained",))
            for tensor_name in {**global_model_dict_val, **local_model_dict_val}
        ]
        self.required_tensorkeys_for_function["validate"]["apply=global"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
            for tensor_name in global_model_dict_val
        ]
        self.required_tensorkeys_for_function["validate"]["apply=global"] += [
            TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
            for tensor_name in local_model_dict_val
        ]

    def load_native(
        self,
        filepath,
        model_state_dict_key="model_state_dict",
        optimizer_state_dict_key="optimizer_state_dict",
        **kwargs,
    ):
        """
        Load model and optimizer states from a pickled file specified by \
        filepath. model_/optimizer_state_dict args can be specified if needed. \
        Uses pt.load().

        Args:
            filepath (str): Path to pickle file created by pt.save().
            model_state_dict_key (str, optional): Key for model state dict in
                pickled file. Defaults to 'model_state_dict'.
            optimizer_state_dict_key (str, optional): Key for optimizer state
                dict in picked file. Defaults to 'optimizer_state_dict'.
            **kwargs: Additional keyword arguments.
        """
        pickle_dict = pt.load(filepath, weights_only=True)  # nosec B614

        if model_state_dict_key in pickle_dict and optimizer_state_dict_key in pickle_dict:
            self.load_state_dict(pickle_dict[model_state_dict_key])
            self.optimizer.load_state_dict(pickle_dict[optimizer_state_dict_key])
        else:
            self.load_state_dict(pickle_dict)

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
            filepath (str): Path to pickle file to be created by pt.save().
            model_state_dict_key (str, optional): Key for model state dict in
                pickled file. Defaults to 'model_state_dict'.
            optimizer_state_dict_key (str, optional): Key for optimizer state
                dict in picked file. Defaults to 'optimizer_state_dict'.
            **kwargs: Additional keyword arguments.
        """
        pickle_dict = {
            model_state_dict_key: self.model.state_dict(),
            optimizer_state_dict_key: self.optimizer.state_dict(),
        }
        pt.save(pickle_dict, filepath)

    def reset_opt_vars(self):
        """Reset optimizer variables."""
        pass


def create_tensorkey_dicts(
    tensor_dict,
    metric_dict,
    col_name,
    round_num,
    tensor_dict_split_fn_kwargs,
):
    """Create dictionaries of TensorKeys for global and local tensors.

    Args:
        tensor_dict (dict): Dictionary of tensors.
        metric_dict (dict): Dictionary of metrics.
        col_name (str): Name of the collaborator.
        round_num (int): Current round number.
        tensor_dict_split_fn_kwargs (dict): Keyword arguments for the tensor
            dict split function.

    Returns:
        global_tensor_dict (dict): Dictionary of global TensorKeys.
        local_tensor_dict (dict): Dictionary of local TensorKeys.
    """
    origin = col_name
    tags = ("trained",)
    output_metric_dict = {}
    for k, v in metric_dict.items():
        tk = TensorKey(k, origin, round_num, True, ("metric",))
        output_metric_dict[tk] = np.array(v)

    global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
        tensor_dict, **tensor_dict_split_fn_kwargs
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

    return global_tensor_dict, local_tensor_dict


def set_pt_model_from_tensor_dict(model, tensor_dict, device, with_opt_vars=False):
    """Set the tensor dictionary for the PyTorch model.

    Args:
        model (Model): The PyTorch model.
        tensor_dict (dict): Tensor dictionary.
        device (str): Device for the model.
        with_opt_vars (bool, optional): Include the optimizer tensors.
            Defaults to False.
    """
    # Sets tensors for model layers and optimizer state.
    # FIXME: model.parameters() instead? Unclear if load_state_dict() or
    #  simple assignment is better
    # for now, state dict gives us names, which is good
    # FIXME: do both and sanity check each time?

    new_state = {}
    # Grabbing keys from model's state_dict helps to confirm we have
    # everything
    for k in model.state_dict():
        new_state[k] = pt.from_numpy(tensor_dict.pop(k)).to(device)

    # set model state
    model.load_state_dict(new_state)

    if with_opt_vars:
        # see if there is state to restore first
        if tensor_dict.pop("__opt_state_needed") == "true":
            _set_optimizer_state(model.get_optimizer(), device, tensor_dict)

        # sanity check that we did not record any state that was not used
        assert len(tensor_dict) == 0


def _derive_opt_state_dict(opt_state_dict):
    """Separate optimizer tensors from the tensor dictionary.

    Flattens the optimizer state dict so as to have key, value pairs with
    values as numpy arrays.
    The keys have sufficient info to restore opt_state_dict using
    expand_derived_opt_state_dict.

    Args:
        opt_state_dict (dict): Optimizer state dictionary.

    Returns:
        derived_opt_state_dict (dict): Optimizer state dictionary.
    """
    derived_opt_state_dict = {}

    # Determine if state is needed for this optimizer.
    if len(opt_state_dict["state"]) == 0:
        derived_opt_state_dict["__opt_state_needed"] = "false"
        return derived_opt_state_dict

    derived_opt_state_dict["__opt_state_needed"] = "true"

    # Using one example state key, we collect keys for the corresponding
    # dictionary value.
    example_state_key = opt_state_dict["param_groups"][0]["params"][0]
    example_state_subkeys = set(opt_state_dict["state"][example_state_key].keys())

    # We assume that the state collected for all params in all param groups is
    # the same.
    # We also assume that whether or not the associated values to these state
    # subkeys is a tensor depends only on the subkey.
    # Using assert statements to break the routine if these assumptions are
    # incorrect.
    for state_key in opt_state_dict["state"].keys():
        assert example_state_subkeys == set(opt_state_dict["state"][state_key].keys())
        for state_subkey in example_state_subkeys:
            assert isinstance(
                opt_state_dict["state"][example_state_key][state_subkey],
                pt.Tensor,
            ) == isinstance(opt_state_dict["state"][state_key][state_subkey], pt.Tensor)

    state_subkeys = list(opt_state_dict["state"][example_state_key].keys())

    # Tags will record whether the value associated to the subkey is a
    # tensor or not.
    state_subkey_tags = []
    for state_subkey in state_subkeys:
        if isinstance(opt_state_dict["state"][example_state_key][state_subkey], pt.Tensor):
            state_subkey_tags.append("istensor")
        else:
            state_subkey_tags.append("")
    state_subkeys_and_tags = list(zip(state_subkeys, state_subkey_tags))

    # Forming the flattened dict, using a concatenation of group index,
    # subindex, tag, and subkey inserted into the flattened dict key -
    # needed for reconstruction.
    nb_params_per_group = []
    for group_idx, group in enumerate(opt_state_dict["param_groups"]):
        for idx, param_id in enumerate(group["params"]):
            for subkey, tag in state_subkeys_and_tags:
                if tag == "istensor":
                    new_v = opt_state_dict["state"][param_id][subkey].cpu().numpy()
                else:
                    new_v = np.array([opt_state_dict["state"][param_id][subkey]])
                derived_opt_state_dict[f"__opt_state_{group_idx}_{idx}_{tag}_{subkey}"] = new_v
        nb_params_per_group.append(idx + 1)
    # group lengths are also helpful for reconstructing
    # original opt_state_dict structure
    derived_opt_state_dict["__opt_group_lengths"] = np.array(nb_params_per_group)

    return derived_opt_state_dict


def expand_derived_opt_state_dict(derived_opt_state_dict, device):
    """Expand the optimizer state dictionary.

    Takes a derived opt_state_dict and creates an opt_state_dict suitable as
    input for load_state_dict for restoring optimizer state.
    Reconstructing state_subkeys_and_tags using the example key prefix,
    "__opt_state_0_0_", certain to be present.

    Args:
        derived_opt_state_dict (dict): Derived optimizer state dictionary.
        device (str): Device for the model.

    Returns:
        opt_state_dict (dict): Expanded optimizer state dictionary.
    """
    state_subkeys_and_tags = []
    for key in derived_opt_state_dict:
        if key.startswith("__opt_state_0_0_"):
            stripped_key = key[16:]
            if stripped_key.startswith("istensor_"):
                this_tag = "istensor"
                subkey = stripped_key[9:]
            else:
                this_tag = ""
                subkey = stripped_key[1:]
            state_subkeys_and_tags.append((subkey, this_tag))

    opt_state_dict = {"param_groups": [], "state": {}}
    nb_params_per_group = list(derived_opt_state_dict.pop("__opt_group_lengths").astype(np.int32))

    # Construct the expanded dict.
    for group_idx, nb_params in enumerate(nb_params_per_group):
        these_group_ids = [f"{group_idx}_{idx}" for idx in range(nb_params)]
        opt_state_dict["param_groups"].append({"params": these_group_ids})
        for this_id in these_group_ids:
            opt_state_dict["state"][this_id] = {}
            for subkey, tag in state_subkeys_and_tags:
                flat_key = f"__opt_state_{this_id}_{tag}_{subkey}"
                if tag == "istensor":
                    new_v = pt.from_numpy(derived_opt_state_dict.pop(flat_key))
                else:
                    # Here (for currently supported optimizers) the subkey
                    # should be 'step' and the length of array should be one.
                    assert subkey == "step"
                    assert len(derived_opt_state_dict[flat_key]) == 1
                    new_v = int(derived_opt_state_dict.pop(flat_key))
                opt_state_dict["state"][this_id][subkey] = new_v

    # sanity check that we did not miss any optimizer state
    assert len(derived_opt_state_dict) == 0

    return opt_state_dict


def _get_optimizer_state(optimizer):
    """Get the state of the optimizer.

    Args:
        optimizer (Optimizer): Optimizer.

    Returns:
        derived_opt_state_dict (dict): State of the optimizer.
    """
    opt_state_dict = deepcopy(optimizer.state_dict())

    # Optimizer state might not have some parts representing frozen parameters
    # So we do not synchronize them
    param_keys_with_state = set(opt_state_dict["state"].keys())
    for group in opt_state_dict["param_groups"]:
        local_param_set = set(group["params"])
        params_to_sync = local_param_set & param_keys_with_state
        group["params"] = sorted(params_to_sync)

    derived_opt_state_dict = _derive_opt_state_dict(opt_state_dict)

    return derived_opt_state_dict


def _set_optimizer_state(optimizer, device, derived_opt_state_dict):
    """Set the state of the optimizer.

    Args:
        optimizer (Optimizer): Optimizer.
        device (str): Device for the model.
        derived_opt_state_dict (dict): Derived optimizer state dictionary.
    """
    temp_state_dict = expand_derived_opt_state_dict(derived_opt_state_dict, device)

    # FIXME: Figure out whether or not this breaks learning rate
    #  scheduling and the like.
    # Setting default values.
    # All optimizer.defaults are considered as not changing over course of
    # training.
    for group in temp_state_dict["param_groups"]:
        for k, v in optimizer.defaults.items():
            group[k] = v

    optimizer.load_state_dict(temp_state_dict)


def to_cpu_numpy(state):
    """Convert state to CPU as Numpy array.

    Args:
        state (State): State to be converted.

    Returns:
        state (dict): State as Numpy array.
    """
    # deep copy so as to decouple from active model
    state = deepcopy(state)

    for k, v in state.items():
        # When restoring, we currently assume all values are tensors.
        if not pt.is_tensor(v):
            raise ValueError(
                "We do not currently support non-tensors coming from model.state_dict()"
            )
        # get as a numpy array, making sure is on cpu
        state[k] = v.cpu().numpy()
    return state
