# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
Base classes for developing a keras.Model() Federated Learning model.

You may copy this file as the starting point of your own keras model.
"""

import copy
from warnings import catch_warnings, simplefilter

import numpy as np

from openfl.federated.task.runner import TaskRunner
from openfl.utilities import Metric, TensorKey, change_tags
from openfl.utilities.split import split_tensor_dict_for_holdouts

with catch_warnings():
    simplefilter(action="ignore")
    import keras

import logging

logger = logging.getLogger(__name__)


class KerasTaskRunner(TaskRunner):
    """The base model for Keras models in the federation.

    Attributes:
        model (keras.Model): The Keras model.
        model_tensor_names (list): List of model tensor names.
        required_tensorkeys_for_function (dict): A map of all of the required
            tensors for each of the public functions in KerasTaskRunner.
    """

    def __init__(self, **kwargs):
        """Initializes the KerasTaskRunner instance.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        self.model = keras.models.Model()

        self.model_tensor_names = []

        # this is a map of all of the required tensors for each of the public
        # functions in KerasTaskRunner
        self.required_tensorkeys_for_function = {}

    def rebuild_model(self, round_num, input_tensor_dict, validation=False):
        """Parse tensor names and update weights of model. Handles the
        optimizer treatment.

        Args:
            round_num (int): The round number.
            input_tensor_dict (dict): The input tensor dictionary.
            validation (bool, optional): If True, validate the model. Defaults
                to False.
        """
        if self.opt_treatment == "RESET":
            self.reset_opt_vars()
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)
        elif round_num > 0 and self.opt_treatment == "CONTINUE_GLOBAL" and not validation:
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=True)
        else:
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)

    def train_task(
        self,
        col_name,
        round_num,
        input_tensor_dict,
        metrics,
        epochs=1,
        batch_size=1,
        **kwargs,
    ):
        """
        Perform the training. Is expected to perform draws randomly, without
        replacement until data is exhausted. Then data is replaced and shuffled
        and draws continue.

        Args:
            col_name (str): The collaborator name.
            round_num (int): The round number.
            input_tensor_dict (dict): The input tensor dictionary.
            metrics (list): List of metrics.
            epochs (int, optional): Number of epochs to train. Defaults to 1.
            batch_size (int, optional): Batch size. Defaults to 1.
            **kwargs: Additional parameters.

        Returns:
            global_tensor_dict (dict): Dictionary of 'TensorKey: nparray'.
            local_tensor_dict (dict): Dictionary of 'TensorKey: nparray'.
        """
        if metrics is None:
            raise KeyError("metrics must be defined")

        # rebuild model with updated weights
        self.rebuild_model(round_num, input_tensor_dict)
        for epoch in range(epochs):
            logger.info("Run %s epoch of %s round", epoch, round_num)
            results = self.train_(
                self.data_loader.get_train_loader(batch_size),
                metrics=metrics,
                **kwargs,
            )

        # output metric tensors (scalar)
        origin = col_name
        tags = ("trained",)
        output_metric_dict = {
            TensorKey(metric_name, origin, round_num, True, ("metric",)): metric_value
            for (metric_name, metric_value) in results
        }

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            output_model_dict, **self.tensor_dict_split_fn_kwargs
        )

        # create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags): nparray
            for tensor_name, nparray in global_model_dict.items()
        }
        # create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags): nparray
            for tensor_name, nparray in local_model_dict.items()
        }
        # the train/validate aggregated function of the next round will look
        # for the updated model parameters.
        # this ensures they will be resolved locally
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

        # update the required tensors if they need to be pulled from the
        # aggregator
        # TODO this logic can break if different collaborators have different
        # roles between rounds.
        # for example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator because
        # these are only created after training occurs. A work around could
        # involve doing a single epoch of training on random data to get the
        # optimizer names, and then throwing away the model.
        if self.opt_treatment == "CONTINUE_GLOBAL":
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        self.update_tensorkeys_for_functions()
        return global_tensor_dict, local_tensor_dict

    def train_(self, batch_generator, metrics: list = None, **kwargs):
        """Train single epoch. Override this function for custom training.

        Args:
            batch_generator (generator): Generator of training batches.
            metrics (list, optional): Names of metrics to save. Defaults to
                None.
            **kwargs: Additional parameters.

        Returns:
            results (list): List of Metric objects.
        """
        if metrics is None:
            metrics = []
        # TODO Currently assuming that all metrics are defined at
        #  initialization (build_model).
        #  If metrics are added (i.e. not a subset of what was originally
        #  defined) then the model must be recompiled.
        try:
            results = self.model.get_metrics_result()
        except ValueError:
            if "batch_size" in kwargs:
                batch_size = kwargs["batch_size"]
            else:
                batch_size = 1
            # evaluation needed before metrics can be resolved
            self.model.evaluate(self.data_loader.get_valid_loader(batch_size), verbose=1)
            results = self.model.get_metrics_result()

        # TODO if there are new metrics in the flplan that were not included
        #  in the originally
        #  compiled model, that behavior is not currently handled.
        for param in metrics:
            if param not in results:
                raise ValueError(
                    f"KerasTaskRunner does not support specifying new metrics. "
                    f"Param_metrics = {metrics}"
                )

        history = self.model.fit(batch_generator, verbose=2, **kwargs)
        results = []
        for metric in metrics:
            value = np.mean([history.history[metric]])
            results.append(Metric(name=metric, value=np.array(value)))
        return results

    def validate_task(self, col_name, round_num, input_tensor_dict, **kwargs):
        """Run the trained model on validation data; report results.

        Args:
            col_name (str): The collaborator name.
            round_num (int): The round number.
            input_tensor_dict (dict): The input tensor dictionary. Either the
                last aggregated or locally trained model
            **kwargs: Additional parameters.

        Returns:
            output_tensor_dict (dict): Dictionary of 'TensorKey: nparray'.
                These correspond to acc, precision, f1_score, etc.
            dict: Empty dictionary.
        """
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
        else:
            batch_size = 1

        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        param_metrics = kwargs["metrics"]

        self.model.evaluate(self.data_loader.get_valid_loader(batch_size), verbose=1)
        results = self.model.get_metrics_result()

        # TODO if there are new metrics in the flplan that were not included in
        #  the originally compiled model, that behavior is not currently
        #  handled.
        for param in param_metrics:
            if param not in results:
                raise ValueError(
                    f"KerasTaskRunner does not support specifying new metrics. "
                    f"Param_metrics = {param_metrics}"
                )

        origin = col_name
        suffix = "validate"
        if kwargs["apply"] == "local":
            suffix += "_local"
        else:
            suffix += "_agg"
        tags = ("metric",)
        tags = change_tags(tags, add_field=suffix)
        output_tensor_dict = {
            TensorKey(metric, origin, round_num, True, tags): np.array(results[metric])
            for metric in param_metrics
        }

        return output_tensor_dict, {}

    def save_native(self, filepath):
        """Save model.

        Args:
            filepath (str): The file path to save the model.
        """
        self.model.export(filepath)

    def load_native(self, filepath):
        """Load model.

        Args:
            filepath (str): The file path to load the model.
        """
        self.model = keras.models.load_model(filepath)

    @staticmethod
    def _get_weights_names(obj):
        """Get the list of weight names.

        Args:
            obj (Model or Optimizer): The target object that we want to get
                the weights.

        Returns:
            weight_names (list): The weight name list.
        """
        if isinstance(obj, keras.optimizers.Optimizer):
            weight_names = [weight.name for weight in obj.variables]
        else:
            weight_names = [
                layer.name + "/" + weight.name for layer in obj.layers for weight in layer.weights
            ]
        return weight_names

    @staticmethod
    def _get_weights_dict(obj, suffix=""):
        """
        Get the dictionary of weights.

        Args:
            obj (Model or Optimizer): The target object that we want to get
                the weights.
            suffix (str, optional): Suffix for weight names. Defaults to ''.

        Returns:
            weights_dict (dict): The weight dictionary.
        """
        weights_dict = {}
        weight_names = KerasTaskRunner._get_weights_names(obj)
        if isinstance(obj, keras.optimizers.Optimizer):
            weights_dict = {
                weight_names[i] + suffix: weight.numpy()
                for i, weight in enumerate(copy.deepcopy(obj.variables))
            }
        else:
            weight_name_index = 0
            for layer in obj.layers:
                if weight_name_index < len(weight_names) and len(layer.get_weights()) > 0:
                    for weight in layer.get_weights():
                        weights_dict[weight_names[weight_name_index] + suffix] = weight
                        weight_name_index += 1
        return weights_dict

    @staticmethod
    def _set_weights_dict(obj, weights_dict):
        """Set the object weights with a dictionary.

        Args:
            obj (Model or Optimizer): The target object that we want to set
                the weights.
            weights_dict (dict): The weight dictionary.
        """
        weight_names = KerasTaskRunner._get_weights_names(obj)
        weight_values = [weights_dict[name] for name in weight_names]
        obj.set_weights(weight_values)

    def get_tensor_dict(self, with_opt_vars, suffix=""):
        """
        Get the model weights as a tensor dictionary.

        Args:
            with_opt_vars (bool): If we should include the optimizer's status.
            suffix (str): Universally.

        Returns:
            model_weights (dict): The tensor dictionary.
        """
        model_weights = self._get_weights_dict(self.model, suffix)

        if with_opt_vars:
            opt_weights = self._get_weights_dict(self.model.optimizer, suffix)

            model_weights.update(opt_weights)
            if len(opt_weights) == 0:
                logger.debug("WARNING: We didn't find variables for the optimizer.")
        return model_weights

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """Set the model weights with a tensor dictionary.

        Args:
            tensor_dict (dict): The tensor dictionary.
            with_opt_vars (bool): True = include the optimizer's status.
        """
        if with_opt_vars is False:
            # It is possible to pass in opt variables from the input tensor
            # dict. This will make sure that the correct layers are updated
            model_weight_names = self._get_weights_names(self.model)
            model_weights_dict = {name: tensor_dict[name] for name in model_weight_names}
            self._set_weights_dict(self.model, model_weights_dict)
        else:
            model_weight_names = self._get_weights_names(self.model)
            model_weights_dict = {name: tensor_dict[name] for name in model_weight_names}
            opt_weight_names = self._get_weights_names(self.model.optimizer)
            opt_weights_dict = {name: tensor_dict[name] for name in opt_weight_names}
            self._set_weights_dict(self.model, model_weights_dict)
            self._set_weights_dict(self.model.optimizer, opt_weights_dict)

    def reset_opt_vars(self):
        """Resets the optimizer variables."""
        pass

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """Get the required tensors for specified function that could be called
        as part of a task.

        By default, this is just all of the layers and optimizer of the model.

        Args:
            func_name (str): The function name.
            **kwargs: Any function arguments.

        Returns:
            list: List of TensorKey objects.
        """
        if func_name == "validate_task":
            local_model = "apply=" + str(kwargs["apply"])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]

    def update_tensorkeys_for_functions(self):
        """Update the required tensors for all publicly accessible methods that
        could be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function
        """
        # TODO complete this function. It is only needed for opt_treatment,
        #  and making the model stateless

        # Minimal required tensors for train function
        model_layer_names = self._get_weights_names(self.model)
        opt_names = self._get_weights_names(self.model.optimizer)
        tensor_names = model_layer_names + opt_names
        logger.debug("Updating model tensor names: %s", tensor_names)
        self.required_tensorkeys_for_function["train_task"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",)) for tensor_name in tensor_names
        ]

        # Validation may be performed on local or aggregated (global) model,
        # so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function["validate_task"] = {}
        self.required_tensorkeys_for_function["validate_task"]["apply=local"] = [
            TensorKey(tensor_name, "LOCAL", 0, False, ("trained",)) for tensor_name in tensor_names
        ]
        self.required_tensorkeys_for_function["validate_task"]["apply=global"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",)) for tensor_name in tensor_names
        ]

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Set the required tensors for all publicly accessible methods that
        could be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function

        Args:
            with_opt_vars (bool, optional): If True, include the optimizer's
                status. Defaults to False.
        """
        # TODO there should be a way to programmatically iterate through all
        #  of the methods in the class and declare the tensors.
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
