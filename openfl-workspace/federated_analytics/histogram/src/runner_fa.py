# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
Base classes for Federated Analytics.

This file can serve as a template for creating your own Federated Analytics experiments.
"""

from openfl.federated.task.runner import TaskRunner
from openfl.utilities import TensorKey
from openfl.utilities.split import split_tensor_dict_for_holdouts

import logging

logger = logging.getLogger(__name__)


class FederatedAnalyticsTaskRunner(TaskRunner):
    """The base class for Federated Analytics Task Runner."""

    def __init__(self, **kwargs):
        """Initializes the FederatedAnalyticsTaskRunner instance.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        # Dummy model initialization
        self.model = None

        self.model_tensor_names = []
        self.required_tensorkeys_for_function = {}
        self.initialize_tensorkeys_for_functions()

    def analytics(self, col_name, round_num, **kwargs):
        """
        Return analytics result as tensors.

        Args:
            col_name (str): The column name for the analysis.
            round_num (int): The current round number.
            **kwargs: Additional parameters for analysis.

        Returns:
            dict: A dictionary of analysis results.
        """
        results = self.analytics_task(**kwargs)
        tags = ("analytics",)
        origin = col_name
        output_metric_dict = {
            TensorKey(metric_name, origin, round_num, False, tags): metric_value
            for metric_name, metric_value in results.items()
        }
        return output_metric_dict, output_metric_dict

    def analysis_task(self, **kwargs):
        """
        Perform analytics on the provided data.
        This method should be implemented by subclasses to perform specific analysis tasks.
        Args:
            **kwargs: Arbitrary keyword arguments that can be used for analysis.
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

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
        return {}

    @staticmethod
    def _get_weights_names(obj):
        """Get the list of weight names.

        Args:
            obj (Model or Optimizer): The target object that we want to get
                the weights.

        Returns:
            weight_names (list): The weight name list.
        """
        return []

    def get_tensor_dict(self, with_opt_vars, suffix=""):
        """
        Get the model weights as a tensor dictionary.

        Args:
            with_opt_vars (bool): If we should include the optimizer's status.
            suffix (str): Universally.

        Returns:
            model_weights (dict): The tensor dictionary.
        """
        return self._get_weights_dict(self.model, suffix)

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """Get the required tensors for specified function that could be called
        as part of a task.

        By default, this is just all of the layers and optimizer of the dummy model.

        Args:
            func_name (str): The function name.
            **kwargs: Any function arguments.

        Returns:
            list: List of TensorKey objects.
        """
        return self.required_tensorkeys_for_function[func_name]

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Set the required tensors for all publicly accessible methods that
        could be called as part of a task.

        By default, this is just all of the layers and optimizer of the dummy model.
        Custom tensors should be added to this function

        Args:
            with_opt_vars (bool, optional): If True, include the optimizer's
                status. Defaults to False.
        """

        output_model_dict = self.get_tensor_dict(with_opt_vars=with_opt_vars)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            output_model_dict, **self.tensor_dict_split_fn_kwargs
        )

        self.required_tensorkeys_for_function["analytics"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
            for tensor_name in global_model_dict
        ]
        self.required_tensorkeys_for_function["analytics"] += [
            TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
            for tensor_name in local_model_dict
        ]
