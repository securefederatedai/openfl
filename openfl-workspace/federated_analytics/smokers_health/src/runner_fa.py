# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
Base classes for Federated Analytics.

This file can serve as a template for creating your own Federated Analytics experiments.
"""

from openfl.federated.task.runner import TaskRunner
from openfl.utilities import TensorKey
from openfl.utilities.split import split_tensor_dict_for_holdouts

import logging
import numpy as np

logger = logging.getLogger(__name__)


class FederatedAnalyticsTaskRunner(TaskRunner):
    """The base class for Federated Analytics Task Runner."""

    def __init__(self, **kwargs):
        """Initializes the FederatedAnalyticsTaskRunner instance.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        # Dummy model initialization. Dummy models and weights are used here as placeholders
        # to ensure compatibility with the core OpenFL framework, which currently assumes
        # the presence of a model for federated learning tasks.
        #
        # This approach is necessary to support Federated Analytics use cases, which do not
        # involve traditional model training, until OpenFL is refactored to accommodate
        # broader use cases beyond learning.
        #
        # For more details, refer to the discussion at:
        # https://github.com/securefederatedai/openfl/discussions/1385#discussioncomment-13009961.
        self.model = None

        self.model_tensor_names = []
        self.required_tensorkeys_for_function = {}

    def analytics(self, col_name, round_num, **kwargs):
        """
        Return analytics result as tensors.

        Args:
            col_name (str): collaborator name.
            round_num (int): The current round number.
            **kwargs: Additional parameters for analysis.

        Returns:
            dict: A dictionary of analysis results.
        """
        results = self.analytics_task(**kwargs)
        tags = ("analytics",)
        origin = col_name
        output_metric_dict = {
            # TensorKey(metric_name, origin, round_num, False, tags): metric_value
            TensorKey(metric_name, origin, round_num, False, tags): np.array(metric_value) if not isinstance(metric_value, np.ndarray) else metric_value
            for metric_name, metric_value in results.items()
        }
        return output_metric_dict, output_metric_dict

    def analytics_task(self, **kwargs):
        """
        Perform analytics on the provided data.
        This method should be implemented by subclasses to perform specific analysis tasks.
        Args:
            **kwargs: Arbitrary keyword arguments that can be used for analysis.
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def get_tensor_dict(self, with_opt_vars, suffix=""):
        """
        Get the model weights as a tensor dictionary.

        Args:
            with_opt_vars (bool): If we should include the optimizer's status.
            suffix (str): Universally.

        Returns:
            model_weights (dict): The tensor dictionary.
        """
        return {'dummy_tensor': np.float32(1)}

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
        return []

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """
        This function is not required and is kept for compatibility.

        Returns:
            None
        """
        pass
