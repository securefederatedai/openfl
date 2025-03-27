# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
Base classes for Federated Analytics.

You may copy this file as the starting point of your own keras model.
"""

import copy
from warnings import catch_warnings, simplefilter

from openfl.federated.task.runner import TaskRunner
from openfl.utilities import TensorKey
from openfl.utilities.split import split_tensor_dict_for_holdouts

with catch_warnings():
    simplefilter(action="ignore")
    import keras

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
        self.model = self.build_model((28, 28, 1), 10, **kwargs)

        self.model_tensor_names = []
        self.required_tensorkeys_for_function = {}
        self.initialize_tensorkeys_for_functions()

    def build_model(
        self,
        input_shape,
        num_classes=10,
        conv_kernel_size=(4, 4),
        conv_strides=(2, 2),
        conv1_channels_out=16,
        conv2_channels_out=32,
        final_dense_inputsize=100,
        **kwargs,
    ):
        """
        Define the model architecture.

        Args:
            input_shape (numpy.ndarray): The shape of the data
            num_classes (int): The number of classes of the dataset

        Returns:
            keras.models.Sequential: The model defined in Keras

        """

        model = keras.models.Sequential()

        model.add(
            keras.layers.Conv2D(
                conv1_channels_out,
                kernel_size=conv_kernel_size,
                strides=conv_strides,
                activation="relu",
                input_shape=input_shape,
            )
        )

        model.add(
            keras.layers.Conv2D(
                conv2_channels_out,
                kernel_size=conv_kernel_size,
                strides=conv_strides,
                activation="relu",
            )
        )

        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(final_dense_inputsize, activation="relu"))

        model.add(keras.layers.Dense(num_classes, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def analysis(self, col_name, round_num, **kwargs):
        """
        Return analytics result as tensors.

        Args:
            col_name (str): The column name for the analysis.
            round_num (int): The current round number.
            **kwargs: Additional parameters for analysis.

        Returns:
            dict: A dictionary of analysis results.
        """
        results = self.analysis_task(**kwargs)
        tags = ("analysis",)
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
        weights_dict = {}
        weight_names = FederatedAnalyticsTaskRunner._get_weights_names(obj)
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
        return self.required_tensorkeys_for_function[func_name]

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Set the required tensors for all publicly accessible methods that
        could be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function

        Args:
            with_opt_vars (bool, optional): If True, include the optimizer's
                status. Defaults to False.
        """

        output_model_dict = self.get_tensor_dict(with_opt_vars=with_opt_vars)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            output_model_dict, **self.tensor_dict_split_fn_kwargs
        )

        self.required_tensorkeys_for_function["analysis"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
            for tensor_name in global_model_dict
        ]
        self.required_tensorkeys_for_function["analysis"] += [
            TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
            for tensor_name in local_model_dict
        ]
