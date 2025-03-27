# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
Base classes for Federated Analytics.

You may copy this file as the starting point of your own keras model.
"""

import copy
from warnings import catch_warnings, simplefilter

# import numpy as np

from openfl.federated.task.runner import TaskRunner
from openfl.utilities import Metric, TensorKey, change_tags
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

        self.model = self.build_model((28, 28, 1), 10, **kwargs)

        self.model_tensor_names = []

        # this is a map of all of the required tensors for each of the public
        # functions in FederatedAnalyticsTaskRunner
        self.required_tensorkeys_for_function = {}

        self.initialize_tensorkeys_for_functions()

        self.model.summary(print_fn=logger.info)

    def build_model(self,
                    input_shape,
                    num_classes=10,
                    conv_kernel_size=(4, 4),
                    conv_strides=(2, 2),
                    conv1_channels_out=16,
                    conv2_channels_out=32,
                    final_dense_inputsize=100,
                    **kwargs):
        """
        Define the model architecture.

        Args:
            input_shape (numpy.ndarray): The shape of the data
            num_classes (int): The number of classes of the dataset

        Returns:
            keras.models.Sequential: The model defined in Keras

        """

        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(conv1_channels_out,
                         kernel_size=conv_kernel_size,
                         strides=conv_strides,
                         activation='relu',
                         input_shape=input_shape))

        model.add(keras.layers.Conv2D(conv2_channels_out,
                         kernel_size=conv_kernel_size,
                         strides=conv_strides,
                         activation='relu'))

        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(final_dense_inputsize, activation='relu'))

        model.add(keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        return model

    def analysis(self, col_name, round_num, **kwargs):
        """
        Perform analytics on the provided data.
        This method should be implemented by subclasses to perform specific analysis tasks.
        Args:
            **kwargs: Arbitrary keyword arguments that can be used for analysis.
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        print("analysis is called")
        results = self.analysis_task(**kwargs)
        tags = ("analysis",)
        origin = col_name
        print("results is", results)
        for metric_name, metric_value in results.items():
            print(f"Metric: {metric_name}, Value: {metric_value}")
        output_metric_dict = {
            TensorKey(metric_name, origin, round_num, False, tags): metric_value
            for metric_name, metric_value in results.items()
        }
        print("output_metric_dict", output_metric_dict)

        # # output model tensors (Doesn't include TensorKey)
        # output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        # global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
        #     output_model_dict, **self.tensor_dict_split_fn_kwargs
        # )

        # # create global tensorkeys
        # global_tensorkey_model_dict = {
        #     TensorKey(tensor_name, origin, round_num, False, tags): nparray
        #     for tensor_name, nparray in global_model_dict.items()
        # }
        # # create tensorkeys that should stay local
        # local_tensorkey_model_dict = {
        #     TensorKey(tensor_name, origin, round_num, False, tags): nparray
        #     for tensor_name, nparray in local_model_dict.items()
        # }
        # # the train/validate aggregated function of the next round will look
        # # for the updated model parameters.
        # # this ensures they will be resolved locally
        # # next_local_tensorkey_model_dict = {
        # #     TensorKey(tensor_name, origin, round_num + 1, False, ("model",)): nparray
        # #     for tensor_name, nparray in local_model_dict.items()
        # # }

        # global_tensor_dict = {
        #     **global_tensorkey_model_dict,
        # }
        # local_tensor_dict = {
        #     **local_tensorkey_model_dict,
        # }
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

    def train_batches(self, num_batches=None, use_tqdm=False):
        """Perform the training for a specified number of batches.

        Is expected to perform draws randomly, without
        replacement until data is exhausted. Then data is replaced and
        shuffled and draws continue.

        Args:
            num_batches (int, optional): Number of batches to train. Default
                is None.
            use_tqdm (bool, optional): If True, use tqdm to print a progress
                bar. Default is False.

        Returns:
            dict: {<metric>: <value>}.
        """
        pass

    def validate(self):
        """Run validation.

        Returns:
            dict: {<metric>: <value>}.
        """
        pass

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


    def initialize_globals(self):
        """Initialize all global variables.

        Returns:
            None
        """
        pass

    def load_native(self, filepath, **kwargs):
        """Load model state from a filepath in ML-framework "native" format,
        e.g. PyTorch pickled models.

        May load from multiple files. Other filepaths may be derived from the
        passed filepath, or they may be in the kwargs.

        Args:
            filepath (str): Path to frame-work specific file to load.
                For frameworks that use multiple files, this string must be
                    used to derive the other filepaths.
            **kwargs: Additional parameters to pass to the function. For
                future-proofing.

        Returns:
            None
        """
        pass

    def save_native(self, filepath, **kwargs):
        """Save model state in ML-framework "native" format, e.g. PyTorch
        pickled models.

        May save one file or multiple files, depending on the framework.

        Args:
            filepath (str): If framework stores a single file, this should be
                a single file path. Frameworks that store multiple files may
                need to derive the other paths from this path.
            **kwargs: Additional parameters to pass to the function. For
                future-proofing.

        Returns:
            None
        """
        pass

    # def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
    #     pass
        # self.required_tensorkeys_for_function["train_task"] = []
        # self.required_tensorkeys_for_function["validate_task"] = {}
        # self.required_tensorkeys_for_function["validate_task"]["apply=local"] = []
        # self.required_tensorkeys_for_function["validate_task"]["apply=global"] = []
        # self.required_tensorkeys_for_function["validate_task"]["apply=global"] += []

    def get_tensor_dict(self, with_opt_vars, suffix=""):
        """
        Get the model weights as a tensor dictionary.

        Args:
            with_opt_vars (bool): If we should include the optimizer's status.
            suffix (str): Universally.

        Returns:
            model_weights (dict): The tensor dictionary.
        """
        print("inside get_tensor_dict")
        # print(self.model)
        model_weights = self._get_weights_dict(self.model, suffix)
        # print("model_weights", model_weights)
        if with_opt_vars:
            opt_weights = self._get_weights_dict(self.model.optimizer, suffix)
            # print("opt_weights", opt_weights)
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

    # def update_tensorkeys_for_functions(self):
    #     """Update the required tensors for all publicly accessible methods that
    #     could be called as part of a task.

    #     By default, this is just all of the layers and optimizer of the model.
    #     Custom tensors should be added to this function
    #     """
    #     # TODO complete this function. It is only needed for opt_treatment,
    #     #  and making the model stateless

    #     # Minimal required tensors for train function
    #     model_layer_names = self._get_weights_names(self.model)
    #     opt_names = self._get_weights_names(self.model.optimizer)
    #     tensor_names = model_layer_names + opt_names
    #     logger.debug("Updating model tensor names: %s", tensor_names)
    #     self.required_tensorkeys_for_function["train_task"] = [
    #         TensorKey(tensor_name, "GLOBAL", 0, False, ("model",)) for tensor_name in tensor_names
    #     ]

    #     # Validation may be performed on local or aggregated (global) model,
    #     # so there is an extra lookup dimension for kwargs
    #     self.required_tensorkeys_for_function["validate_task"] = {}
    #     self.required_tensorkeys_for_function["validate_task"]["apply=local"] = [
    #         TensorKey(tensor_name, "LOCAL", 0, False, ("trained",)) for tensor_name in tensor_names
    #     ]
    #     self.required_tensorkeys_for_function["validate_task"]["apply=global"] = [
    #         TensorKey(tensor_name, "GLOBAL", 0, False, ("model",)) for tensor_name in tensor_names
    #     ]

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

        self.required_tensorkeys_for_function["analysis"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
            for tensor_name in global_model_dict
        ]
        self.required_tensorkeys_for_function["analysis"] += [
            TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
            for tensor_name in local_model_dict
        ]

        # # Validation may be performed on local or aggregated (global) model,
        # # so there is an extra lookup dimension for kwargs
        # self.required_tensorkeys_for_function["validate_task"] = {}
        # # TODO This is not stateless. The optimizer will not be
        # self.required_tensorkeys_for_function["validate_task"]["apply=local"] = [
        #     TensorKey(tensor_name, "LOCAL", 0, False, ("trained",))
        #     for tensor_name in {**global_model_dict_val, **local_model_dict_val}
        # ]
        # self.required_tensorkeys_for_function["validate_task"]["apply=global"] = [
        #     TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
        #     for tensor_name in global_model_dict_val
        # ]
        # self.required_tensorkeys_for_function["validate_task"]["apply=global"] += [
        #     TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
        #     for tensor_name in local_model_dict_val
        # ]
