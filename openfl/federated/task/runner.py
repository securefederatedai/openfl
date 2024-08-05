# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
Mixin class for FL models. No default implementation.

Each framework will likely have its own baseclass implementation (e.g.
TensorflowTaskRunner) that uses this mixin.

You may copy use this file or the appropriate framework-specific base-class to
port your own models.
"""

from logging import getLogger


class TaskRunner:
    """Federated Learning Task Runner Class.

    Attributes:
        data_loader: The data_loader object.
        tensor_dict_split_fn_kwargs (dict): Key word arguments for determining
            which parameters to hold out from aggregation.
        logger (logging.Logger): Logger object for logging events.
        opt_treatment (str): Treatment of current instance optimizer.
    """

    def __init__(self, data_loader, tensor_dict_split_fn_kwargs: dict = None, **kwargs):
        """Intializes the TaskRunner object.

        Args:
            data_loader: The data_loader object
            tensor_dict_split_fn_kwargs (dict, optional): Key word arguments
                for determining which parameters to hold out from aggregation.
                Default is None.
            **kwargs: Additional parameters to pass to the function.
        """
        self.data_loader = data_loader
        self.feature_shape = self.data_loader.get_feature_shape()
        # TODO: Should this comment a path of the doc string?
        # key word arguments for determining which parameters to hold out from
        # aggregation.
        # If set to none, an empty dict will be passed, currently resulting in
        # the defaults:
        # be held out
        # holdout_tensor_names=[]                   # NOQA:E800
        # params with these names will be held out  # NOQA:E800
        # TODO: params are restored from protobufs as float32 numpy arrays, so
        # non-floats arrays and non-arrays are not currently supported for
        # passing to and from protobuf (and as a result for aggregation) - for
        # such params in current examples, aggregation does not make sense
        # anyway, but if this changes support should be added.
        if tensor_dict_split_fn_kwargs is None:
            tensor_dict_split_fn_kwargs = {}
        self.tensor_dict_split_fn_kwargs = tensor_dict_split_fn_kwargs
        self.set_logger()

    def set_logger(self):
        """Set up the log object.

        Returns:
            None
        """
        self.logger = getLogger(__name__)

    def set_optimizer_treatment(self, opt_treatment):
        """Change the treatment of current instance optimizer.

        Args:
            opt_treatment (str): The optimizer treatment.

        Returns:
            None
        """
        self.opt_treatment = opt_treatment

    def get_data_loader(self):
        """Get the data_loader object.

        Serves up batches and provides info regarding data_loader.

        Returns:
            data_loader object
        """
        return self.data_loader

    def set_data_loader(self, data_loader):
        """Set data_loader object.

        Args:
            data_loader: data_loader object to set.

        Returns:
            None
        """
        if data_loader.get_feature_shape() != self.data_loader.get_feature_shape():
            raise ValueError("The data_loader feature shape is not compatible with model.")

        self.data_loader = data_loader

    def get_train_data_size(self):
        """Get the number of training examples.

        It will be used for weighted averaging in aggregation.

        Returns:
            int: The number of training examples.
        """
        return self.data_loader.get_train_data_size()

    def get_valid_data_size(self):
        """Get the number of examples.

        It will be used for weighted averaging in aggregation.

        Returns:
            int: The number of validation examples.
        """
        return self.data_loader.get_valid_data_size()

    def train_batches(self, num_batches=None, use_tqdm=False):
        """Perform the training for a specified number of batches.

        Is expected to perform draws randomly, without
        replacement until data is exausted. Then data is replaced and
        shuffled and draws continue.

        Args:
            num_batches (int, optional): Number of batches to train. Default
                is None.
            use_tqdm (bool, optional): If True, use tqdm to print a progress
                bar. Default is False.

        Returns:
            dict: {<metric>: <value>}.
        """
        raise NotImplementedError

    def validate(self):
        """Run validation.

        Returns:
            dict: {<metric>: <value>}.
        """
        raise NotImplementedError

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """When running a task, a map of named tensorkeys must be provided to
        the function as dependencies.

        Args:
            func_name (str): The function name.
            **kwargs: Additional parameters to pass to the function.

        Returns:
            list: List of required TensorKey. (TensorKey(tensor_name, origin,
                round_number))
        """
        raise NotImplementedError

    def get_tensor_dict(self, with_opt_vars):
        """Get the weights.

        Args:
            with_opt_vars (bool): Specify if we also want to get the variables
                of the optimizer.

        Returns:
            dict: The weight dictionary {<tensor_name>: <value>}.
        """
        raise NotImplementedError

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """Set the model weights with a tensor dictionary:
        {<tensor_name>: <value>}.

        Args:
            tensor_dict (dict): The model weights dictionary.
            with_opt_vars (bool): Specify if we also want to set the variables
                of the optimizer.

        Returns:
            None
        """
        raise NotImplementedError

    def reset_opt_vars(self):
        """Reinitialize the optimizer variables.

        Returns:
            None
        """
        raise NotImplementedError

    def initialize_globals(self):
        """Initialize all global variables.

        Returns:
            None
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
