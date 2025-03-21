# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
Base classes for Federated Analytics.

You may copy this file as the starting point of your own keras model.
"""

from openfl.federated.task.runner import TaskRunner


class FederatedAnalyticsTaskRunner(TaskRunner):
    """The base class for Federated Analytics Task Runner."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def analysis(self, **kwargs):
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
        pass

    def get_tensor_dict(self, with_opt_vars):
        """Get the weights.

        Args:
            with_opt_vars (bool): Specify if we also want to get the variables
                of the optimizer.

        Returns:
            dict: The weight dictionary {<tensor_name>: <value>}.
        """
        pass

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
        pass

    def reset_opt_vars(self):
        """Reinitialize the optimizer variables.

        Returns:
            None
        """
        pass

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
