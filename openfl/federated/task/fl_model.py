# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FederatedModel module."""

import inspect

from .runner import TaskRunner


class FederatedModel(TaskRunner):
    """
    A wrapper that adapts to Tensorflow and Pytorch models to a federated context.

    Args:
        model : tensorflow/keras (function) , pytorch (class)
            For keras/tensorflow model, expects a function that returns the
            model definition
            For pytorch models, expects a class (not an instance) containing
            the model definition and forward function
        optimizer : lambda function (only required for pytorch)
            The optimizer should be definied within a lambda function. This
            allows the optimizer to be attached to the federated models spawned
            for each collaborator.
        loss_fn : pytorch loss_fun (only required for pytorch)
    """

    def __init__(self, build_model, optimizer=None, loss_fn=None, device=None, **kwargs):
        """Initialize.

        Args:
            model:    build_model function
            **kwargs: Additional parameters to pass to the function

        """
        super().__init__(**kwargs)
        self.build_model = build_model
        self.device = device
        self.loss_fn = loss_fn
        self.lambda_opt = optimizer
        self.__kwargs = kwargs
        self.__model = None
        self.__optimizer = None
        self.__runner = None

    @property
    def model(self):
        """
        Model lazy property.

        If self.__model exists, return self.__model
        If self.__model does not exist, build model based on build_model param
        """
        # TODO pass params to model
        if not self.__model:
            if inspect.isclass(self.build_model):
                self.__model = self.build_model()
            else:
                self.__model = self.build_model(self.feature_shape, self.data_loader.num_classes)
        return self.__model

    @property
    def optimizer(self):
        """
        Optimizer lazy property.

        If self.__optimizer exists, return self.__optimizer
        If self.__optimizer does not exist, create optimizer based on build_model param
        """
        if not self.__optimizer:
            if inspect.isclass(self.build_model):
                if self.lambda_opt is not None:
                    self.__optimizer = self.lambda_opt(self.model.parameters())
            else:
                self.__optimizer = self.model.optimizer
        return self.__optimizer

    @property
    def runner(self):
        """
        Runner lazy property.

        If self.__runner exists, return self.__runner
        If self.__runner does not exist, create TaskRunner based on build_model param
        """
        if not self.__runner:
            if inspect.isclass(self.build_model):
                from .runner_pt import PyTorchTaskRunner
                if self.device:
                    self.__kwargs.update({'device': self.device})
                self.__runner = PyTorchTaskRunner(**self.__kwargs)
                if hasattr(self.model, 'forward'):
                    self.__runner.forward = self.model.forward
            else:
                from .runner_keras import KerasTaskRunner
                self.__runner = KerasTaskRunner(**self.__kwargs)
            if hasattr(self.model, 'validate'):
                self.__runner.validate = lambda *args, **kwargs: self.build_model.validate(
                    self.__runner, *args, **kwargs)
            if hasattr(self.model, 'train_epoch'):
                self.runner.train_epoch = lambda *args, **kwargs: self.build_model.train_epoch(
                    self.runner, *args, **kwargs)
            self.__runner.loss_fn = self.loss_fn
            self.__runner.model = self.model
            self.__runner.optimizer = self.optimizer
            self.tensor_dict_split_fn_kwargs = self.__runner.tensor_dict_split_fn_kwargs
            self.initialize_tensorkeys_for_functions()
        return self.__runner

    def __getattribute__(self, attr):
        """Direct call into self.runner methods if necessary."""
        if attr in ['reset_opt_vars', 'initialize_globals',
                    'set_tensor_dict', 'get_tensor_dict',
                    'get_required_tensorkeys_for_function',
                    'initialize_tensorkeys_for_functions',
                    'save_native', 'load_native', 'rebuild_model',
                    'set_optimizer_treatment',
                    'train', 'train_batches', 'validate']:
            return self.runner.__getattribute__(attr)
        return super(FederatedModel, self).__getattribute__(attr)

    def setup(self, num_collaborators, shuffle=True, equally=True, **kwargs):
        """
        Create new models for all of the collaborators in the experiment.

        Args:
            num_collaborators:  Number of experiment collaborators

        Returns:
            List of models
        """
        return [
            FederatedModel(
                self.build_model,
                optimizer=self.lambda_opt,
                loss_fn=self.loss_fn,
                device=self.device,
                data_loader=data_slice,
                **kwargs
            )
            for data_slice in self.data_loader.split(
                num_collaborators, shuffle=shuffle, equally=equally
            )]
