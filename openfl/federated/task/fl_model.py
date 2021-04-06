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
    def __init__(self, model_lambda, optimizer_lambda=None, loss_fn=None, device=None, **kwargs):
        """Initialize.

        Args:
            model:    build_model function
            **kwargs: Additional parameters to pass to the function

        """
        super().__init__(**kwargs)
        self.model_lambda = model_lambda
        self.optimizer_lambda = optimizer_lambda
        self.loss_fn = loss_fn
        self.device = device
        self.kwargs = kwargs
        # TODO pass params to model

    def __getattribute__(self, attr):
        """Direct call into self.runner methods if necessary."""
        if attr in ['reset_opt_vars', 'initialize_globals',
                    'set_tensor_dict', 'get_tensor_dict',
                    'get_required_tensorkeys_for_function',
                    'initialize_tensorkeys_for_functions',
                    'save_native', 'load_native', 'rebuild_model',
                    'set_optimizer_treatment',
                    'train', 'train_batches', 'validate'] and hasattr(self, 'runner'):
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
                self.model_lambda,
                optimizer_lambda=self.optimizer_lambda,
                loss_fn=self.loss_fn,
                device=self.device,
                data_loader=data_slice,
                **kwargs
            )
            for data_slice in self.data_loader.split(
                num_collaborators, shuffle=shuffle, equally=equally
            )]

    def initialize(self, **kwargs):
        if inspect.isclass(self.model_lambda):
            
            self.model = self.model_lambda()
            
            from .runner_pt import PyTorchTaskRunner
            if self.device:
                self.kwargs.update({'device': self.device})
            if self.optimizer_lambda is not None:
                self.optimizer = self.optimizer_lambda(self.model.parameters())
            # build_model.__init__()
            self.runner = PyTorchTaskRunner(**self.kwargs)
            if hasattr(self.model, 'forward'):
                self.runner.forward = self.model.forward
            self.runner.model = self.model
        else:
            (self.model, session) = self.model_lambda(
                self.feature_shape, self.data_loader.num_classes)
            from .runner_keras import KerasTaskRunner
            self.runner = KerasTaskRunner(self.model, session, **self.kwargs)
            self.optimizer = self.model.optimizer
        if hasattr(self.model, 'validate'):
            self.runner.validate = lambda *args, **kwargs: self.build_model.validate(
                self.runner, *args, **kwargs)
        self.runner.optimizer = self.optimizer
        self.runner.loss_fn = self.loss_fn
        self.tensor_dict_split_fn_kwargs = \
            self.runner.tensor_dict_split_fn_kwargs
        self.initialize_tensorkeys_for_functions()
        super().initialize(**kwargs)
