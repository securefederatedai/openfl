# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""FederatedModel module."""
import inspect

from openfl.federated.task.runner import TaskRunner


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

    def __init__(self, build_model, optimizer=None, loss_fn=None, **kwargs):
        """Initialize.

        Args:
            model:    build_model function
            **kwargs: Additional parameters to pass to the function

        """
        super().__init__(**kwargs)

        self.build_model = build_model
        self.lambda_opt = None
        # TODO pass params to model
        if inspect.isclass(build_model):
            self.model = build_model()
            from openfl.federated.task.runner_pt import PyTorchTaskRunner  # noqa: E501

            if optimizer is not None:
                self.optimizer = optimizer(self.model.parameters())
            self.runner = PyTorchTaskRunner(**kwargs)
            if hasattr(self.model, "forward"):
                self.runner.forward = self.model.forward
        else:
            self.model = self.build_model(self.feature_shape, self.data_loader.num_classes)
            from openfl.federated.task.runner_keras import KerasTaskRunner  # noqa: E501

            self.runner = KerasTaskRunner(**kwargs)
            self.optimizer = self.model.optimizer
        self.lambda_opt = optimizer
        if hasattr(self.model, "validate"):
            self.runner.validate = lambda *args, **kwargs: build_model.validate(
                self.runner, *args, **kwargs
            )
        if hasattr(self.model, "train_epoch"):
            self.runner.train_epoch = lambda *args, **kwargs: build_model.train_epoch(
                self.runner, *args, **kwargs
            )
        self.runner.model = self.model
        self.runner.optimizer = self.optimizer
        self.loss_fn = loss_fn
        self.runner.loss_fn = self.loss_fn
        self.tensor_dict_split_fn_kwargs = self.runner.tensor_dict_split_fn_kwargs
        self.initialize_tensorkeys_for_functions()

    def __getattribute__(self, attr):
        """Direct call into self.runner methods if necessary."""

        if attr in [
            "reset_opt_vars",
            "initialize_globals",
            "set_tensor_dict",
            "get_tensor_dict",
            "get_required_tensorkeys_for_function",
            "initialize_tensorkeys_for_functions",
            "save_native",
            "load_native",
            "rebuild_model",
            "set_optimizer_treatment",
            "train",
            "train_batches",
            "validate",
            "validate_task",
            "train_task",
        ]:
            return self.runner.__getattribute__(attr)
        return super().__getattribute__(attr)

    def setup(self, num_collaborators, **kwargs):
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
                data_loader=data_slice,
                **kwargs
            )
            for data_slice in self.data_loader.split(num_collaborators)
        ]
