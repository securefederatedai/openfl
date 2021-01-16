# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""FastEstimatorTaskRunner module."""

import numpy as np
import tensorflow as tf

from openfl.utilities import TensorKey, split_tensor_dict_for_holdouts

from .runner import TaskRunner
from .runner_keras import KerasTaskRunner
from .runner_pt import PyTorchTaskRunner


class FastEstimatorTaskRunner(TaskRunner):
    """A wrapper for fastestimator.estimator."""

    def __init__(self, estimator, **kwargs):
        """Initialize.

        Args:
            estimator: object of type fastestimator.estimator
        """
        super().__init__(**kwargs)
        import fastestimator as fe

        class ProgressLoader(fe.trace.Trace):
            def __init__(self, get_progress) -> None:
                super().__init__(mode="train")
                self.get_progress = get_progress

            def on_begin(self, data) -> None:
                """Run once at the beginning of training or testing.

                Args:
                    data: A dictionary through which traces can communicate with
                    each other or write values for logging.
                """
                progress = self.get_progress()
                self.system.epoch_idx = progress['epoch_idx']
                self.system.global_step = progress['global_step']

        tf.config.run_functions_eagerly(True)
        estimator_kwargs = {}
        for k, v in estimator.system.__dict__.items():
            if k in ['pipeline', 'network', 'log_steps',
                     'max_train_steps_per_epoch', 'max_eval_steps_per_epoch']:
                estimator_kwargs[k] = v
            if k == 'traces':
                self.logger.debug(f'traces={estimator.system.traces}')
                estimator_kwargs[k] = v + [
                    ProgressLoader(lambda: {
                        'epoch_idx': self.epoch_idx,
                        'global_step': self.global_step
                    })]
        estimator_kwargs.update({
            'epochs': estimator.system.total_epochs,
            'monitor_names': estimator.monitor_names
        })
        self.estimator = fe.Estimator(**estimator_kwargs)
        assert (len(estimator.network.models) == 1), (
            'Only one-model networks are currently supported')
        if isinstance(estimator.network, fe.network.TorchNetwork):
            impl = PyTorchTaskRunner
        elif isinstance(estimator.network, fe.network.TFNetwork):
            impl = KerasTaskRunner
        self.model = self.estimator.network.models[0]
        self.optimizer = self.model.optimizer
        self.runner = impl(**kwargs)
        self.runner.model = self.model
        self.runner.optimizer = self.optimizer
        self.required_tensorkeys_for_function = {}
        self.tensor_dict_split_fn_kwargs = \
            self.runner.tensor_dict_split_fn_kwargs
        self.initialize_tensorkeys_for_functions()
        self.epoch_idx = 0
        self.global_step = None
        self.total_epochs = self.estimator.system.total_epochs

    def train(self, col_name, round_num, input_tensor_dict, epochs, **kwargs):
        """Perform training for a specified number of epochs."""
        if 'metrics' not in kwargs:
            raise KeyError('metrics must be included in kwargs')
        param_metrics = kwargs['metrics']

        self.rebuild_model(round_num, input_tensor_dict)

        # Estimators need to be given an experiment name to produce an output
        # summary
        summary = self.estimator.fit("experiment", warmup=False)
        self.epoch_idx = self.estimator.system.epoch_idx
        self.global_step = self.estimator.system.global_step
        self.estimator.system.total_epochs += self.total_epochs
        # Define what the ouptut is to encapsulate in tensorkeys and return
        # output metric tensors (scalar)
        origin = col_name
        tags = ('trained',)
        output_metric_dict = {
            TensorKey(
                metric, origin, round_num, True, ('metric',)
            ): np.array(
                list(
                    summary.history['train'][metric].values()
                )[-1]) for metric in param_metrics}

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )

        # create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num, False, tags
            ): nparray for tensor_name, nparray in global_model_dict.items()
        }
        # create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num, False, tags
            ): nparray for tensor_name, nparray in local_model_dict.items()
        }
        # the train/validate aggregated function of the next round will look
        # for the updated model parameters.
        # this ensures they will be resolved locally
        next_local_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num + 1, False, ('model',)
            ): nparray for tensor_name, nparray in local_model_dict.items()
        }

        global_tensor_dict = {
            **output_metric_dict,
            **global_tensorkey_model_dict
        }
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict
        }

        # update the required tensors if they need to be pulled from the
        # aggregator
        # TODO this logic can break if different collaborators have different
        #  roles between rounds.
        # for example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator
        # because these are only created after training occurs.
        # A work around could involve doing a single epoch of training
        # on random data to get the optimizer names, and then throwing away
        # the model.
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        # return global_tensor_dict, local_tensor_dict
        return global_tensor_dict, local_tensor_dict

    def validate(self, col_name, round_num, input_tensor_dict, **kwargs):
        """
        Run the trained model on validation data; report results.

        Parameters
        ----------
        input_tensor_dict : either the last aggregated or locally trained model

        Returns
        -------
        output_tensor_dict : {TensorKey: nparray} (these correspond to acc,
         precision, f1_score, etc.)
        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        param_metrics = kwargs['metrics']

        results = self.estimator.test('experiment')
        ret_dict = {
            metric: list(results.history['test'][metric].values())[-1]
            for metric in param_metrics
        }

        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)
        output_tensor_dict = {
            TensorKey(
                metric, origin, round_num, True, tags
            ): np.array(ret_dict[metric])
            for metric in param_metrics
        }

        return output_tensor_dict, {}

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """
        Set the required tensors for all publicly accessible methods that could \
            be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.
         Custom tensors should be added to this function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.runner.initialize_tensorkeys_for_functions(with_opt_vars)

    def build_model(self):
        """Abstract method."""
        raise NotImplementedError

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """
        When running a task, a map of named tensorkeys must be provided to the \
            function as dependencies.

        Returns:
            list: (TensorKey(tensor_name, origin, round_number))
        """
        return self.runner.get_required_tensorkeys_for_function(
            func_name, **kwargs)

    def get_tensor_dict(self, with_opt_vars):
        """
        Get the weights.

        Args:
            with_opt_vars (bool): Specify if we also want to get the variables
             of the optimizer.

        Returns:
            dict: The weight dictionary {<tensor_name>: <value>}
        """
        return self.runner.get_tensor_dict(with_opt_vars)

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """
        Set the model weights with a tensor dictionary: {<tensor_name>: <value>}.

        Args:
            tensor_dict (dict): The model weights dictionary.
            with_opt_vars (bool): Specify if we also want to set the variables
            of the optimizer.

        Returns:
            None
        """
        return self.runner.set_tensor_dict(tensor_dict, with_opt_vars)

    def reset_opt_vars(self):
        """Reinitialize the optimizer variables."""
        return self.runner.reset_opt_vars()

    def initialize_globals(self):
        """
        Initialize all global variables.

        Returns:
            None
        """
        return self.runner.initialize_globals()

    def load_native(self, filepath, **kwargs):
        """
        Load model state from a filepath in ML-framework "native" format, \
            e.g. PyTorch pickled models.

        May load from multiple files. Other filepaths may be derived from the
        passed filepath, or they may be in the kwargs.

        Args:
            filepath (string): Path to frame-work specific file to load. For
                               frameworks that use multiple files, this string
                               must be used to derive the other filepaths.
            kwargs           : For future-proofing

        Returns:
            None
        """
        return self.runner.load_native(filepath, **kwargs)

    def save_native(self, filepath, **kwargs):
        """
        Save model state in ML-framework "native" format, \
            e.g. PyTorch pickled models.

        May save one file or multiple files, depending on the framework.

        Args:
            filepath (string): If framework stores a single file, this should
                               be a single file path.
            Frameworks that store multiple files may need to derive the other
            paths from this path.
            kwargs           : For future-proofing

        Returns:
            None
        """
        return self.runner.save_native(filepath, **kwargs)

    def rebuild_model(self, round_num, input_tensor_dict, validation=False):
        """
        Parse tensor names and update weights of model. Handles the optimizer treatment.

        Returns:
            None
        """
        return self.runner.rebuild_model(
            round_num, input_tensor_dict, validation)

    def set_optimizer_treatment(self, opt_treatment):
        """Change treatment of current instance optimizer."""
        super().set_optimizer_treatment(opt_treatment)
        self.runner.opt_treatment = opt_treatment
