# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TensorFlowTaskRunner module."""
import numpy as np
import tensorflow as tf

from openfl.utilities import change_tags, Metric, TensorKey
from openfl.utilities.split import split_tensor_dict_for_holdouts
from .runner import TaskRunner


class TensorFlowTaskRunner(TaskRunner):
    """The base model for Keras models in the federation."""

    def __init__(self, **kwargs):
        """
        Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        self.model = tf.keras.Model()

        self.model_tensor_names = []

        # this is a map of all of the required tensors for each of the public
        # functions in KerasTaskRunner
        self.required_tensorkeys_for_function = {}
        tf.keras.backend.clear_session()

    def rebuild_model(self, round_num, input_tensor_dict, validation=False):
        """
        Parse tensor names and update weights of model. Handles the optimizer treatment.

        Returns
        -------
        None
        """

        if self.opt_treatment == 'RESET':
            self.reset_opt_vars()
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)
        elif (round_num > 0 and self.opt_treatment == 'CONTINUE_GLOBAL'
              and not validation):
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=True)
        else:
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)

    def train_task(self, col_name, round_num, input_tensor_dict,
                   metrics, epochs=1, batch_size=1, **kwargs):
        """
        Perform the training.

        Is expected to perform draws randomly, without replacement until data is exausted.
        Then data is replaced and shuffled and draws continue.

        Returns
        -------
        dict
            'TensorKey: nparray'
        """
        if metrics is None:
            raise KeyError('metrics must be defined')

        # rebuild model with updated weights
        self.rebuild_model(round_num, input_tensor_dict)
        for epoch in range(epochs):
            self.logger.info(f'Run {epoch} epoch of {round_num} round')
            results = self.train_(self.data_loader.get_train_loader(batch_size),
                                           metrics=metrics,
                                           **kwargs)

        # output metric tensors (scalar)
        origin = col_name
        tags = ('trained',)
        output_metric_dict = {
            TensorKey(
                metric_name, origin, round_num, True, ('metric',)
            ): metric_value
            for (metric_name, metric_value) in results
        }

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )

        # create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags):
                nparray for tensor_name, nparray in global_model_dict.items()
        }
        # create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags):
                nparray for tensor_name, nparray in local_model_dict.items()
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
        # roles between rounds.
        # for example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator because
        # these are only created after training occurs. A work around could
        # involve doing a single epoch of training on random data to get the
        # optimizer names, and then throwing away the model.
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        return global_tensor_dict, local_tensor_dict

    def train_(self, batch_generator, metrics: list = None, **kwargs):
        """Train single epoch.

        Override this function for custom training.

        Args:
            batch_generator: Generator of training batches.
                Each batch is a tuple of N train images and N train labels
                where N is the batch size of the DataLoader of the current TaskRunner instance.

            epochs: Number of epochs to train.
            metrics: Names of metrics to save.
        """
        if metrics is None:
            metrics = []
        # TODO Currently assuming that all metrics are defined at
        #  initialization (build_model).
        #  If metrics are added (i.e. not a subset of what was originally
        #  defined) then the model must be recompiled.
        model_metrics_names = self.model.metrics_names

        # TODO if there are new metrics in the flplan that were not included
        #  in the originally
        #  compiled model, that behavior is not currently handled.
        for param in metrics:
            if param not in model_metrics_names:
                raise ValueError(
                    f'KerasTaskRunner does not support specifying new metrics. '
                    f'Param_metrics = {metrics}, model_metrics_names = {model_metrics_names}'
                )

        history = self.model.fit(batch_generator,
                                 verbose=1,
                                 **kwargs)
        results = []
        for metric in metrics:
            value = np.mean([history.history[metric]])
            results.append(Metric(name=metric, value=np.array(value)))
        return results

    def validate_task(self, col_name, round_num, input_tensor_dict, **kwargs):
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
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size = 1

        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        param_metrics = kwargs['metrics']

        vals = self.model.evaluate(
            self.data_loader.get_valid_loader(batch_size),
            verbose=1
        )
        model_metrics_names = self.model.metrics_names
        if type(vals) is not list:
            vals = [vals]
        ret_dict = dict(zip(model_metrics_names, vals))

        # TODO if there are new metrics in the flplan that were not included in
        #  the originally compiled model, that behavior is not currently
        #  handled.
        for param in param_metrics:
            if param not in model_metrics_names:
                raise ValueError(
                    f'KerasTaskRunner does not support specifying new metrics. '
                    f'Param_metrics = {param_metrics}, model_metrics_names = {model_metrics_names}'
                )

        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric',)
        tags = change_tags(tags, add_field=suffix)
        output_tensor_dict = {
            TensorKey(metric, origin, round_num, True, tags):
                np.array(ret_dict[metric])
            for metric in param_metrics}

        return output_tensor_dict, {}

    def save_native(self, filepath):
        """Save model."""
        self.model.save(filepath)

    def load_native(self, filepath):
        """Load model."""
        self.model = tf.keras.models.load_model(filepath)

    @staticmethod
    def _get_weights_names(obj, with_opt_vars):
        """
        Get the list of weight names.

        Parameters
        ----------
        obj : Model or Optimizer
            The target object that we want to get the weights.

        with_opt_vars (bool): Specify if we want to get optimizer weights

        Returns
        -------
        dict
            The weight name list
        """
        if with_opt_vars:
            # When acquiring optimizer weights, check optimizer version.
            # Current optimizer does not use 'weights' attributes
            if 'legacy' in obj.__class__.__module__:
                weight_names = [weight.name for weight in obj.weights]
            else:
                weight_names = [weight.name for weight in obj.variables]
                
        weight_names = [weight.name for weight in obj.weights]
        return weight_names

    @staticmethod
    def _get_weights_dict(obj, suffix='', with_opt_vars=False):
        """
        Get the dictionary of weights.

        Parameters
        ----------
        obj : Model or Optimizer
            The target object that we want to get the weights.

        with_opt_vars (bool): Specify if we want to get optimizer weights

        Returns
        -------
        dict
            The weight dictionary.
        """

        weights_dict = {}
        if with_opt_vars:
            # When acquiring optimizer weights, check optimizer version.
            # Current optimizer does not use 'weights' or '.get_weights()' attributes
            if 'legacy' in obj.__class__.__module__:
                weight_names = [weight.name for weight in obj.weights]
                weight_values = obj.get_weights()
            else:
                weight_names = [weight.name for weight in obj.variables]
                weight_values = [weight.numpy() for weight in obj.variables]
        else:
            weight_names = [weight.name for weight in obj.weights]
            weight_values = obj.get_weights()


        for name, value in zip(weight_names, weight_values):
            weights_dict[name + suffix] = value
        return weights_dict

    @staticmethod
    def _set_weights_dict(obj, weights_dict, with_opt_vars=False):
        """Set the object weights with a dictionary.

        The obj can be a model or an optimizer.

        Args:
            obj (Model or Optimizer): The target object that we want to set
            the weights.
            weights_dict (dict): The weight dictionary.
            with_opt_vars (bool): Specify if we want to set optimizer weights
        Returns:
            None
        """
                
        if with_opt_vars:
            # When acquiring optimizer weights, check optimizer version.
            # Current optimizer does not use 'weights' attributes
            if 'legacy' in obj.__class__.__module__:
                weight_names = [weight.name for weight in obj.weights]
            else:
                weight_names = [weight.name for weight in obj.variables]
        else:
            weight_names = [weight.name for weight in obj.weights]

        weight_values = [weights_dict[name] for name in weight_names]

        obj.set_weights(weight_values)

    def get_tensor_dict(self, with_opt_vars, suffix=''):
        """
        Get the model weights as a tensor dictionary.

        Parameters
        ----------
        with_opt_vars : bool
            If we should include the optimizer's status.
        suffix : string
            Universally

        Returns:
            dict: The tensor dictionary.
        """
        
        model_weights = self._get_weights_dict(self.model, suffix)

        if with_opt_vars:
            
            opt_weights = self._get_weights_dict(self.model.optimizer, suffix, with_opt_vars)

            model_weights.update(opt_weights)
            
            if len(opt_weights) == 0:
                self.logger.debug(
                    "WARNING: We didn't find variables for the optimizer.")
        return model_weights

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """
        Set the model weights with a tensor dictionary.

        Args:
            tensor_dict: the tensor dictionary
            with_opt_vars (bool): True = include the optimizer's status.
        """
        if with_opt_vars is False:
            # It is possible to pass in opt variables from the input tensor dict
            # This will make sure that the correct layers are updated
            model_weight_names = [weight.name for weight in self.model.weights]
            model_weights_dict = {
                name: tensor_dict[name] for name in model_weight_names
            }
            self._set_weights_dict(self.model, model_weights_dict)
        else:
            model_weight_names = [
                weight.name for weight in self.model.weights
            ]
            model_weights_dict = {
                name: tensor_dict[name] for name in model_weight_names
            }
            if 'legacy' in self.model.optimizer.__class__.__module__:
                opt_weight_names = [
                    weight.name for weight in self.model.optimizer.weights
                    ]
            else:
                opt_weight_names = [
                    weight.name for weight in self.model.optimizer.variables
                    ]

            opt_weights_dict = {
                name: tensor_dict[name] for name in opt_weight_names
            }
            self._set_weights_dict(self.model, model_weights_dict)
            self._set_weights_dict(self.model.optimizer, opt_weights_dict, with_opt_vars)

    def reset_opt_vars(self):
        """
        Reset optimizer variables.

        Resets the optimizer variables

        """
        for var in self.model.optimizer.variables():
            var.assign(tf.zeros_like(var))
        self.logger.debug('Optimizer variables reset')

    def set_required_tensorkeys_for_function(self, func_name,
                                             tensor_key, **kwargs):
        """
        Set the required tensors for specified function that could be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.
         Custom tensors should be added to this function

        Parameters
        ----------
        func_name: string
        tensor_key: TensorKey (namedtuple)
        **kwargs: Any function arguments {}

        Returns
        -------
        None
        """
        # TODO there should be a way to programmatically iterate through all
        #  of the methods in the class and declare the tensors.
        # For now this is done manually

        if func_name == 'validate_task':
            # Should produce 'apply=global' or 'apply=local'
            local_model = 'apply' + kwargs['apply']
            self.required_tensorkeys_for_function[func_name][
                local_model].append(tensor_key)
        else:
            self.required_tensorkeys_for_function[func_name].append(tensor_key)

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """
        Get the required tensors for specified function that could be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.

        Parameters
        ----------
        None

        Returns
        -------
        List
            [TensorKey]
        """
        if func_name == 'validate_task':
            local_model = 'apply=' + str(kwargs['apply'])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]

    def update_tensorkeys_for_functions(self):
        """
        Update the required tensors for all publicly accessible methods \
            that could be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # TODO complete this function. It is only needed for opt_treatment,
        #  and making the model stateless

        # Minimal required tensors for train function
        model_layer_names = self._get_weights_names(self.model)
        opt_names = self._get_weights_names(self.model.optimizer)
        tensor_names = model_layer_names + opt_names
        self.logger.debug(f'Updating model tensor names: {tensor_names}')
        self.required_tensorkeys_for_function['train_task'] = [
            TensorKey(tensor_name, 'GLOBAL', 0, ('model',))
            for tensor_name in tensor_names
        ]

        # Validation may be performed on local or aggregated (global) model,
        # so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function['validate_task'] = {}
        self.required_tensorkeys_for_function['validate_task']['local_model=True'] = [
            TensorKey(tensor_name, 'LOCAL', 0, ('trained',))
            for tensor_name in tensor_names
        ]
        self.required_tensorkeys_for_function['validate_task']['local_model=False'] = [
            TensorKey(tensor_name, 'GLOBAL', 0, ('model',))
            for tensor_name in tensor_names
        ]

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """
        Set the required tensors for all publicly accessible methods \
            that could be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # TODO there should be a way to programmatically iterate through all
        #  of the methods in the class and declare the tensors.
        # For now this is done manually

        output_model_dict = self.get_tensor_dict(with_opt_vars=with_opt_vars)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )
        if not with_opt_vars:
            global_model_dict_val = global_model_dict
            local_model_dict_val = local_model_dict
        else:
            output_model_dict = self.get_tensor_dict(with_opt_vars=False)
            global_model_dict_val, local_model_dict_val = split_tensor_dict_for_holdouts(
                self.logger,
                output_model_dict,
                **self.tensor_dict_split_fn_kwargs
            )

        self.required_tensorkeys_for_function['train_task'] = [
            TensorKey(tensor_name, 'GLOBAL', 0, False, ('model',))
            for tensor_name in global_model_dict
        ]
        self.required_tensorkeys_for_function['train_task'] += [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('model',))
            for tensor_name in local_model_dict
        ]

        # Validation may be performed on local or aggregated (global) model,
        # so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function['validate_task'] = {}
        # TODO This is not stateless. The optimizer will not be
        self.required_tensorkeys_for_function['validate_task']['apply=local'] = [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('trained',))
            for tensor_name in {
                **global_model_dict_val,
                **local_model_dict_val
            }
        ]
        self.required_tensorkeys_for_function['validate_task']['apply=global'] = [
            TensorKey(tensor_name, 'GLOBAL', 0, False, ('model',))
            for tensor_name in global_model_dict_val
        ]
        self.required_tensorkeys_for_function['validate_task']['apply=global'] += [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('model',))
            for tensor_name in local_model_dict_val
        ]


import tensorflow.compat.v1
from tqdm import tqdm


class TensorFlowTaskRunner_v1(TaskRunner):
    """
    Base class for TensorFlow models in the Federated Learning solution.

        child classes should have __init__ function signature (self, data, kwargs),
        and should overwrite at least the following while defining the model
    """

    def __init__(self, **kwargs):
        """
        Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        tensorflow.compat.v1.disable_v2_behavior()

        super().__init__(**kwargs)

        self.assign_ops = None
        self.placeholders = None

        self.tvar_assign_ops = None
        self.tvar_placeholders = None

        # construct the shape needed for the input features
        self.input_shape = (None,) + self.data_loader.get_feature_shape()

        # Required tensorkeys for all public functions in TensorFlowTaskRunner
        self.required_tensorkeys_for_function = {}

        # Required tensorkeys for all public functions in TensorFlowTaskRunner
        self.required_tensorkeys_for_function = {}

        # tensorflow session
        self.sess = None
        # input featrures to the model
        self.X = None
        # input labels to the model
        self.y = None
        # optimizer train step operation
        self.train_step = None
        # model loss function
        self.loss = None
        # model output tensor
        self.output = None
        # function used to validate the model outputs against labels
        self.validation_metric = None
        # tensorflow trainable variables
        self.tvars = None
        # self.optimizer.variables() once self.optimizer is defined
        self.opt_vars = None
        # self.tvars + self.opt_vars
        self.fl_vars = None

    def rebuild_model(self, round_num, input_tensor_dict, validation=False):
        """
        Parse tensor names and update weights of model. Handles the optimizer treatment.

        Returns:
            None
        """
        if self.opt_treatment == 'RESET':
            self.reset_opt_vars()
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)
        elif (round_num > 0 and self.opt_treatment == 'CONTINUE_GLOBAL'
              and not validation):
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=True)
        else:
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)

    def train_task(self, col_name, round_num, input_tensor_dict,
                   epochs=1, use_tqdm=False, **kwargs):
        """
        Perform the training.

        Is expected to perform draws randomly, without replacement until data is exausted. Then
        data is replaced and shuffled and draws continue.

        Args:
            use_tqdm (bool): True = use tqdm to print a progress
             bar (Default=False)
            epochs (int): Number of epochs to train
        Returns:
            float: loss metric
        """
        batch_size = self.data_loader.batch_size

        if kwargs['batch_size']:
            batch_size = kwargs['batch_size']

        # rebuild model with updated weights
        self.rebuild_model(round_num, input_tensor_dict)

        tensorflow.compat.v1.keras.backend.set_learning_phase(True)
        losses = []

        for epoch in range(epochs):
            self.logger.info(f'Run {epoch} epoch of {round_num} round')
            # get iterator for batch draws (shuffling happens here)
            gen = self.data_loader.get_train_loader(batch_size)
            if use_tqdm:
                gen = tqdm.tqdm(gen, desc='training epoch')

            for (X, y) in gen:
                losses.append(self.train_batch(X, y))

        # Output metric tensors (scalar)
        origin = col_name
        tags = ('trained',)
        output_metric_dict = {
            TensorKey(
                self.loss_name, origin, round_num, True, ('metric',)
            ): np.array(np.mean(losses))
        }

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )

        # Create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags):
                nparray for tensor_name, nparray in global_model_dict.items()
        }
        # Create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags):
                nparray for tensor_name, nparray in local_model_dict.items()
        }
        # The train/validate aggregated function of the next round will
        # look for the updated model parameters.
        # This ensures they will be resolved locally
        next_local_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num + 1, False, ('model',)
            ): nparray for tensor_name, nparray in local_model_dict.items()}

        global_tensor_dict = {
            **output_metric_dict,
            **global_tensorkey_model_dict
        }
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict
        }

        # Update the required tensors if they need to be pulled from
        # the aggregator
        # TODO this logic can break if different collaborators have different
        #  roles between rounds.
        # For example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator because
        # these are only created after training occurs. A work around could
        # involve doing a single epoch of training on random data to get the
        # optimizer names, and then throwing away the model.
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        return global_tensor_dict, local_tensor_dict

    def train_batch(self, X, y):
        """
        Train the model on a single batch.

        Args:
            X: Input to the model
            y: Ground truth label to the model

        Returns:
            float: loss metric
        """
        feed_dict = {self.X: X, self.y: y}

        # run the train step and return the loss
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)

        return loss

    def validate_task(self, col_name, round_num,
                      input_tensor_dict, use_tqdm=False, **kwargs):
        """
        Run validation.

        Returns:
            dict: {<metric>: <value>}
        """
        batch_size = self.data_loader.batch_size

        if kwargs['batch_size']:
            batch_size = kwargs['batch_size']

        self.rebuild_model(round_num, input_tensor_dict, validation=True)

        tensorflow.compat.v1.keras.backend.set_learning_phase(False)

        score = 0

        gen = self.data_loader.get_valid_loader(batch_size)
        if use_tqdm:
            gen = tqdm.tqdm(gen, desc='validating')

        for X, y in gen:
            weight = X.shape[0] / self.data_loader.get_valid_data_size()
            _, s = self.validate_(X, y)
            score += s * weight

        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)
        output_tensor_dict = {
            TensorKey(
                self.validation_metric_name, origin, round_num, True, tags
            ): np.array(score)}

        # return empty dict for local metrics
        return output_tensor_dict, {}

    def validate_(self, X, y):
        """Validate the model on a single local batch.

        Args:
            X: Input to the model
            y: Ground truth label to the model

        Returns:
            float: loss metric

        """
        feed_dict = {self.X: X, self.y: y}

        return self.sess.run(
            [self.output, self.validation_metric], feed_dict=feed_dict)

    def get_tensor_dict(self, with_opt_vars=True):
        """Get the dictionary weights.

        Get the weights from the tensor

        Args:
            with_opt_vars (bool): Specify if we also want to get the variables
             of the optimizer

        Returns:
            dict: The weight dictionary {<tensor_name>: <value>}

        """
        if with_opt_vars is True:
            variables = self.fl_vars
        else:
            variables = self.tvars

        # FIXME: do this in one call?
        return {var.name: val for var, val in zip(
            variables, self.sess.run(variables))}

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """Set the tensor dictionary.

        Set the model weights with a tensor
         dictionary: {<tensor_name>: <value>}.

        Args:
            tensor_dict (dict): The model weights dictionary
            with_opt_vars (bool): Specify if we also want to set the variables
             of the optimizer

        Returns:
            None
        """
        if with_opt_vars:
            self.assign_ops, self.placeholders = tf_set_tensor_dict(
                tensor_dict, self.sess, self.fl_vars,
                self.assign_ops, self.placeholders
            )
        else:
            self.tvar_assign_ops, self.tvar_placeholders = tf_set_tensor_dict(
                tensor_dict,
                self.sess,
                self.tvars,
                self.tvar_assign_ops,
                self.tvar_placeholders
            )

    def reset_opt_vars(self):
        """Reinitialize the optimizer variables."""
        for v in self.opt_vars:
            v.initializer.run(session=self.sess)

    def initialize_globals(self):
        """Initialize Global Variables.

        Initialize all global variables

        Returns:
            None
        """
        self.sess.run(tensorflow.compat.v1.global_variables_initializer())

    def _get_weights_names(self, with_opt_vars=True):
        """Get the weights.

        Args:
            with_opt_vars (bool): Specify if we also want to get the variables
             of the optimizer.

        Returns:
            list : The weight names list
        """
        if with_opt_vars is True:
            variables = self.fl_vars
        else:
            variables = self.tvars

        return [var.name for var in variables]

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """
        Get the required tensors for specified function that could be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.

        Returns:
            list : [TensorKey]
        """
        if func_name == 'validate':
            local_model = 'apply=' + str(kwargs['apply'])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """
        Set the required tensors for all publicly accessible methods \
            that could be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function

        """
        # TODO there should be a way to programmatically iterate through
        #  all of the methods in the class and declare the tensors.
        # For now this is done manually

        output_model_dict = self.get_tensor_dict(with_opt_vars=with_opt_vars)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )
        if not with_opt_vars:
            global_model_dict_val = global_model_dict
            local_model_dict_val = local_model_dict
        else:
            output_model_dict = self.get_tensor_dict(with_opt_vars=False)
            global_model_dict_val, local_model_dict_val = split_tensor_dict_for_holdouts(
                self.logger,
                output_model_dict,
                **self.tensor_dict_split_fn_kwargs
            )

        self.required_tensorkeys_for_function['train_task'] = [
            TensorKey(tensor_name, 'GLOBAL', 0, False, ('model',))
            for tensor_name in global_model_dict]
        self.required_tensorkeys_for_function['train_task'] += [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('model',))
            for tensor_name in local_model_dict]

        # Validation may be performed on local or aggregated (global)
        # model, so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function['validate_task'] = {}
        # TODO This is not stateless. The optimizer will not be
        self.required_tensorkeys_for_function['validate_task']['apply=local'] = [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('trained',))
            for tensor_name in {
                **global_model_dict_val,
                **local_model_dict_val
            }
        ]
        self.required_tensorkeys_for_function['validate']['apply=global'] = [
            TensorKey(tensor_name, 'GLOBAL', 0, False, ('model',))
            for tensor_name in global_model_dict_val
        ]
        self.required_tensorkeys_for_function['validate']['apply=global'] += [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('model',))
            for tensor_name in local_model_dict_val
        ]


# FIXME: what's a nicer construct than this? ugly interface. Perhaps we
#  get an object with an assumed interface that lets is set/get these?
# Note that this will return the assign_ops and placeholder nodes it uses
# if called with None, it will create them.
# to avoid inflating the graph, caller should keep these and pass them back
# What if we want to set a different group of vars in the middle?
# It is good if it is the subset of the original variables.
def tf_set_tensor_dict(tensor_dict, session, variables,
                       assign_ops=None, placeholders=None):
    """Tensorflow set tensor dictionary.

    Args:
        tensor_dict: Dictionary of tensors
        session: TensorFlow session
        variables: TensorFlow variables
        assign_ops: TensorFlow operations (Default=None)
        placeholders: TensorFlow placeholders (Default=None)

    Returns:
        assign_ops, placeholders

    """
    if placeholders is None:
        placeholders = {
            v.name: tensorflow.compat.v1.placeholder(v.dtype, shape=v.shape) for v in variables
        }
    if assign_ops is None:
        assign_ops = {
            v.name: tensorflow.compat.v1.assign(v, placeholders[v.name]) for v in variables
        }

    for k, v in tensor_dict.items():
        session.run(assign_ops[k], feed_dict={placeholders[k]: v})

    return assign_ops, placeholders