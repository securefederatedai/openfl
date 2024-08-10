# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""TensorFlowTaskRunner module."""

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from openfl.federated.task.runner import TaskRunner
from openfl.utilities import TensorKey
from openfl.utilities.split import split_tensor_dict_for_holdouts


class TensorFlowTaskRunner(TaskRunner):
    """Base class for TensorFlow models in the Federated Learning solution.

    Attributes:
        assign_ops (tf.Operation): TensorFlow operations for assignment.
        placeholders (tf.Tensor): TensorFlow placeholders for tensors.
        tvar_assign_ops (tf.Operation): TensorFlow operations for assignment
            of trainable variables.
        tvar_placeholders (tf.Tensor): TensorFlow placeholders for trainable
            variables.
        input_shape (tuple): Shape of the input features.
        required_tensorkeys_for_function (dict): Required tensorkeys for all
            public functions in TensorFlowTaskRunner.
        sess (tf.Session): TensorFlow session.
        X (tf.Tensor): Input features to the model.
        y (tf.Tensor): Input labels to the model.
        train_step (tf.Operation): Optimizer train step operation.
        loss (tf.Tensor): Model loss function.
        output (tf.Tensor): Model output tensor.
        validation_metric (tf.Tensor): Function used to validate the model
            outputs against labels.
        tvars (list): TensorFlow trainable variables.
        opt_vars (list): Optimizer variables.
        fl_vars (list): Trainable variables and optimizer variables.

    .. note::
        Child classes should have __init__ function signature (self, data,
            kwargs),
        and should overwrite at least the following while defining the model.
    """

    def __init__(self, **kwargs):
        """Initializes the TensorFlowTaskRunner object.

        Args:
            **kwargs: Additional parameters to pass to the function.
        """
        tf.disable_v2_behavior()

        super().__init__(**kwargs)

        self.assign_ops = None
        self.placeholders = None

        self.tvar_assign_ops = None
        self.tvar_placeholders = None

        # construct the shape needed for the input features
        self.input_shape = (None,) + self.data_loader.get_feature_shape()

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
        """Parse tensor names and update weights of model. Handles the
        optimizer treatment.

        Args:
            round_num (int): The round number.
            input_tensor_dict (dict): The input tensor dictionary.
            validation (bool): If True, perform validation. Default is False.

        Returns:
            None
        """
        if self.opt_treatment == "RESET":
            self.reset_opt_vars()
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)
        elif round_num > 0 and self.opt_treatment == "CONTINUE_GLOBAL" and not validation:
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=True)
        else:
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)

    def train_batches(
        self, col_name, round_num, input_tensor_dict, epochs=1, use_tqdm=False, **kwargs
    ):
        """
        Perform the training.

        Is expected to perform draws randomly, without replacement until data
            is exausted. Then data is replaced and shuffled and draws continue.

        Args:
            col_name (str): The column name.
            round_num (int): The round number.
            input_tensor_dict (dict): The input tensor dictionary.
            epochs (int): Number of epochs to train. Default is 1.
            use_tqdm (bool): If True, use tqdm to print a progress bar.
                Default is False.
            **kwargs: Additional parameters to pass to the function.

        Returns:
            float: loss metric.
        """
        batch_size = self.data_loader.batch_size

        if kwargs["batch_size"]:
            batch_size = kwargs["batch_size"]

        # rebuild model with updated weights
        self.rebuild_model(round_num, input_tensor_dict)

        tf.keras.backend.set_learning_phase(True)
        losses = []

        for epoch in range(epochs):
            self.logger.info("Run %s epoch of %s round", epoch, round_num)
            # get iterator for batch draws (shuffling happens here)
            gen = self.data_loader.get_train_loader(batch_size)
            if use_tqdm:
                gen = tqdm.tqdm(gen, desc="training epoch")

            for X, y in gen:
                losses.append(self.train_batch(X, y))

        # Output metric tensors (scalar)
        origin = col_name
        tags = ("trained",)
        output_metric_dict = {
            TensorKey(self.loss_name, origin, round_num, True, ("metric",)): np.array(
                np.mean(losses)
            )
        }

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs
        )

        # Create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags): nparray
            for tensor_name, nparray in global_model_dict.items()
        }
        # Create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags): nparray
            for tensor_name, nparray in local_model_dict.items()
        }
        # The train/validate aggregated function of the next round will
        # look for the updated model parameters.
        # This ensures they will be resolved locally
        next_local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num + 1, False, ("model",)): nparray
            for tensor_name, nparray in local_model_dict.items()
        }

        global_tensor_dict = {
            **output_metric_dict,
            **global_tensorkey_model_dict,
        }
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict,
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
        if self.opt_treatment == "CONTINUE_GLOBAL":
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        return global_tensor_dict, local_tensor_dict

    def train_batch(self, X, y):
        """Train the model on a single batch.

        Args:
            X (tf.Tensor): Input to the model.
            y (tf.Tensor): Ground truth label to the model.

        Returns:
            loss (float): loss metric.
        """
        feed_dict = {self.X: X, self.y: y}

        # run the train step and return the loss
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)

        return loss

    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs):
        """
        Run validation.

        Args:
            col_name (str): The column name.
            round_num (int): The round number.
            input_tensor_dict (dict): The input tensor dictionary.
            use_tqdm (bool): If True, use tqdm to print a progress bar.
                Default is False.
            **kwargs: Additional parameters to pass to the function.

        Returns:
            output_tensor_dict (dict): {<metric>: <value>}.
        """
        batch_size = self.data_loader.batch_size

        if kwargs["batch_size"]:
            batch_size = kwargs["batch_size"]

        self.rebuild_model(round_num, input_tensor_dict, validation=True)

        tf.keras.backend.set_learning_phase(False)

        score = 0

        gen = self.data_loader.get_valid_loader(batch_size)
        if use_tqdm:
            gen = tqdm.tqdm(gen, desc="validating")

        for X, y in gen:
            weight = X.shape[0] / self.data_loader.get_valid_data_size()
            _, s = self.validate_batch(X, y)
            score += s * weight

        origin = col_name
        suffix = "validate"
        if kwargs["apply"] == "local":
            suffix += "_local"
        else:
            suffix += "_agg"
        tags = ("metric", suffix)
        output_tensor_dict = {
            TensorKey(self.validation_metric_name, origin, round_num, True, tags): np.array(score)
        }

        # return empty dict for local metrics
        return output_tensor_dict, {}

    def validate_batch(self, X, y):
        """Validate the model on a single local batch.

        Args:
            X (tf.Tensor): Input to the model.
            y (tf.Tensor): Ground truth label to the model.

        Returns:
            float: loss metric.
        """
        feed_dict = {self.X: X, self.y: y}

        return self.sess.run([self.output, self.validation_metric], feed_dict=feed_dict)

    def get_tensor_dict(self, with_opt_vars=True):
        """Get the dictionary weights.

        Get the weights from the tensor.

        Args:
            with_opt_vars (bool): Specify if we also want to get the variables
                of the optimizer. Default is True.

        Returns:
            dict: The weight dictionary {<tensor_name>: <value>}.
        """
        if with_opt_vars is True:
            variables = self.fl_vars
        else:
            variables = self.tvars

        # FIXME: do this in one call?
        return {var.name: val for var, val in zip(variables, self.sess.run(variables))}

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """Set the tensor dictionary.

        Set the model weights with a tensor dictionary:
        {<tensor_name>: <value>}.

        Args:
            tensor_dict (dict): The model weights dictionary.
            with_opt_vars (bool): Specify if we also want to set the variables
                of the optimizer.

        Returns:
            None
        """
        if with_opt_vars:
            self.assign_ops, self.placeholders = tf_set_tensor_dict(
                tensor_dict,
                self.sess,
                self.fl_vars,
                self.assign_ops,
                self.placeholders,
            )
        else:
            self.tvar_assign_ops, self.tvar_placeholders = tf_set_tensor_dict(
                tensor_dict,
                self.sess,
                self.tvars,
                self.tvar_assign_ops,
                self.tvar_placeholders,
            )

    def reset_opt_vars(self):
        """Reinitialize the optimizer variables.

        Returns:
            None
        """
        for v in self.opt_vars:
            v.initializer.run(session=self.sess)

    def initialize_globals(self):
        """Initialize Global Variables.

        Initialize all global variables

        Returns:
            None
        """
        self.sess.run(tf.global_variables_initializer())

    def _get_weights_names(self, with_opt_vars=True):
        """Get the weights.

        Args:
            with_opt_vars (bool): Specify if we also want to get the variables
                of the optimizer. Default is True.

        Returns:
            list: The weight names list.
        """
        if with_opt_vars is True:
            variables = self.fl_vars
        else:
            variables = self.tvars

        return [var.name for var in variables]

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """Get the required tensors for specified function that could be called
        as part of a task.

        By default, this is just all of the layers and optimizer of the model.

        Args:
            func_name (str): The function name.
            **kwargs: Additional parameters to pass to the function.

        Returns:
            required_tensorkeys_for_function (list): List of required
                TensorKey. [TensorKey].
        """
        if func_name == "validate":
            local_model = "apply=" + str(kwargs["apply"])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Set the required tensors for all publicly accessible methods that
        could be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function

        Args:
            with_opt_vars (bool): Specify if we also want to set the variables
                of the optimizer. Default is False.

        Returns:
            None
        """
        # TODO there should be a way to programmatically iterate through
        #  all of the methods in the class and declare the tensors.
        # For now this is done manually

        output_model_dict = self.get_tensor_dict(with_opt_vars=with_opt_vars)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs
        )
        if not with_opt_vars:
            global_model_dict_val = global_model_dict
            local_model_dict_val = local_model_dict
        else:
            output_model_dict = self.get_tensor_dict(with_opt_vars=False)
            global_model_dict_val, local_model_dict_val = split_tensor_dict_for_holdouts(
                self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs
            )

        self.required_tensorkeys_for_function["train_batches"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
            for tensor_name in global_model_dict
        ]
        self.required_tensorkeys_for_function["train_batches"] += [
            TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
            for tensor_name in local_model_dict
        ]

        # Validation may be performed on local or aggregated (global)
        # model, so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function["validate"] = {}
        # TODO This is not stateless. The optimizer will not be
        self.required_tensorkeys_for_function["validate"]["apply=local"] = [
            TensorKey(tensor_name, "LOCAL", 0, False, ("trained",))
            for tensor_name in {**global_model_dict_val, **local_model_dict_val}
        ]
        self.required_tensorkeys_for_function["validate"]["apply=global"] = [
            TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
            for tensor_name in global_model_dict_val
        ]
        self.required_tensorkeys_for_function["validate"]["apply=global"] += [
            TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
            for tensor_name in local_model_dict_val
        ]


# FIXME: what's a nicer construct than this? ugly interface. Perhaps we
#  get an object with an assumed interface that lets is set/get these?
# Note that this will return the assign_ops and placeholder nodes it uses
# if called with None, it will create them.
# to avoid inflating the graph, caller should keep these and pass them back
# What if we want to set a different group of vars in the middle?
# It is good if it is the subset of the original variables.
def tf_set_tensor_dict(tensor_dict, session, variables, assign_ops=None, placeholders=None):
    """Tensorflow set tensor dictionary.

    Args:
        tensor_dict (dict): Dictionary of tensors.
        session (tf.Session): TensorFlow session.
        variables (list): List of TensorFlow variables.
        assign_ops (tf.Operation, optional): TensorFlow operations for
            assignment. Default is None.
        placeholders (tf.Tensor, optional): TensorFlow placeholders for
            tensors. Default is None.

    Returns:
        assign_ops (tf.Operation): TensorFlow operations for assignment.
        placeholders (tf.Tensor): TensorFlow placeholders for tensors.
    """
    if placeholders is None:
        placeholders = {v.name: tf.placeholder(v.dtype, shape=v.shape) for v in variables}
    if assign_ops is None:
        assign_ops = {v.name: tf.assign(v, placeholders[v.name]) for v in variables}

    for k, v in tensor_dict.items():
        session.run(assign_ops[k], feed_dict={placeholders[k]: v})

    return assign_ops, placeholders
