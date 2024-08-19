# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Interactive API package."""

from logging import getLogger

import numpy as np

from openfl.utilities import TensorKey, change_tags
from openfl.utilities.split import split_tensor_dict_for_holdouts


class CoreTaskRunner:
    """Federated Learning Task Runner Class.

    Attributes:
        kwargs (dict): Additional parameters passed to the function.
        TASK_REGISTRY (dict): Registry of tasks.
        training_round_completed (bool): Flag indicating if a training round
            has been completed.
        tensor_dict_split_fn_kwargs (dict): Key word arguments for determining
            which parameters to hold out from aggregation.
        required_tensorkeys_for_function (dict): Required tensorkeys for all
            public functions in CoreTaskRunner.
        logger (logging.Logger): Logger object for logging events.
        opt_treatment (str): Treatment of current instance optimizer.
    """

    def _prepare_tensorkeys_for_agggregation(
        self, metric_dict, validation_flag, col_name, round_num
    ):
        """Prepare tensorkeys for aggregation.
        Args:
            metric_dict (dict): Dictionary of metrics.
            validation_flag (bool): Flag indicating if validation is to be
                performed.
            col_name (str): The column name.
            round_num (int): The round number.

        Returns:
            tuple: Tuple containing global_tensor_dict and local_tensor_dict.
        """
        global_tensor_dict, local_tensor_dict = {}, {}
        origin = col_name
        if not validation_flag:
            # Output metric tensors (scalar)
            tags = ("trained",)

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
            # The train/validate aggregated function of the next
            # round will look for the updated model parameters.
            # This ensures they will be resolved locally
            next_local_tensorkey_model_dict = {
                TensorKey(tensor_name, origin, round_num + 1, False, ("model",)): nparray
                for tensor_name, nparray in local_model_dict.items()
            }

            global_tensor_dict = global_tensorkey_model_dict
            local_tensor_dict = {
                **local_tensorkey_model_dict,
                **next_local_tensorkey_model_dict,
            }

            # Update the required tensors if they need to be
            # pulled from the aggregator
            # TODO this logic can break if different collaborators
            #  have different roles between rounds.
            # For example, if a collaborator only performs validation
            # in the first round but training
            # in the second, it has no way of knowing the optimizer
            # state tensor names to request from the aggregator
            # because these are only created after training occurs.
            # A work around could involve doing a single epoch of training
            # on random data to get the optimizer names,
            # and then throwing away the model.
            if self.opt_treatment == "CONTINUE_GLOBAL":
                self.initialize_tensorkeys_for_functions(with_opt_vars=True)

            # This will signal that the optimizer values are now present,
            # and can be loaded when the model is rebuilt
            self.training_round_completed = True

        else:
            suffix = "validate" + validation_flag
            tags = (suffix,)
        tags = change_tags(tags, add_field="metric")
        metric_dict = {
            TensorKey(metric, origin, round_num, True, tags): np.array(value)
            for metric, value in metric_dict.items()
        }
        global_tensor_dict = {**global_tensor_dict, **metric_dict}

        return global_tensor_dict, local_tensor_dict

    def adapt_tasks(self):
        """Prepare tasks for the collaborator.

        Using functions from a task provider (deserialized interface object)
        and registered task contracts prepares callable tasks to be invoked by
        the collaborator.

        Preparing includes conditional model rebuilding and filling output
        dicts with tensors for aggregation and storing in local DB.

        There is an assumption that any training task accepts optimizer as one
        of the arguments, thus the model should be aggregated after such tasks.

        Returns:
            None
        """

        def task_binder(task_name, callable_task):

            def collaborator_adapted_task(col_name, round_num, input_tensor_dict, **kwargs):
                task_contract = self.task_provider.task_contract[task_name]
                # Validation flag can be [False, '_local', '_agg']
                validation_flag = True if task_contract["optimizer"] is None else False
                task_settings = self.task_provider.task_settings[task_name]

                device = kwargs.get("device", "cpu")

                self.rebuild_model(input_tensor_dict, validation=validation_flag, device=device)
                task_kwargs = {}
                if validation_flag:
                    loader = self.data_loader.get_valid_loader()
                    if kwargs["apply"] == "local":
                        validation_flag = "_local"
                    else:
                        validation_flag = "_agg"
                else:
                    loader = self.data_loader.get_train_loader()
                    # If train task we also pass optimizer
                    task_kwargs[task_contract["optimizer"]] = self.optimizer

                if task_contract["round_num"] is not None:
                    task_kwargs[task_contract["round_num"]] = round_num

                for en_name, entity in zip(
                    ["model", "data_loader", "device"],
                    [self.model, loader, device],
                ):
                    task_kwargs[task_contract[en_name]] = entity

                # Add task settings to the keyword arguments
                task_kwargs.update(task_settings)

                # Here is the training metod call
                metric_dict = callable_task(**task_kwargs)

                return self._prepare_tensorkeys_for_agggregation(
                    metric_dict, validation_flag, col_name, round_num
                )

            return collaborator_adapted_task

        for (
            task_name,
            callable_task,
        ) in self.task_provider.task_registry.items():
            self.TASK_REGISTRY[task_name] = task_binder(task_name, callable_task)

    def __init__(self, **kwargs):
        """Initializes the Task Runner object.

        This class is a part of the Interactive python API release.
        It is no longer a user interface entity that should be subclassed
        but a part of OpenFL kernel.

        Args:
            **kwargs: Additional parameters to pass to the function.
        """
        self.set_logger()

        self.kwargs = kwargs

        self.TASK_REGISTRY = {}

        # Why is it here
        self.opt_treatment = "RESET"
        self.tensor_dict_split_fn_kwargs = {}
        self.required_tensorkeys_for_function = {}

        # Complete hell below
        self.training_round_completed = False
        # overwrite attribute to account for one optimizer param (in every
        # child model that does not overwrite get and set tensordict) that is
        # not a numpy array
        self.tensor_dict_split_fn_kwargs.update({"holdout_tensor_names": ["__opt_state_needed"]})

    def set_task_provider(self, task_provider):
        """Set task registry.

        This method recieves Task Interface object as an argument
        and uses provided callables and information to prepare
        tasks that may be called by the collaborator component.

        Args:
            task_provider: Task provider object.

        Returns:
            None
        """
        if task_provider is None:
            return
        self.task_provider = task_provider
        self.adapt_tasks()

    def set_data_loader(self, data_loader):
        """Register a data loader initialized with local data path.

        Args:
            data_loader: Data loader object.

        Returns:
            None
        """
        self.data_loader = data_loader

    def set_model_provider(self, model_provider):
        """Retrieve a model and an optimizer from the interface object.

        Args:
            model_provider: Model provider object.

        Returns:
            None
        """
        self.model_provider = model_provider
        self.model = self.model_provider.provide_model()
        self.optimizer = self.model_provider.provide_optimizer()

    def set_framework_adapter(self, framework_adapter):
        """Set framework adapter.

        Setting a framework adapter allows first extraction of the weigths
        of the model with the purpose to make a list of parameters to be
        aggregated.

        Args:
            framework_adapter: Framework adapter object.

        Returns:
            None
        """
        self.framework_adapter = framework_adapter
        if self.opt_treatment == "CONTINUE_GLOBAL":
            aggregate_optimizer_parameters = True
        else:
            aggregate_optimizer_parameters = False
        self.initialize_tensorkeys_for_functions(with_opt_vars=aggregate_optimizer_parameters)

    def set_logger(self):
        """Set up the log object.

        Returns:
            None
        """
        self.logger = getLogger(__name__)

    def set_optimizer_treatment(self, opt_treatment):
        # SHould be removed! We have this info at the initialization time
        # and do not change this one during training.
        """Change the treatment of current instance optimizer.

        Args:
            opt_treatment (str): The optimizer treatment.

        Returns:
            None
        """
        self.opt_treatment = opt_treatment

    def rebuild_model(self, input_tensor_dict, validation=False, device="cpu"):
        """
        Parse tensor names and update weights of model. Handles the
        optimizer treatment.

        Args:
            input_tensor_dict (dict): The input tensor dictionary.
            validation (bool): If True, perform validation. Default is False.
            device (str): The device to use. Default is 'cpu'.

        Returns:
            None
        """
        if self.opt_treatment == "RESET":
            self.reset_opt_vars()
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False, device=device)
        elif (
            self.training_round_completed
            and self.opt_treatment == "CONTINUE_GLOBAL"
            and not validation
        ):
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=True, device=device)
        else:
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False, device=device)

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """Get the required tensors for specified function that could be called
        as part of a task.

        By default, this is just all of the layers and optimizer of the model.

        Args:
            func_name (str): The function name.
            **kwargs: Additional parameters to pass to the function.

        Returns:
            list: List of required TensorKey.
        """
        # We rely on validation type tasks parameter `apply`
        # In the interface layer we add those parameters automatically
        if "apply" not in kwargs:
            return [
                TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
                for tensor_name in self.required_tensorkeys_for_function["global_model_dict"]
            ] + [
                TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
                for tensor_name in self.required_tensorkeys_for_function["local_model_dict"]
            ]

        if kwargs["apply"] == "local":
            return [
                TensorKey(tensor_name, "LOCAL", 0, False, ("trained",))
                for tensor_name in {
                    **self.required_tensorkeys_for_function["local_model_dict_val"],
                    **self.required_tensorkeys_for_function["global_model_dict_val"],
                }
            ]

        elif kwargs["apply"] == "global":
            return [
                TensorKey(tensor_name, "GLOBAL", 0, False, ("model",))
                for tensor_name in self.required_tensorkeys_for_function["global_model_dict_val"]
            ] + [
                TensorKey(tensor_name, "LOCAL", 0, False, ("model",))
                for tensor_name in self.required_tensorkeys_for_function["local_model_dict_val"]
            ]

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Set the required tensors for all publicly accessible task methods.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function.

        Args:
            with_opt_vars (bool): Specify if we also want to set the variables
                of the optimizer. Default is False.

        Returns:
            None
        """
        # TODO: Framework adapters should have separate methods for dealing
        # with optimizer. Set model dict for validation tasks
        output_model_dict = self.get_tensor_dict(with_opt_vars=False)
        global_model_dict_val, local_model_dict_val = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs
        )
        # Now set model dict for training tasks
        if with_opt_vars:
            output_model_dict = self.get_tensor_dict(with_opt_vars=True)
            global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
                self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs
            )
        else:
            global_model_dict = global_model_dict_val
            local_model_dict = local_model_dict_val

        self.required_tensorkeys_for_function["global_model_dict"] = global_model_dict
        self.required_tensorkeys_for_function["local_model_dict"] = local_model_dict
        self.required_tensorkeys_for_function["global_model_dict_val"] = global_model_dict_val
        self.required_tensorkeys_for_function["local_model_dict_val"] = local_model_dict_val

    def reset_opt_vars(self):
        """Reset optimizer variables.

        Returns:
            None
        """
        self.optimizer = self.model_provider.provide_optimizer()

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

    def get_tensor_dict(self, with_opt_vars=False):
        """Return the tensor dictionary.

        Args:
            with_opt_vars (bool): Return the tensor dictionary including the
                optimizer tensors (Default=False).

        Returns:
            dict: Tensor dictionary {**dict, **optimizer_dict}.
        """
        args = [self.model]
        if with_opt_vars:
            args.append(self.optimizer)

        return self.framework_adapter.get_tensor_dict(*args)

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False, device="cpu"):
        """Set the tensor dictionary.

        Args:
            tensor_dict: The tensor dictionary
            with_opt_vars (bool): Return the tensor dictionary including the
                optimizer tensors (Default=False).
        """
        # Sets tensors for model layers and optimizer state.
        # FIXME: self.parameters() instead? Unclear if load_state_dict() or
        #  simple assignment is better
        # for now, state dict gives us names, which is good
        # FIXME: do both and sanity check each time?
        args = [self.model, tensor_dict]
        if with_opt_vars:
            args.append(self.optimizer)

        kwargs = {
            "device": device,
        }

        return self.framework_adapter.set_tensor_dict(*args, **kwargs)
