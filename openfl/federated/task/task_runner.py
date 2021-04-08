from logging import getLogger
import functools
import numpy as np
import os

from openfl.utilities import TensorKey, split_tensor_dict_for_holdouts


class CoreTaskRunner(object):
    """Federated Learning Task Runner Class."""

    def prepare_tensorkeys_for_agggregation(self, metric_dict, validation_flag, col_name, round_num):
        """
        Returns (global_tensor_dict, local_tensor_dict)
        """
        global_tensor_dict, local_tensor_dict = {}, {}
        origin = col_name
        if not validation_flag:
            # Output metric tensors (scalar)
            tags = ('trained',)

            # output model tensors (Doesn't include TensorKey)
            output_model_dict = self.get_tensor_dict(with_opt_vars=True)
            global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
                self.logger, output_model_dict,
                **self.tensor_dict_split_fn_kwargs
            )

            # Create global tensorkeys
            global_tensorkey_model_dict = {
                TensorKey(tensor_name, origin, round_num, False, tags):
                    nparray for tensor_name, nparray in global_model_dict.items()}
            # Create tensorkeys that should stay local
            local_tensorkey_model_dict = {
                TensorKey(tensor_name, origin, round_num, False, tags):
                    nparray for tensor_name, nparray in local_model_dict.items()}
            # The train/validate aggregated function of the next
            # round will look for the updated model parameters.
            # This ensures they will be resolved locally
            next_local_tensorkey_model_dict = {TensorKey(
                tensor_name, origin, round_num + 1, False, ('model',)): nparray
                for tensor_name, nparray in local_model_dict.items()}

            global_tensor_dict = global_tensorkey_model_dict
            local_tensor_dict = {**local_tensorkey_model_dict, **next_local_tensorkey_model_dict}

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
            if self.opt_treatment == 'CONTINUE_GLOBAL':
                self.initialize_tensorkeys_for_functions(with_opt_vars=True)

            # This will signal that the optimizer values are now present,
            # and can be loaded when the model is rebuilt
            self.training_round_completed = True

        else:
            suffix = 'validate' + validation_flag
            tags = ('metric', suffix)

        metric_dict = {
                TensorKey(metric, origin, round_num, True, tags):
                    np.array(value) for metric, value in metric_dict.items()
            }
        global_tensor_dict = {**global_tensor_dict, **metric_dict}

        return global_tensor_dict, local_tensor_dict


    def adapt_tasks(self):
        def task_binder(task_name, callable_task):
            def collaborator_adapted_task(col_name, round_num, input_tensor_dict, **kwargs):
                # print('\n\n',task_name, kwargs, '\n\n')
                task_contract = self.task_provider.task_contract[task_name]
                # Validation flag can be [False, '_local', '_agg']
                validation_flag = True if task_contract['optimizer'] is None else False
                task_settings = self.task_provider.task_settings[task_name]

                cuda_device = 0 if self.data_loader.rank==1 else 2
                device = kwargs.get('device', f'cuda:{cuda_device}')
                
                self.rebuild_model(input_tensor_dict, validation=validation_flag, device=device)
                task_kwargs = dict()
                if validation_flag:  
                    loader = self.data_loader.get_valid_loader() 
                    if kwargs['apply'] == 'local':
                        validation_flag = '_local'
                    else:
                        validation_flag = '_agg'
                else:
                    loader = self.data_loader.get_train_loader()
                    # If train task we also pass optimizer
                    task_kwargs[task_contract['optimizer']] = self.optimizer

                
                for en_name, entity in zip(['model', 'data_loader', 'device'],
                                            [self.model, loader, device]):
                    task_kwargs[task_contract[en_name]] = entity

                # Add task settings to the keyword arguments
                task_kwargs.update(task_settings)

                # Here is the training metod call
                
                metric_dict = callable_task(**task_kwargs)
                
                return self.prepare_tensorkeys_for_agggregation(metric_dict, validation_flag, col_name, round_num)

            return collaborator_adapted_task


        for task_name, callable_task in self.task_provider.task_registry.items():
            self.TASK_REGISTRY[task_name] = task_binder(task_name, callable_task)
                

    def __init__(self, **kwargs):
        """
        model_provider should have a method provide_model
        """
        self.set_logger()

        self.kwargs = kwargs

        self.TASK_REGISTRY = dict()

        # Why is it here
        self.tensor_dict_split_fn_kwargs = dict()
        self.required_tensorkeys_for_function = {}

        # Complete hell below
        self.training_round_completed = False
        # overwrite attribute to account for one optimizer param (in every
        # child model that does not overwrite get and set tensordict) that is
        # not a numpy array
        self.tensor_dict_split_fn_kwargs.update({
            'holdout_tensor_names': ['__opt_state_needed']
        })

    def set_task_provider(self, task_provider):
        if task_provider is None:
            return
        self.task_provider = task_provider
        self.adapt_tasks()

    def set_data_loader(self, data_loader):
        # Is called on a collaborator node
        self.data_loader = data_loader

    def set_model_provider(self, model_provider):
        self.model_provider = model_provider
        self.model = self.model_provider.provide_model()
        self.optimizer = self.model_provider.provide_optimizer()

    def set_framework_adapter(self, framework_adapter):
        self.framework_adapter = framework_adapter
        self.initialize_tensorkeys_for_functions()


    def set_logger(self):
        """Set up the log object."""
        self.logger = getLogger(__name__)

    def set_optimizer_treatment(self, opt_treatment):
        # SHould be removed! We have this info at the initialization time
        # and do not change this one during training.
        """Change the treatment of current instance optimizer."""
        self.opt_treatment = opt_treatment
        
    def rebuild_model(self, input_tensor_dict, validation=False, device='cpu'):
        """
        Parse tensor names and update weights of model. Handles the optimizer treatment.

        Returns:
            None
        """
        if self.opt_treatment == 'RESET':
            self.reset_opt_vars()
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False, device=device)
        elif (self.training_round_completed
              and self.opt_treatment == 'CONTINUE_GLOBAL' and not validation):
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=True, device=device)
        else:
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False, device=device)


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
        if func_name == 'validate':
            local_model = 'apply=' + str(kwargs['apply'])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]


    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Set the required tensors for all publicly accessible methods that \
        could be called as part of a task. \
        By default, this is just all of the layers and optimizer of the model. \
        Custom tensors should be added to this function.

        Args:
            None

        Returns:
            None
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
            validation_global_model_dict = global_model_dict
            validation_local_model_dict = local_model_dict
        else:
            output_model_dict = self.get_tensor_dict(with_opt_vars=False)
            validation_global_model_dict, validation_local_model_dict = \
                split_tensor_dict_for_holdouts(
                    self.logger,
                    output_model_dict,
                    **self.tensor_dict_split_fn_kwargs
                )

        self.required_tensorkeys_for_function['train_batches'] = [
            TensorKey(tensor_name, 'GLOBAL', 0, False, ('model',))
            for tensor_name in global_model_dict]
        self.required_tensorkeys_for_function['train_batches'] += [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('model',))
            for tensor_name in local_model_dict]

        self.required_tensorkeys_for_function['train'] = [
            TensorKey(
                tensor_name, 'GLOBAL', 0, False, ('model',)
            ) for tensor_name in global_model_dict
        ]
        self.required_tensorkeys_for_function['train'] += [
            TensorKey(
                tensor_name, 'LOCAL', 0, False, ('model',)
            ) for tensor_name in local_model_dict
        ]

        # Validation may be performed on local or aggregated (global) model,
        # so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function['validate'] = {}
        # TODO This is not stateless. The optimizer will not be
        self.required_tensorkeys_for_function['validate']['apply=local'] = \
            [TensorKey(
                tensor_name, 'LOCAL', 0, False, ('trained',)
            ) for tensor_name in {
                **validation_global_model_dict,
                **validation_local_model_dict
            }]
        self.required_tensorkeys_for_function['validate']['apply=global'] = \
            [TensorKey(
                tensor_name, 'GLOBAL', 0, False, ('model',)
            ) for tensor_name in validation_global_model_dict]
        self.required_tensorkeys_for_function['validate']['apply=global'] += \
            [TensorKey(
                tensor_name, 'LOCAL', 0, False, ('model',)
            ) for tensor_name in validation_local_model_dict]


    def reset_opt_vars(self):
        """
        Reset optimizer variables.

        Resets the optimizer variables

        """
        self.optimizer = self.model_provider.provide_optimizer()



        
    def get_train_data_size(self):
        """
        Get the number of training examples.

        It will be used for weighted averaging in aggregation.

        Returns:
            int: The number of training examples.
        """
        return self.data_loader.get_train_data_size()

    def get_valid_data_size(self):
        """
        Get the number of examples.

        It will be used for weighted averaging in aggregation.

        Returns:
            int: The number of validation examples.
        """
        return self.data_loader.get_valid_data_size()


#####################################################################################################
    
    def get_tensor_dict(self, with_opt_vars=False):
        """Return the tensor dictionary.

        Args:
            with_opt_vars (bool): Return the tensor dictionary including the
                                  optimizer tensors (Default=False)

        Returns:
            dict: Tensor dictionary {**dict, **optimizer_dict}

        """
        args = [self.model]
        if with_opt_vars:
            args.append(self.optimizer)

        return self.framework_adapter.get_tensor_dict(*args)

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False, device='cpu'):
        """Set the tensor dictionary.

        Args:
            tensor_dict: The tensor dictionary
            with_opt_vars (bool): Return the tensor dictionary including the
                                  optimizer tensors (Default=False)

        """
        # Sets tensors for model layers and optimizer state.
        # FIXME: self.parameters() instead? Unclear if load_state_dict() or
        #  simple assignment is better
        # for now, state dict gives us names, which is good
        # FIXME: do both and sanity check each time?
        args = [self.model, tensor_dict]
        if with_opt_vars:
            args.append(self.optimizer)

        kwargs = {'device': device, }
        
        return self.framework_adapter.set_tensor_dict(*args, **kwargs)