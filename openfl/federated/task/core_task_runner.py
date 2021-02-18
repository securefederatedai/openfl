from logging import getLogger


class CoreTaskRunner(object):
    """Federated Learning Task Runner Class."""

    def __init__(self, data_loader, tensor_dict_split_fn_kwargs={}, **kwargs):
        self.set_logger()

    def set_logger(self):
        """Set up the log object."""
        self.logger = getLogger(__name__)

    def set_optimizer_treatment(self, opt_treatment):
        # SHould be removed! We have this info at the initialization time
        # and do not change this one during training.
        """Change the treatment of current instance optimizer."""
        self.opt_treatment = opt_treatment
        
    def rebuild_model(self, round_num, input_tensor_dict, validation=False):
        """
        Parse tensor names and update weights of model. Handles the optimizer treatment.

        Returns:
            None
        """
        if self.opt_treatment == 'RESET':
            self.reset_opt_vars()
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)
        elif (self.training_round_completed
              and self.opt_treatment == 'CONTINUE_GLOBAL' and not validation):
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=True)
        else:
            self.set_tensor_dict(input_tensor_dict, with_opt_vars=False)


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


#####################################################################################################
    
    def get_tensor_dict(self, with_opt_vars=False):
        """Return the tensor dictionary.

        Args:
            with_opt_vars (bool): Return the tensor dictionary including the
                                  optimizer tensors (Default=False)

        Returns:
            dict: Tensor dictionary {**dict, **optimizer_dict}

        """
        pass

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False):
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
        pass

    def reset_opt_vars(self):
        """
        Reset optimizer variables.

        Resets the optimizer variables

        """
        pass