from logging import getLogger
import functools
import numpy as np

from openfl.utilities import TensorKey, split_tensor_dict_for_holdouts


class CoreTaskRunner(object):
    """Federated Learning Task Runner Class."""

    def register_fl_task(self, task_type='train', task_name='train'):
        """
        allowed types: 'train', 'validate'

        Collaborator interface:
        Task recieves the following: col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs
        Task is expected to return global_tensor_dict, local_tensor_dict 

        User method interface:
        Training Method recieves the following: model, train_loader, device, optimizer
        Validation Method recieves the following: model, val_loader, device

        User Methods return: dict{name: metric}
        """
        def decorator_with_args(training_method):
            @functools.wraps(training_method)
            def wrapper_decorator(col_name, round_num, input_tensor_dict, **kwargs):

                device = kwargs.get('device', 'cpu') # collaborator should know the correct device
                if task_type=='train':
                    self.rebuild_model(input_tensor_dict, validation=False, device=device)
                    loader = self.data_loader.get_train_loader()
                    args_method = [self.model, loader, device, self.optimizer,]
                # elif task_type=='validation':
                else:
                    self.rebuild_model(input_tensor_dict, validation=True, device=device)
                    loader = self.data_loader.get_valid_loader()
                    args_method = [self.model, loader, device,]


                # Here is the training metod call
                metric_dict = training_method(*args_method)
                # Do something after
                
                global_tensor_dict, local_tensor_dict = {}, {}
                origin = col_name
                if task_type=='train':
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
                    self.train_round_completed = True

                elif task_type=='validation':
                    suffix = 'validate'
                    if kwargs['apply'] == 'local':
                        suffix += '_local'
                    else:
                        suffix += '_agg'
                    tags = ('metric', suffix)

                else:
                    tags = ('unknown')

                metric_dict = {
                        TensorKey(metric, origin, round_num, True, tags):
                            np.array(value) for metric, value in metric_dict.items()
                    }
                global_tensor_dict = {**global_tensor_dict, **metric_dict}

                return global_tensor_dict, local_tensor_dict

            # We do not even have to decorate the function 
            self.TASK_REGISTRY[task_name] = wrapper_decorator

        # This line returns the decorator itself
        return decorator_with_args


    def __init__(self, model_provider, **kwargs):
        """
        model_provider should have a method provide_model
        """
        self.framework_adapter = FrameworkAdapterPlugin()

        self.set_logger()
        self.TASK_REGISTRY = dict()
        self.model_provider = model_provider
        self.model = self.model_provider.provide_model()
        self.optimizer = self.model_provider.provide_optimizer()
        self.kwargs = kwargs

        # I do not know what is this
        self.tensor_dict_split_fn_kwargs = dict()
        self.required_tensorkeys_for_function = {}
        self.initialize_tensorkeys_for_functions()

        # Complete hell below
        self.training_round_completed = False
        # overwrite attribute to account for one optimizer param (in every
        # child model that does not overwrite get and set tensordict) that is
        # not a numpy array
        self.tensor_dict_split_fn_kwargs.update({
            'holdout_tensor_names': ['__opt_state_needed']
        })

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
        
    def set_optimizer_treatment(self, opt_treatment):
        """Change the treatment of current instance optimizer."""
        self.opt_treatment = opt_treatment

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
            try:
                return self.required_tensorkeys_for_function[func_name][local_model]
            except:
                print(list(self.required_tensorkeys_for_function.keys()))
                raise
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

import torch as pt

from copy import deepcopy
import numpy as np
class FrameworkAdapterPlugin:
    def __init__(self) -> None:
        pass
        

    @staticmethod
    def get_tensor_dict(model, optimizer=None):
        state = to_cpu_numpy(model.state_dict())

        if not optimizer is None:
            opt_state = _get_optimizer_state(optimizer)
            state = {**state, **opt_state}

        return state

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device='cpu'):
        new_state = {}
        # Grabbing keys from model's state_dict helps to confirm we have
        # everything
        for k in model.state_dict():
            new_state[k] = pt.from_numpy(tensor_dict.pop(k)).to(device)

        # set model state
        model.load_state_dict(new_state)

        if not optimizer is None:
            # see if there is state to restore first
            if tensor_dict.pop('__opt_state_needed') == 'true':
                _set_optimizer_state(optimizer, device, tensor_dict)

            # sanity check that we did not record any state that was not used
            assert len(tensor_dict) == 0


#------------------------------------------------------------    

def _set_optimizer_state(optimizer, device, derived_opt_state_dict):
    """Set the optimizer state.

    Args:
        optimizer:
        device:
        derived_opt_state_dict:

    """
    temp_state_dict = expand_derived_opt_state_dict(
        derived_opt_state_dict, device)

    # FIXME: Figure out whether or not this breaks learning rate
    #  scheduling and the like.
    # Setting default values.
    # All optimizer.defaults are considered as not changing over course of
    # training.
    for group in temp_state_dict['param_groups']:
        for k, v in optimizer.defaults.items():
            group[k] = v

    optimizer.load_state_dict(temp_state_dict)

def _get_optimizer_state(optimizer):
    """Return the optimizer state.

    Args:
        optimizer
    """
    opt_state_dict = deepcopy(optimizer.state_dict())

    # Optimizer state might not have some parts representing frozen parameters
    # So we do not synchronize them
    param_keys_with_state = set(opt_state_dict['state'].keys())
    for group in opt_state_dict['param_groups']:
        local_param_set = set(group['params'])
        params_to_sync = local_param_set & param_keys_with_state
        group['params'] = sorted(list(params_to_sync))

    derived_opt_state_dict = _derive_opt_state_dict(opt_state_dict)

    return derived_opt_state_dict


def _derive_opt_state_dict(opt_state_dict):
    """Separate optimizer tensors from the tensor dictionary.

    Flattens the optimizer state dict so as to have key, value pairs with
    values as numpy arrays.
    The keys have sufficient info to restore opt_state_dict using
    expand_derived_opt_state_dict.

    Args:
        opt_state_dict: The optimizer state dictionary

    """
    derived_opt_state_dict = {}

    # Determine if state is needed for this optimizer.
    if len(opt_state_dict['state']) == 0:
        derived_opt_state_dict['__opt_state_needed'] = 'false'
        return derived_opt_state_dict

    derived_opt_state_dict['__opt_state_needed'] = 'true'

    # Using one example state key, we collect keys for the corresponding
    # dictionary value.
    example_state_key = opt_state_dict['param_groups'][0]['params'][0]
    example_state_subkeys = set(
        opt_state_dict['state'][example_state_key].keys()
    )

    # We assume that the state collected for all params in all param groups is
    # the same.
    # We also assume that whether or not the associated values to these state
    # subkeys is a tensor depends only on the subkey.
    # Using assert statements to break the routine if these assumptions are
    # incorrect.
    for state_key in opt_state_dict['state'].keys():
        assert example_state_subkeys == set(opt_state_dict['state'][state_key].keys())
        for state_subkey in example_state_subkeys:
            assert (isinstance(
                opt_state_dict['state'][example_state_key][state_subkey],
                pt.Tensor)
                == isinstance(
                    opt_state_dict['state'][state_key][state_subkey],
                    pt.Tensor))

    state_subkeys = list(opt_state_dict['state'][example_state_key].keys())

    # Tags will record whether the value associated to the subkey is a
    # tensor or not.
    state_subkey_tags = []
    for state_subkey in state_subkeys:
        if isinstance(
                opt_state_dict['state'][example_state_key][state_subkey],
                pt.Tensor
        ):
            state_subkey_tags.append('istensor')
        else:
            state_subkey_tags.append('')
    state_subkeys_and_tags = list(zip(state_subkeys, state_subkey_tags))

    # Forming the flattened dict, using a concatenation of group index,
    # subindex, tag, and subkey inserted into the flattened dict key -
    # needed for reconstruction.
    nb_params_per_group = []
    for group_idx, group in enumerate(opt_state_dict['param_groups']):
        for idx, param_id in enumerate(group['params']):
            for subkey, tag in state_subkeys_and_tags:
                if tag == 'istensor':
                    new_v = opt_state_dict['state'][param_id][
                        subkey].cpu().numpy()
                else:
                    new_v = np.array(
                        [opt_state_dict['state'][param_id][subkey]]
                    )
                derived_opt_state_dict[
                    '__opt_state_{}_{}_{}_{}'.format(
                        group_idx, idx, tag, subkey)
                ] = new_v
        nb_params_per_group.append(idx + 1)
    # group lengths are also helpful for reconstructing
    # original opt_state_dict structure
    derived_opt_state_dict['__opt_group_lengths'] = np.array(
        nb_params_per_group
    )

    return derived_opt_state_dict



def to_cpu_numpy(state):
    """Send data to CPU as Numpy array.

    Args:
        state

    """
    # deep copy so as to decouple from active model
    state = deepcopy(state)

    for k, v in state.items():
        # When restoring, we currently assume all values are tensors.
        if not pt.is_tensor(v):
            raise ValueError('We do not currently support non-tensors '
                             'coming from model.state_dict()')
        # get as a numpy array, making sure is on cpu
        state[k] = v.cpu().numpy()
    return state