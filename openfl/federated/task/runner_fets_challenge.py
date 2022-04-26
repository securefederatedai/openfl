# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GaNDLFTaskRunner module."""

from copy import deepcopy

import numpy as np
import torch as pt

from openfl.utilities import split_tensor_dict_for_holdouts
from openfl.utilities import TensorKey

from .runner import TaskRunner

from GANDLF.compute.generic             import create_pytorch_objects
from GANDLF.compute.training_loop       import train_network
from GANDLF.compute.forward_pass        import validate_network

class FeTSChallengeTaskRunner(TaskRunner):
    """FeTSChallenge Model class for Federated Learning."""

    def __init__(
            self,
            train_csv: str = None,
            val_csv: str = None,
            fets_config_dict: dict = None,
            device: str = None,
            **kwargs
    ):
        """Initialize.
        Args:
            device (string): Compute device (default="cpu")
            **kwargs: Additional parameters to pass to the functions
        """
        super().__init__(**kwargs)

        # if device:
        #     self.device = device
        # else:
        #     self.device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

        model, optimizer, train_loader, val_loader, scheduler, params = create_pytorch_objects(fets_config_dict, train_csv=train_csv, val_csv=val_csv, device=device)
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.params     = params
        self.device     = device

        # pass the actual dataloaders to the wrapper loader
        self.data_loader.set_dataloaders(train_loader, val_loader)

        self.training_round_completed = False

        self.required_tensorkeys_for_function = {}

        # FIXME: why isn't this initial call in runner_pt?
        self.initialize_tensorkeys_for_functions(with_opt_vars=False)

        # overwrite attribute to account for one optimizer param (in every
        # child model that does not overwrite get and set tensordict) that is
        # not a numpy array
        self.tensor_dict_split_fn_kwargs.update({
            'holdout_tensor_names': ['__opt_state_needed']
        })

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

    def validate(self, col_name, round_num, input_tensor_dict,
                 use_tqdm=False, **kwargs):
        """Validate.
        Run validation of the model on the local data.
        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)
            kwargs:              Key word arguments passed to GaNDLF main_run
        Returns:
            global_output_dict:   Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.model.eval()
        # self.model.to(self.device)

        epoch_valid_loss, epoch_valid_metric = validate_network(self.model,
                                                                self.data_loader.val_dataloader,
                                                                self.scheduler,
                                                                self.params,
                                                                round_num,
                                                                mode="validation")

        self.logger.info(epoch_valid_loss)
        self.logger.info(epoch_valid_metric)

        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)

        output_tensor_dict = {}
        output_tensor_dict[TensorKey('valid_loss', origin, round_num, True, tags)] = np.array(epoch_valid_loss)
        for k, v in epoch_valid_metric.items():
            output_tensor_dict[TensorKey(f'valid_{k}', origin, round_num, True, tags)] = np.array(v)

        # Empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def train(self, col_name, round_num, input_tensor_dict, use_tqdm=False, epochs=1, **kwargs):
        """Train batches.
        Train the model on the requested number of batches.
        Args:
            col_name                : Name of the collaborator
            round_num               : What round is it
            input_tensor_dict       : Required input tensors (for model)
            use_tqdm (bool)         : Use tqdm to print a progress bar (Default=True)
            epochs                  : The number of epochs to train
            crossfold_test          : Whether or not to use cross fold trainval/test
                                    to evaluate the quality of the model under fine tuning
                                    (this uses a separate prameter to pass in the data and 
                                    config used)
            crossfold_test_data_csv : Data csv used to define data used in crossfold test.
                                      This csv does not itself define the folds, just
                                      defines the total data to be used.
            crossfold_val_n         : number of folds to use for the train,val level of the nested crossfold.
            corssfold_test_n        : number of folds to use for the trainval,test level of the nested crossfold.
            kwargs                  : Key word arguments passed to GaNDLF main_run
        Returns:
            global_output_dict      : Tensors to send back to the aggregator
            local_output_dict       : Tensors to maintain in the local TensorDB
        """
        self.rebuild_model(round_num, input_tensor_dict)
        # set to "training" mode
        self.model.train()
        # self.model.to(self.device)
        for epoch in range(epochs):
            self.logger.info(f'Run {epoch} epoch of {round_num} round')
            # FIXME: do we want to capture these in an array rather than simply taking the last value?
            epoch_train_loss, epoch_train_metric = train_network(self.model,
                                                                 self.data_loader.train_dataloader,
                                                                 self.optimizer,
                                                                 self.params)

        # output model tensors (Doesn't include TensorKey)
        tensor_dict = self.get_tensor_dict(with_opt_vars=True)

        metric_dict = {'loss': epoch_train_loss}
        for k, v in epoch_train_metric.items():
            metric_dict[f'train_{k}'] = v

        # Return global_tensor_dict, local_tensor_dict
        # is this even pt-specific really?
        global_tensor_dict, local_tensor_dict = create_tensorkey_dicts(tensor_dict,
                                                                       metric_dict,
                                                                       col_name,
                                                                       round_num,
                                                                       self.logger,
                                                                       self.tensor_dict_split_fn_kwargs)

        # Update the required tensors if they need to be pulled from the
        # aggregator
        # TODO this logic can break if different collaborators have different
        # roles between rounds.
        # For example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator because
        # these are only created after training occurs. A work around could
        # involve doing a single epoch of training on random data to get the
        # optimizer names, and then throwing away the model.
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        # This will signal that the optimizer values are now present,
        # and can be loaded when the model is rebuilt
        self.train_round_completed = True

        # Return global_tensor_dict, local_tensor_dict
        return global_tensor_dict, local_tensor_dict

    def get_tensor_dict(self, with_opt_vars=False):
        """Return the tensor dictionary.
        Args:
            with_opt_vars (bool): Return the tensor dictionary including the
                                  optimizer tensors (Default=False)
        Returns:
            dict: Tensor dictionary {**dict, **optimizer_dict}
        """
        # Gets information regarding tensor model layers and optimizer state.
        # FIXME: self.parameters() instead? Unclear if load_state_dict() or
        # simple assignment is better
        # for now, state dict gives us names which is good
        # FIXME: do both and sanity check each time?

        state = to_cpu_numpy(self.model.state_dict())

        if with_opt_vars:
            opt_state = _get_optimizer_state(self.optimizer)
            state = {**state, **opt_state}

        return state

    def _get_weights_names(self, with_opt_vars=False):
        # Gets information regarding tensor model layers and optimizer state.
        # FIXME: self.parameters() instead? Unclear if load_state_dict() or
        # simple assignment is better
        # for now, state dict gives us names which is good
        # FIXME: do both and sanity check each time?

        state = self.model.state_dict().keys()

        if with_opt_vars:
            opt_state = _get_optimizer_state(self.model.optimizer)
            state += opt_state.keys()

        return state

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False):
        """Set the tensor dictionary.
        Args:
            tensor_dict: The tensor dictionary
            with_opt_vars (bool): Return the tensor dictionary including the
                                  optimizer tensors (Default=False)
        """
        set_pt_model_from_tensor_dict(self.model, tensor_dict, self.device, with_opt_vars)

    def get_optimizer(self):
        """Get the optimizer of this instance."""
        return self.optimizer

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """
        Get the required tensors for specified function that could be called \
        as part of a task. By default, this is just all of the layers and \
        optimizer of the model.
        Args:
            func_name
        Returns:
            list : [TensorKey]
        """
        if func_name == 'validate':
            local_model = 'apply=' + str(kwargs['apply'])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Set the required tensors for all publicly accessible task methods.
        By default, this is just all of the layers and optimizer of the model.
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
            global_model_dict_val = global_model_dict
            local_model_dict_val = local_model_dict
        else:
            output_model_dict = self.get_tensor_dict(with_opt_vars=False)
            global_model_dict_val, local_model_dict_val = split_tensor_dict_for_holdouts(
                self.logger,
                output_model_dict,
                **self.tensor_dict_split_fn_kwargs
            )

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
        self.required_tensorkeys_for_function['validate']['apply=local'] = [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('trained',))
            for tensor_name in {
                **global_model_dict_val,
                **local_model_dict_val
            }]
        self.required_tensorkeys_for_function['validate']['apply=global'] = [
            TensorKey(tensor_name, 'GLOBAL', 0, False, ('model',))
            for tensor_name in global_model_dict_val
        ]
        self.required_tensorkeys_for_function['validate']['apply=global'] += [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('model',))
            for tensor_name in local_model_dict_val
        ]

    def load_native(self, filepath, model_state_dict_key='model_state_dict',
                    optimizer_state_dict_key='optimizer_state_dict', **kwargs):
        """
        Load model and optimizer states from a pickled file specified by \
        filepath. model_/optimizer_state_dict args can be specified if needed. \
        Uses pt.load().
        Args:
            filepath (string)                 : Path to pickle file created
                                                by pt.save().
            model_state_dict_key (string)     : key for model state dict
                                                in pickled file.
            optimizer_state_dict_key (string) : key for optimizer state dict
                                                in picked file.
            kwargs                            : unused
        Returns:
            None
        """
        pickle_dict = pt.load(filepath)
        self.model.load_state_dict(pickle_dict[model_state_dict_key])
        self.optimizer.load_state_dict(pickle_dict[optimizer_state_dict_key])

    def save_native(self, filepath, model_state_dict_key='model_state_dict',
                    optimizer_state_dict_key='optimizer_state_dict', **kwargs):
        """
        Save model and optimizer states in a picked file specified by the \
        filepath. model_/optimizer_state_dicts are stored in the keys provided. \
        Uses pt.save().
        Args:
            filepath (string)                 : Path to pickle file to be
                                                created by pt.save().
            model_state_dict_key (string)     : key for model state dict
                                                in pickled file.
            optimizer_state_dict_key (string) : key for optimizer state
                                                dict in picked file.
            kwargs                            : unused
        Returns:
            None
        """
        pickle_dict = {
            model_state_dict_key: self.model.state_dict(),
            optimizer_state_dict_key: self.optimizer.state_dict()
        }
        pt.save(pickle_dict, filepath)

    def reset_opt_vars(self):
        """
        Reset optimizer variables.
        Resets the optimizer variables
        """
        pass


def create_tensorkey_dicts(tensor_dict, metric_dict, col_name, round_num, logger, tensor_dict_split_fn_kwargs):
    origin = col_name
    tags = ('trained',)
    output_metric_dict = {}
    for k, v in metric_dict.items():
        tk = TensorKey(k, origin, round_num, True, ('metric',))
        output_metric_dict[tk] = np.array(v)

    global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
        logger, tensor_dict, **tensor_dict_split_fn_kwargs
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
    # The train/validate aggregated function of the next round will look
    # for the updated model parameters.
    # This ensures they will be resolved locally
    next_local_tensorkey_model_dict = {
        TensorKey(tensor_name, origin, round_num + 1, False, ('model',)): nparray
        for tensor_name, nparray in local_model_dict.items()}

    global_tensor_dict = {
        **output_metric_dict,
        **global_tensorkey_model_dict
    }
    local_tensor_dict = {
        **local_tensorkey_model_dict,
        **next_local_tensorkey_model_dict
    }

    return global_tensor_dict, local_tensor_dict


def set_pt_model_from_tensor_dict(model, tensor_dict, device, with_opt_vars=False):
    """Set the tensor dictionary.
    Args:
        model: the pytorch nn.module object
        tensor_dict: The tensor dictionary
        device: the device where the tensor values need to be sent
        with_opt_vars (bool): Return the tensor dictionary including the
                                optimizer tensors (Default=False)
    """
    # Sets tensors for model layers and optimizer state.
    # FIXME: model.parameters() instead? Unclear if load_state_dict() or
    #  simple assignment is better
    # for now, state dict gives us names, which is good
    # FIXME: do both and sanity check each time?

    new_state = {}
    # Grabbing keys from model's state_dict helps to confirm we have
    # everything
    for k in model.state_dict():
        new_state[k] = pt.from_numpy(tensor_dict.pop(k)).to(device)

    # set model state
    model.load_state_dict(new_state)

    if with_opt_vars:
        # see if there is state to restore first
        if tensor_dict.pop('__opt_state_needed') == 'true':
            _set_optimizer_state(model.get_optimizer(), device, tensor_dict)

        # sanity check that we did not record any state that was not used
        assert len(tensor_dict) == 0

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
                derived_opt_state_dict[f'__opt_state_{group_idx}_{idx}_{tag}_{subkey}'] = new_v
        nb_params_per_group.append(idx + 1)
    # group lengths are also helpful for reconstructing
    # original opt_state_dict structure
    derived_opt_state_dict['__opt_group_lengths'] = np.array(
        nb_params_per_group
    )

    return derived_opt_state_dict


def expand_derived_opt_state_dict(derived_opt_state_dict, device):
    """Expand the optimizer state dictionary.
    Takes a derived opt_state_dict and creates an opt_state_dict suitable as
    input for load_state_dict for restoring optimizer state.
    Reconstructing state_subkeys_and_tags using the example key
    prefix, "__opt_state_0_0_", certain to be present.
    Args:
        derived_opt_state_dict: Optimizer state dictionary
    Returns:
        dict: Optimizer state dictionary
    """
    state_subkeys_and_tags = []
    for key in derived_opt_state_dict:
        if key.startswith('__opt_state_0_0_'):
            stripped_key = key[16:]
            if stripped_key.startswith('istensor_'):
                this_tag = 'istensor'
                subkey = stripped_key[9:]
            else:
                this_tag = ''
                subkey = stripped_key[1:]
            state_subkeys_and_tags.append((subkey, this_tag))

    opt_state_dict = {'param_groups': [], 'state': {}}
    nb_params_per_group = list(
        derived_opt_state_dict.pop('__opt_group_lengths').astype(np.int)
    )

    # Construct the expanded dict.
    for group_idx, nb_params in enumerate(nb_params_per_group):
        these_group_ids = [f'{group_idx}_{idx}' for idx in range(nb_params)]
        opt_state_dict['param_groups'].append({'params': these_group_ids})
        for this_id in these_group_ids:
            opt_state_dict['state'][this_id] = {}
            for subkey, tag in state_subkeys_and_tags:
                flat_key = f'__opt_state_{this_id}_{tag}_{subkey}'
                if tag == 'istensor':
                    new_v = pt.from_numpy(derived_opt_state_dict.pop(flat_key))
                else:
                    # Here (for currrently supported optimizers) the subkey
                    # should be 'step' and the length of array should be one.
                    assert subkey == 'step'
                    assert len(derived_opt_state_dict[flat_key]) == 1
                    new_v = int(derived_opt_state_dict.pop(flat_key))
                opt_state_dict['state'][this_id][subkey] = new_v

    # sanity check that we did not miss any optimizer state
    assert len(derived_opt_state_dict) == 0

    return opt_state_dict


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
        group['params'] = sorted(params_to_sync)

    derived_opt_state_dict = _derive_opt_state_dict(opt_state_dict)

    return derived_opt_state_dict


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
