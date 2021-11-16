# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import tensorflow.keras as ke
import ml_privacy_meter

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from openfl.federated import KerasTaskRunner


class KerasPrivacyMeter(KerasTaskRunner):
    """A strawman for integration with ML Privacy Meter."""

    def __init__(self, **kwargs):
        """
        Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        self.model = self.build_model(self.feature_shape, self.data_loader.num_classes, **kwargs)

        # This will generate the standard dependencies for train / validate
        self.initialize_tensorkeys_for_functions()

        # The following function is where you can add the dependencies for
        # train_attack_model, validate_attack_model, and filter_training_weights
        self.add_tensorkey_dependencies_for_new_functions()

        self.model.summary(print_fn=self.logger.info)

        if self.data_loader is not None:
            self.logger.info(f'Train Set Size : {self.get_train_data_size()}')
            self.logger.info(f'Valid Set Size : {self.get_valid_data_size()}')

        # Initialize ML Privacy Meter
        self.priv_meter_obj = ml_privacy_meter.attack.federated_meminf.initialize(
            target_train_model=self.model,
            target_attack_model=self.model,
            train_datahandler=self.data_loader.attack_data_handler,
            attack_datahandler=self.data_loader.attack_data_handler,
            layers_to_exploit=[5],
            gradients_to_exploit=[2],
            window_size=5,
            device=None, epochs=epochs, model_name='fl_model'
        )

    def add_tensorkey_dependencies_for_new_tasks(self):
        """
        This function adds the new dependencies for functions:
            train_attack_model
            validate_attack_model
            filter_training_weights
        """
        
        output_model_dict = self.get_tensor_dict(with_opt_vars=with_opt_vars)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )

        # Add 'train_attack_model' requirements
        self.required_tensorkeys_for_function['train_attack_model'] = \
            [TensorKey(tensor_name, 'LOCAL', 0, False, ('trained',))
             for tensor_name in {
                 **global_model_dict,
                 **local_model_dict}]

        # Add 'validate_attack_model' requirements
        # TODO If there is a way to easily extract the attack model weights
        # ideally this would be the place to specify those dependencies
        # However, in practice you can just use self.attackobj
        # because the model state will not have changed since running 
        # 'train_attack_model'
        self.required_tensorkeys_for_function['validate_attack_model'] = \
            [TensorKey(tensor_name, 'LOCAL', 0, False, ('trained',))
             for tensor_name in {
                 **global_model_dict,
                 **local_model_dict}]

        # Add 'train_attack_model' requirements
        self.required_tensorkeys_for_function['filter_training_weights'] = \
            [TensorKey(tensor_name, 'LOCAL', 0, False, ('trained',))
             for tensor_name in {
                 **global_model_dict,
                 **local_model_dict}]


    def build_model(self,
                    input_shape,
                    num_classes,
                    conv_kernel_size=(4, 4),
                    conv_strides=(2, 2),
                    conv1_channels_out=16,
                    conv2_channels_out=32,
                    final_dense_inputsize=100,
                    **kwargs):
        """
        Define the model architecture.

        Args:
            input_shape (numpy.ndarray): The shape of the data
            num_classes (int): The number of classes of the dataset

        Returns:
            tensorflow.python.keras.engine.sequential.Sequential: The model defined in Keras

        """
        model = Sequential()

        model.add(Conv2D(conv1_channels_out,
                         kernel_size=conv_kernel_size,
                         strides=conv_strides,
                         activation='relu',
                         input_shape=input_shape))

        model.add(Conv2D(conv2_channels_out,
                         kernel_size=conv_kernel_size,
                         strides=conv_strides,
                         activation='relu'))

        model.add(Flatten())

        model.add(Dense(final_dense_inputsize, activation='relu'))

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=ke.losses.categorical_crossentropy,
                      optimizer=ke.optimizers.Adam(),
                      metrics=['accuracy'])

        # initialize the optimizer variables
        opt_vars = model.optimizer.variables()

        for v in opt_vars:
            v.initializer.run(session=self.sess)

        return model

    def train(self, col_name, round_num, input_tensor_dict, metrics, num_batches=None, **kwargs):
        """
        Perform the training for a specified number of batches.
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

        results = self.train_iteration(self.data_loader.get_train_loader(num_batches),
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

        # create global tensorkeys (these will be stored locally until the ML privacy meter is run on the model)
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
            **output_metric_dict
        }
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict,
            **global_tensorkey_model_dict
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

    def train_privacy_meter_model(self, col_name, round_num, input_tensor_dict, epochs):
        """
        Train an attack model for N epochs.
        Returns
        -------
        Global tensorkey dict - anything to store in aggregator TensorDB
        Local tensorkey dict - Tensorkey:nparray dict to store on local collaborator
            
        """
        # rebuild collaborator model with updated weights
        self.rebuild_model(round_num, input_tensor_dict)
        
        self.priv_meter_obj.train_privacy_meter_for_fl(round_num, self.model)
        
        global_tensorkey_dict = {}
        local_tensorkey_dict = {}
        return global_tensorkey_dict, local_tensorkey_dict

    def test_privacy_meter_model(self, col_name, round_num, input_tensor_dict):
        """
        Validates the previously trained attack model
        Returns
        -------
        Global tensorkey dict - anything to store in aggregator TensorDB
        Local tensorkey dict - Tensorkey:nparray dict to store on local collaborator
            
        """
        # rebuild collaborator model with updated weights
        self.rebuild_model(round_num, input_tensor_dict)

        self.priv_meter_obj.test_privacy_meter_for_fl(round_num)

        #attack_metrics = self.attackobj.test_attack_new_api()
        
        #### the attack metrics can either be added to a tensorkey
        ####  or just saved as an object attribute
        #self.attack_metrics = attack_metrics
        
        global_tensorkey_dict = {}
        local_tensorkey_dict = {}
        return global_tensorkey_dict, local_tensorkey_dict

    def filter_training_weights(self, col_name, round_num, input_tensor_dict):
        """
        Validates the previously trained attack model
        Returns
        -------
        Global tensorkey dict - anything to store in aggregator TensorDB
        Local tensorkey dict - Tensorkey:nparray dict to store on local collaborator
            
        """
        # rebuild collaborator model with updated weights
        self.rebuild_model(round_num, input_tensor_dict)

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )

        tags = ('trained',)

        # create global tensorkeys (these will be stored locally until the ML privacy meter is run on the model)
        global_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags):
                nparray for tensor_name, nparray in global_model_dict.items()
        }
        # no need to restore local tensorkeys (these were previously stored after training)
        local_tensorkey_model_dict = {}

        if self.attack_metrics == model_is_vulnerable:
          # Send nothing or add noise to model
          global_tensorkey_model_dict = {
              tensorkey: nparray + np.random.random(nparray.shape) for tensorkey, nparray in global_tensorkey_model_dict
          }
               
        return global_tensorkey_dict, local_tensorkey_model_dict 

