# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Keras Framework Adapter plugin."""
from logging import getLogger

from .framework_adapter_interface import FrameworkAdapterPluginInterface

logger = getLogger(__name__)
class FrameworkAdapterPlugin(FrameworkAdapterPluginInterface):
    """Framework adapter plugin class."""

    def __init__(self) -> None:
        """Initialize framework adapter."""
        pass

    @staticmethod
    def serialization_setup():
        """Prepare model for serialization (optional)."""
        # Source: https://github.com/tensorflow/tensorflow/issues/34697
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.python.keras.layers import deserialize
        from tensorflow.python.keras.layers import serialize
        from tensorflow.python.keras.saving import saving_utils

        def unpack(model, training_config, weights):
            restored_model = deserialize(model)
            if training_config is not None:
                restored_model.compile(
                    **saving_utils.compile_args_from_training_config(
                        training_config
                    )
                )
            restored_model.set_weights(weights)
            return restored_model

        # Hotfix function, not required for TF versions above 2.7.1.
        # https://github.com/keras-team/keras/pull/14748.
        def make_keras_picklable():

            def __reduce__(self):  # NOQA:N807
                model_metadata = saving_utils.model_metadata(self)
                training_config = model_metadata.get('training_config', None)
                model = serialize(self)
                weights = self.get_weights()
                return (unpack, (model, training_config, weights))

            cls = Model
            cls.__reduce__ = __reduce__

        # Run the function
        if tf.__version__ <= '2.7.1':
            logger.warn(f'Applying hotfix for model serialization.'
            'Please consider updating to tensorflow>=2.8 to silence this warning.')
            make_keras_picklable()

    @staticmethod
    def get_tensor_dict(model, optimizer=None, suffix=''):
        """
        Extract tensor dict from a model and an optimizer.

        Returns:
        dict {weight name: numpy ndarray}
        """
        model_weights = _get_weights_dict(model, suffix)

        if optimizer is not None:
            opt_weights = _get_weights_dict(optimizer, suffix)

            model_weights.update(opt_weights)
            if len(opt_weights) == 0:
                # ToDo: warn user somehow
                pass

        return model_weights

    @staticmethod
    def set_tensor_dict(model, tensor_dict, optimizer=None, device='cpu'):
        """
        Set the model weights with a tensor dictionary.

        Args:
            tensor_dict: the tensor dictionary
            with_opt_vars (bool): True = include the optimizer's status.
        """
        model_weight_names = [weight.name for weight in model.weights]
        model_weights_dict = {
            name: tensor_dict[name] for name in model_weight_names
        }
        _set_weights_dict(model, model_weights_dict)

        if optimizer is not None:
            opt_weight_names = [
                weight.name for weight in optimizer.weights
            ]
            opt_weights_dict = {
                name: tensor_dict[name] for name in opt_weight_names
            }
            _set_weights_dict(optimizer, opt_weights_dict)


def _get_weights_dict(obj, suffix=''):
    """
    Get the dictionary of weights.

    Parameters
    ----------
    obj : Model or Optimizer
        The target object that we want to get the weights.

    Returns
    -------
    dict
        The weight dictionary.
    """
    weights_dict = {}
    weight_names = [weight.name for weight in obj.weights]
    weight_values = obj.get_weights()
    for name, value in zip(weight_names, weight_values):
        weights_dict[name + suffix] = value
    return weights_dict


def _set_weights_dict(obj, weights_dict):
    """Set the object weights with a dictionary.

    The obj can be a model or an optimizer.

    Args:
        obj (Model or Optimizer): The target object that we want to set
        the weights.
        weights_dict (dict): The weight dictionary.

    Returns:
        None
    """
    weight_names = [weight.name for weight in obj.weights]
    weight_values = [weights_dict[name] for name in weight_names]
    obj.set_weights(weight_values)
