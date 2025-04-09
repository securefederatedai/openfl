# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from keras.models import model_from_json
from keras_contrib.layers import InstanceNormalization

from openfl.federated import KerasTaskRunner
from keras import backend as K
import gdown
import hashlib

MODEL_JSON_HASH = "c35cfa990000ad87825f182460395ec5d1437a707bd33de4df55d45664e94214"
MODEL_WEIGHTS_HASH = "cd5e52d42e2c6d737e370fb0e673aec5d257134e127c0e59478f11676fa327a5"


def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def compute_file_hash(file_path):
    with open(file_path, "rb") as file:
        data = file.read()
        file_hash = hashlib.sha256(data).hexdigest()
    return file_hash


class KerasHippmapp3r(KerasTaskRunner):
    """A basic convolutional neural network model."""

    def __init__(self, **kwargs):
        """
        Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        weights_id = "1_VEOScLGyr1qV-t-zggq8Lxwgf_z-IpQ"
        gdown.download(id=weights_id, output="model.h5")
        json_id = "1RUE3Cw_rpKnKfwlu75kLbkcr9hde9nV4"
        gdown.download(id=json_id, output="model.json")

        model_json_hash = compute_file_hash("model.json")
        if model_json_hash != MODEL_JSON_HASH:
            raise ValueError("Model JSON file hash does not match expected value.")

        model_weights_hash = compute_file_hash("model.h5")
        if model_weights_hash != MODEL_WEIGHTS_HASH:
            raise ValueError("Model weights file hash does not match expected value.")

        self.model = self.build_model(model_json="model.json", model_weights="model.h5", **kwargs)

        self.initialize_tensorkeys_for_functions()

    def build_model(self, model_json, model_weights, **kwargs):
        """
        Define the model architecture.

        Args:
            model_json (str): Path to model json config
            model_weights (str): Path to model weights

        Returns:
            keras.models.Sequential: The model defined in Keras

        """
        custom_objects = {}
        custom_objects["InstanceNormalization"] = InstanceNormalization
        json_file = open(model_json, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json, custom_objects=custom_objects)
        model.load_weights(model_weights)

        model.compile(loss=dice_coefficient_loss, optimizer="adam", metrics=[dice_coefficient])

        return model
