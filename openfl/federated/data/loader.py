# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DataLoader module."""


class DataLoader:
    """Federated Learning Data Loader Class."""

    def __init__(self, **kwargs):
        """
        Instantiate the data object.

        Returns:
            None
        """
        pass

    def get_feature_shape(self):
        """
        Get the shape of an example feature array.

        Returns:
            tuple: shape of an example feature array
        """
        raise NotImplementedError

    def get_train_loader(self, **kwargs):
        """
        Get training data loader.

        Returns:
            loader object (class defined by inheritor)
        """
        raise NotImplementedError

    def get_valid_loader(self):
        """
        Get validation data loader.

        Returns:
            loader object (class defined by inheritor)
        """
        raise NotImplementedError

    def get_infer_loader(self):
        """
        Get inferencing data loader.

        Returns
        -------
        loader object (class defined by inheritor)
        """
        return NotImplementedError

    def get_train_data_size(self):
        """
        Get total number of training samples.

        Returns:
            int: number of training samples
        """
        raise NotImplementedError

    def get_valid_data_size(self):
        """
        Get total number of validation samples.

        Returns:
            int: number of validation samples
        """
        raise NotImplementedError
