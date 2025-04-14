# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.federated.task.runner import TaskRunner
import numpy as np

class NoOpTaskRunner(TaskRunner):
    """No-op Task Runner Class.

    This class is a no-op implementation of the TaskRunner class. It does not
    perform any operations and is used as a placeholder.
    """

    def __init__(self, data_loader, **kwargs):
        super().__init__(data_loader, **kwargs)

    def train_batches(self, num_batches=None, use_tqdm=False):
        pass

    def validate(self):
        pass

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        return []

    def get_tensor_dict(self, with_opt_vars):
        return {'dummy_tensor': np.float32(1)}

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        pass

    def reset_opt_vars(self):
        pass

    def initialize_globals(self):
        pass

    def set_optimizer_treatment(self, opt_treatment):
        pass

    def initialize_tensorkeys_for_functions(self):
        pass

    def load_native(self, filepath, **kwargs):
        pass

    def save_native(self, filepath, **kwargs):
        pass
