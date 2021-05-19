# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experiment description."""

import logging

import tensorflow as tf

from openfl.interface.interactive_api.experiment import ModelInterface, FLExperiment
from openfl.interface.interactive_api.federation import Federation
from tests.github.interactive_api.experiment_runner import run_experiment
from tests.github.interactive_api.experiments.tensorflow_mnist.settings import model
from tests.github.interactive_api.experiments.tensorflow_mnist.settings import optimizer
from tests.github.interactive_api.experiments.tensorflow_mnist.settings import X_train
from tests.github.interactive_api.experiments.tensorflow_mnist.settings import y_train
from tests.github.interactive_api.experiments.tensorflow_mnist.settings import X_valid
from tests.github.interactive_api.experiments.tensorflow_mnist.settings import y_valid
from tests.github.interactive_api.experiments.tensorflow_mnist.settings import batch_size
from tests.github.interactive_api.experiments.tensorflow_mnist.dataset import FedDataset
from tests.github.interactive_api.experiments.tensorflow_mnist.tasks import train
from tests.github.interactive_api.experiments.tensorflow_mnist.tasks import validate
from tests.github.interactive_api.experiments.tensorflow_mnist.tasks import task_interface

logger = logging.getLogger(__name__)


# Describing FL experiment
framework_adapter = 'openfl.plugins.frameworks_adapters.keras_adapter.FrameworkAdapterPlugin'
model_interface = ModelInterface(
    model=model, optimizer=optimizer, framework_plugin=framework_adapter)

# Register dataset
fed_dataset = FedDataset(X_train, y_train, X_valid, y_valid, batch_size=batch_size)

# Perform model warm up
# The model warmup is necessary to initialize weights when using Tensorflow Gradient Tape

train(model, fed_dataset.get_train_loader(), optimizer, 'cpu', warmup=True)

# Make a copy of the model for later comparison
initial_model = tf.keras.models.clone_model(model)


# Prepare Federated Dataset for Serialization
# tf.data.DataSet does not serialize well with pickle.
# It will be recreated on the collaborators with the delayed init function
fed_dataset.train_dataset = None
fed_dataset.valid_dataset = None


# Start a federated learning experiment

# Create a federation
# will determine fqdn by itself
federation = Federation(central_node_fqdn='localhost', disable_tls=True)
# Datapath corresonds to 'RANK,WORLD_SIZE'
col_data_paths = {
    'one': '1,2',
    'two': '2,2'
}
federation.register_collaborators(col_data_paths=col_data_paths)

# create an experimnet in federation
fl_experiment = FLExperiment(federation=federation)

# If I use autoreload I got a pickling error
arch_path = fl_experiment.prepare_workspace_distribution(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=fed_dataset,
    rounds_to_train=7,
    opt_treatment='CONTINUE_GLOBAL'
)

run_experiment(col_data_paths, model_interface, arch_path, fl_experiment)

best_model = fl_experiment.get_best_model()
fed_dataset._delayed_init()

logger.info('Validating initial model')
validate(initial_model, fed_dataset.get_valid_loader(), 'cpu')

logger.info('Validating trained model')
validate(best_model, fed_dataset.get_valid_loader(), 'cpu')
