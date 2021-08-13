import logging
from time import sleep
import getpass

import tensorflow as tf

from openfl.interface.interactive_api.experiment import ModelInterface, FLExperiment
from openfl.interface.interactive_api.federation import Federation
# from openfl.services.tests.experiment_runner import run_experiment
from tests.github.interactive_api_director.experiment_runner import run_federation
from tests.github.interactive_api_director.experiment_runner import stop_federation
from tests.github.interactive_api_director.experiment_runner import Shard
from tests.github.interactive_api_director.experiment_runner import create_federation
from openfl.transport.grpc.director_client import DirectorClient
from tests.github.interactive_api.experiments.tensorflow_mnist.settings import model, optimizer, X_train, y_train, X_valid, y_valid, batch_size
from tests.github.interactive_api.experiments.tensorflow_mnist.dataset import FedDataset
from tests.github.interactive_api.experiments.tensorflow_mnist.tasks import train, validate, task_interface

logger = logging.getLogger(__name__)


# create federation
col_names = ['one', 'two']
username = getpass.getuser()
director_path = f'/home/{username}/test/exp_1/director'

director_addr = 'localhost'
director_port = 50051

shards = {
    f'/home/{username}/test/exp_1/{col_name}':
        Shard(
            shard_name=col_name,
            director_addr=director_addr,
            director_port=director_port,
            data_path=f'/home/{username}/test/data/{col_name}'
        )
    for col_name in col_names
}

create_federation(director_path, shards.keys())

processes = run_federation(shards, director_path)

input('Please enter to run first experiment')

experiment_name = 'tensorflow_mnist'
# Describing FL experiment
framework_adapter = 'openfl.plugins.frameworks_adapters.keras_adapter.FrameworkAdapterPlugin'
model_interface = ModelInterface(
    model=model, optimizer=optimizer, framework_plugin=framework_adapter)

# Register dataset
fed_dataset = FedDataset(X_train, y_train, X_valid, y_valid, batch_size=batch_size)

# Perform model warm up
# The model warmup is necessary to initialize weights when using Tensorflow Gradient Tape

train(model, fed_dataset.get_train_loader(), optimizer, 'cpu', warmup=True)

#Make a copy of the model for later comparison
initial_model = tf.keras.models.clone_model(model)


# Prepare Federated Dataset for Serialization
# tf.data.DataSet does not serialize well with pickle.
# It will be recreated on the collaborators with the delayed init function
fed_dataset.train_dataset = None
fed_dataset.valid_dataset = None


# Start a federated learning experiment

# Create a federation
# will determine fqdn by itself
federation = Federation(director_node_fqdn='localhost', tls=False)
# Datapath corresonds to 'RANK,WORLD_SIZE'
col_data_paths = {
    'one': '1,2',
    'two': '2,2'
}
# federation.register_collaborators(col_data_paths=col_data_paths)

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


sleep(2)

director_client = DirectorClient(
    director_addr=director_addr,
    director_port=director_port
)
resp = director_client.set_new_experiment(experiment_name, col_names, arch_path,
                                          model_interface, fl_experiment)
logger.info(f'Response from director: {resp}')

# fl_experiment.start_experiment(model_interface)

# best_model = fl_experiment.get_best_model()
# fed_dataset._delayed_init()
#
# logger.info('Validating initial model')
# validate(initial_model, fed_dataset.get_valid_loader(), 'cpu')
#
# logger.info('Validating trained model')
# validate(best_model, fed_dataset.get_valid_loader(), 'cpu')

while True:
    sleep(1)

input('Press Enter to run second experiment')

# Second experiment

from tests.github.interactive_api_director.experiments.pytorch_kvasir_unet.experiment import (
    arch_path, validate, fed_dataset, fl_experiment, model_interface, initial_model)

resp = director_client.set_new_experiment('pytorch_kvasir_unet', col_names, arch_path)
logger.info(f'Response from director: {resp}')

fl_experiment.start_experiment(model_interface)

best_model = fl_experiment.get_best_model()
fed_dataset._delayed_init()

logger.info('Validating initial model')
validate(initial_model, fed_dataset.get_valid_loader(), 'cpu')
validate(initial_model, fed_dataset.get_valid_loader(), 'cpu')

logger.info('Validating trained model')
validate(best_model, fed_dataset.get_valid_loader(), 'cpu')

stop_federation(processes)
