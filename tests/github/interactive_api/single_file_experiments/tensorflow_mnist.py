# Describe the model and optimizer
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

logger = logging.getLogger(__name__)

inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Prepare data

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

X_valid = x_train[-10000:]
y_valid = y_train[-10000:]
X_train = x_train[:-10000]
y_train = y_train[:-10000]

# Describing FL experiment

from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, \
    ModelInterface, FLExperiment

framework_adapter = 'openfl.plugins.frameworks_adapters.keras_adapter.FrameworkAdapterPlugin'
model_interface = ModelInterface(model=model, optimizer=optimizer,
                                 framework_plugin=framework_adapter)


# Register dataset

class FedDataset(DataInterface):
    def __init__(self, x_train, y_train, x_valid, y_valid, **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.batch_size = kwargs['batch_size']
        self.kwargs = kwargs
        self._setup_datasets()

    def _setup_datasets(self):
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        self.valid_dataset = tf.data.Dataset.from_tensor_slices((self.X_valid, self.y_valid))
        self.valid_dataset = self.valid_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

    def _delayed_init(self, data_path='1,1'):
        # With the next command the local dataset will be loaded on the collaborator node
        # For this example we have the same dataset on the same path, and we will shard it
        # So we use `data_path` information for this purpose.
        self.rank, self.world_size = [int(part) for part in data_path.split(',')]

        # Do the actual sharding
        self._do_sharding(self.rank, self.world_size)

    def _do_sharding(self, rank, world_size):
        self.X_train = self.X_train[rank - 1:: world_size]
        self.y_train = self.y_train[rank - 1:: world_size]
        self.X_valid = self.X_valid[rank - 1:: world_size]
        self.y_valid = self.y_valid[rank - 1:: world_size]
        self._setup_datasets()

    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        return self.train_dataset

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        return self.valid_dataset

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        return len(self.X_train)

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.X_valid)


fed_dataset = FedDataset(X_train, y_train, X_valid, y_valid, batch_size=batch_size)

# Register tasks

TI = TaskInterface()

import time


@TI.register_fl_task(model='model', data_loader='train_dataset',
                     device='device', optimizer='optimizer')
def train(model, train_dataset, optimizer, device, loss_fn=loss_fn, warmup=False):
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * 64))
        if warmup:
            break

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    return {'train_acc': train_acc, }


@TI.register_fl_task(model='model', data_loader='val_dataset', device='device')
def validate(model, val_dataset, device):
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))

    return {'validation_accuracy': val_acc, }


# Perform model warm up
# The model warmup is necessary to initialize weights when using Tensorflow Gradient Tape

train(model, fed_dataset.get_train_loader(), optimizer, 'cpu', warmup=True)

# Make a copy of the model for later comparison
initial_model = tf.keras.models.clone_model(model)

# Prepare Federated Dataset for Serialization
# tf.data.DataSet does not serialize well with pickle. It will be recreated on the
# collaborators with the delayed init function
fed_dataset.train_dataset = None
fed_dataset.valid_dataset = None

# Start a federated learning experiment

# Create a federation
from openfl.interface.interactive_api.federation import Federation

# will determine fqdn by itself
federation = Federation(central_node_fqdn='localhost', tls=False)
# Datapath corresonds to 'RANK,WORLD_SIZE'
col_data_paths = {'one': '1,2',
                  'two': '2,2'}
federation.register_collaborators(col_data_paths=col_data_paths)

# create an experimnet in federation
fl_experiment = FLExperiment(federation=federation)

# If I use autoreload I got a pickling error
arch_path = fl_experiment.prepare_workspace_distribution(
    model_provider=model_interface,
    task_keeper=TI,
    data_loader=fed_dataset,
    rounds_to_train=7,
    opt_treatment='CONTINUE_GLOBAL'
)


from tests.github.interactive_api.experiment_runner import run_experiment


run_experiment(col_data_paths, model_interface, arch_path, fl_experiment)

best_model = fl_experiment.get_best_model()
fed_dataset._delayed_init()

logger.info('Validating initial model')
validate(initial_model, fed_dataset.get_valid_loader(), 'cpu')

logger.info('Validating trained model')
validate(best_model, fed_dataset.get_valid_loader(), 'cpu')
