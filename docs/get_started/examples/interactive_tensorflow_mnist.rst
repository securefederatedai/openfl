.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _interactive_tensorflow_mnist:

Interactive API (Deprecated): MNIST Classification Tutorial
===========================================================

In this tutorial, we will set up a federation and train a basic TensoFlow model on the MNIST dataset using the interactive API.
See `full tutorial <https://github.com/securefederatedai/openfl/tree/f1657abe88632d542504d6d71ca961de9333913f/openfl-tutorials/interactive_api/Tensorflow_MNIST>`_.

**About the dataset**

It is a dataset of 60,000 small square 28x28 pixel grayscale images of handwritten single digits
between 0 and 9. More info at `wiki <https://en.wikipedia.org/wiki/MNIST_database>`_.

.. note::

    This tutorial will be run without TLS and will be done locally as a simulation

-----------------------------------
Step 0: Installation
-----------------------------------
- If you haven't done so already, create a virtual environment, upgrade pip and install OpenFL (See :ref:`install_package`)

-----------------------------------
Step 1: Set up environment
-----------------------------------
Split terminal into 3 (1 terminal for the director, 1 for the envoy, and 1 for the experiment) and activate the virtual environment created in Step 0

.. code-block:: console

    $ source venv/bin/activate

Clone the OpenFL repository:

.. code-block:: console

    $ git clone https://github.com/securefederatedai/openfl.git


Navigate to the tutorial:

.. code-block:: console
    
    $ cd openfl/openfl-tutorials/interactive_api/Tensorflow_MNIST

-----------------------------------
Step 2: Setting up Director
-----------------------------------
In the first terminal, run the director:

.. code-block:: console
    
    $ cd director
    $ ./start_director.sh

-----------------------------------
Step 3: Setting up Envoy
-----------------------------------
In the second terminal, run the envoy:

.. code-block:: console
    
    $ cd envoy
    $ ./start_envoy.sh env_one envoy_config_one.yaml

Optional: Run a second envoy in an additional terminal:

- Ensure steps 0 and 1 are complete for this terminal as well.

- Run the second envoy:

.. code-block:: console
    
    $ cd envoy
    $ ./start_envoy.sh env_two envoy_config_two.yaml

-----------------------------------
Step 4: Run the federation
-----------------------------------
In the third terminal (or forth terminal, if you chose to do two envoys) run the `Tensorflow_MNIST.ipynb` Jupyter Notebook:

.. code-block:: console

    $ cd workspace
    $ jupyter lab Tensorflow_MNIST.ipynb


**Notebook walkthrough:**

Contents of this notebook can be found `here <https://github.com/securefederatedai/openfl/blob/f1657abe88632d542504d6d71ca961de9333913f/openfl-tutorials/interactive_api/Tensorflow_MNIST/workspace/Tensorflow_MNIST.ipynb>`_.

Install additional dependencies if not already installed

.. code-block:: console

    $ pip install tensorflow==2.8

Import:

.. code-block:: python

    import tensorflow as tf
    print('TensorFlow', tf.__version__)

Connect to the Federation

Be sure to start Director and Envoy (Steps 2 and 3) before proceeding with this cell.

This cell connects this notebook to the Federation.

.. code-block:: python

    from openfl.interface.interactive_api.federation import Federation

    # please use the same identificator that was used in signed certificate
    client_id = 'api'
    cert_dir = 'cert'
    director_node_fqdn = 'localhost'
    director_port = 50051

    # Run with TLS disabled (trusted environment)

    # Create a Federation
    federation = Federation(
        client_id=client_id,
        director_node_fqdn=director_node_fqdn,
        director_port=director_port, 
        tls=False
    )

Query Datasets from Shard Registry

.. code-block:: python

    shard_registry = federation.get_shard_registry()
    shard_registry 

.. code-block:: python 

    # First, request a dummy_shard_desc that holds information about the federated dataset 
    dummy_shard_desc = federation.get_dummy_shard_descriptor(size=10)
    dummy_shard_dataset = dummy_shard_desc.get_dataset('train')
    sample, target = dummy_shard_dataset[0]
    f"Sample shape: {sample.shape}, target shape: {target.shape}"

Describing FL experiment

.. code-block:: python

    from openfl.interface.interactive_api.experiment import TaskInterface
    from openfl.interface.interactive_api.experiment import ModelInterface
    from openfl.interface.interactive_api.experiment import FLExperiment

Register model

.. code-block:: python

    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation=None),
    ], name='simplecnn')
    model.summary()

    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)

    # Loss and metrics. These will be used later.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # Create ModelInterface
    framework_adapter = 'openfl.plugins.frameworks_adapters.keras_adapter.FrameworkAdapterPlugin'
    MI = ModelInterface(model=model, optimizer=optimizer, framework_plugin=framework_adapter)

Register dataset

.. code-block:: python

    import numpy as np
    from tensorflow.keras.utils import Sequence

    from openfl.interface.interactive_api.experiment import DataInterface


    class DataGenerator(Sequence):

        def __init__(self, shard_descriptor, batch_size):
            self.shard_descriptor = shard_descriptor
            self.batch_size = batch_size
            self.indices = np.arange(len(shard_descriptor))
            self.on_epoch_end()

        def __len__(self):
            return len(self.indices) // self.batch_size

        def __getitem__(self, index):
            index = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
            batch = [self.indices[k] for k in index]

            X, y = self.shard_descriptor[batch]
            return X, y

        def on_epoch_end(self):
            np.random.shuffle(self.indices)


    class MnistFedDataset(DataInterface):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        @property
        def shard_descriptor(self):
            return self._shard_descriptor

        @shard_descriptor.setter
        def shard_descriptor(self, shard_descriptor):
            """
            Describe per-collaborator procedures or sharding.

            This method will be called during a collaborator initialization.
            Local shard_descriptor will be set by Envoy.
            """
            self._shard_descriptor = shard_descriptor
            
            self.train_set = shard_descriptor.get_dataset('train')
            self.valid_set = shard_descriptor.get_dataset('val')

        def __getitem__(self, index):
            return self.shard_descriptor[index]

        def __len__(self):
            return len(self.shard_descriptor)

        def get_train_loader(self):
            """
            Output of this method will be provided to tasks with optimizer in contract
            """
            if self.kwargs['train_bs']:
                batch_size = self.kwargs['train_bs']
            else:
                batch_size = 32
            return DataGenerator(self.train_set, batch_size=batch_size)

        def get_valid_loader(self):
            """
            Output of this method will be provided to tasks without optimizer in contract
            """
            if self.kwargs['valid_bs']:
                batch_size = self.kwargs['valid_bs']
            else:
                batch_size = 32
            
            return DataGenerator(self.valid_set, batch_size=batch_size)

        def get_train_data_size(self):
            """
            Information for aggregation
            """
            
            return len(self.train_set)

        def get_valid_data_size(self):
            """
            Information for aggregation
            """
            return len(self.valid_set)

Create Mnist federated dataset

.. code-block:: python

    fed_dataset = MnistFedDataset(train_bs=64, valid_bs=512)

Define and register FL tasks

.. code-block:: python

    import time

    TI = TaskInterface()

    # from openfl.interface.aggregation_functions import AdagradAdaptiveAggregation    # Uncomment this lines to use 
    # agg_fn = AdagradAdaptiveAggregation(model_interface=MI, learning_rate=0.4)       # Adaptive Federated Optimization
    # @TI.set_aggregation_function(agg_fn)                                             # alghorithm!
    #                                                                                  # See details in the:
    #                                                                                  # https://arxiv.org/abs/2003.00295

    @TI.register_fl_task(model='model', data_loader='train_dataset', device='device', optimizer='optimizer')     
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

            
        return {'train_acc': train_acc,}


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
                
        return {'validation_accuracy': val_acc,}

Time to start a federated learning experiment

.. code-block:: python

    # create an experimnet in federation
    experiment_name = 'mnist_experiment'
    fl_experiment = FLExperiment(federation=federation, experiment_name=experiment_name,serializer_plugin='openfl.plugins.interface_serializer.keras_seri

.. code-block:: python

    # print the default federated learning plan
    import openfl.native as fx
    print(fx.get_plan(fl_plan=fl_experiment.plan))

.. code-block:: python

    # The following command zips the workspace and python requirements to be transfered to collaborator nodes
    fl_experiment.start(model_provider=MI, 
                    task_keeper=TI,
                    data_loader=fed_dataset,
                    rounds_to_train=5,
                    opt_treatment='CONTINUE_GLOBAL',
                    override_config={'aggregator.settings.db_store_rounds': 1, 'compression_pipeline.template': 'openfl.pipelines.KCPip

.. code-block:: python

    fl_experiment.stream_metrics()
