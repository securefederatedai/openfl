.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_notebook:

**********************************
Aggregator-Based Workflow Tutorial (Deprecated)
**********************************

You will start a Jupyter\* \  lab server and receive a URL you can use to access the tutorials. Jupyter notebooks are provided for PyTorch\* \  and TensorFlow\* \  that simulate a federation on a local machine.

.. note::

	Follow the procedure to become familiar with the APIs used in aggregator-based workflow and conventions such as *FL Plans*, *Aggregators*, and *Collaborators*. 
	

Start the Tutorials
===================

1. Start a Python\* \  3.9 (>=3.9, <3.12) virtual environment and confirm |productName| is available.

    .. code-block:: python

		fx
    
    You should see a list of available commands

2. Start a Jupyter server. This returns a URL to access available tutorials.

	.. code-block:: python

		fx tutorial start

3. Open the URL (including the token) in your browser.

4. Choose a tutorial from which to start. Each tutorial is a demonstration of a simulated federated learning. The following are examples of available tutorials:

 - :code:`Federated Keras MNIST Tutorial`: workspace with a simple `Keras <http://keras.io/>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.
 - :code:`Federated Pytorch MNIST Tutorial`: workspace with a simple `PyTorch <https://pytorch.org/>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.
 - :code:`Federated PyTorch UNET Tutorial`: workspace with a UNET `PyTorch <https://pytorch.org/>`_ model that will download the `Hyper-Kvasir <https://datasets.simula.no/hyper-kvasir/>`_ dataset and train in a federation.
 - :code:`Federated PyTorch TinyImageNet`: workspace with a MobileNet-V2 `PyTorch <https://pytorch.org/>`_ model that will download the `Tiny-ImageNet <https://www.kaggle.com/c/tiny-imagenet/>`_ dataset and train in a federation.


Familiarize with the API Concepts in an Aggregator-Based Worklow
================================================================

Step 1: Enable the |productName| Python API
-------------------------------------------

Add the following lines to your Python script.

    .. code-block:: python

     import openfl.native as fx
     from openfl.federated import FederatedModel, FederatedDataSet

This loads the |productName| package and import wrappers that adapt your existing data and models to a (simulated) federated context.

Step 2: Set Up the Experiment
-----------------------------

For a basic experiment, run the following command.

    .. code-block:: python

     fx.init()
	 
	 
This creates a workspace directory containing default FL plan values for your experiments, and sets up a an experiment with two collaborators (the collaborators are creatively named **one** and **two**).

For an experiment with more collaborators, run the following command.

    .. code-block:: python

     collaborator_list = [str(i) for i in range(NUM_COLLABORATORS)]
     fx.init('keras_cnn_mnist', col_names=collaborator_list)


.. note::

	The following are template recommendations for training models:
	
	- For Keras models, run :code:`fx.init('keras_cnn_mnist')` to start with the *keras_cnn_mnist* template.
	- For PyTorch models, run :code:`fx.init('torch_cnn_mnist')` to start with the *torch_cnn_mnist* template.
	

Step 3: Customize the Federated Learning Plan (FL Plan)
-------------------------------------------------------

For this example, the experiment is set up with the *keras_cnn_mnist* template.	

   .. code-block:: python

		fx.init('keras_cnn_mnist')
	 

See the FL plan values that can be set with the :code:`fx.get_plan()` command.

    .. code-block:: python

     print(fx.get_plan())

     {
       "aggregator.settings.best_state_path": "save/keras_cnn_mnist_best.pbuf",
       "aggregator.settings.init_state_path": "save/keras_cnn_mnist_init.pbuf",
       "aggregator.settings.last_state_path": "save/keras_cnn_mnist_last.pbuf",
       "aggregator.settings.rounds_to_train": 10,
       "aggregator.template": "openfl.component.Aggregator",
       ...
     }

Based on this plan values, the experiment will run for 10 rounds. You can customize the experiment to run for 20 rounds either at runtime or ahead of time.

Set the value at **runtime** with the :code:`override-config` parameter of :code:`fx.run_experiment`.

    .. code-block:: python

     #set values at experiment runtime
     fx.run_experiment(experiment_collaborators, override_config={"aggregator.settings.rounds_to_train": 20})


Set the value **ahead of time** with :code:`fx.update_plan()`.

    .. code-block:: python

     #Set values ahead of time with fx.update_plan() 
     fx.update_plan({"aggregator.settings.rounds_to_train": 20})


Step 4: Wrap the Data and Model
-------------------------------

Use the :code:`FederatedDataSet` function to wrap in-memory numpy datasets and split the data into N mutually-exclusive chunks for each collaborator participating in the experiment.

    .. code-block:: python

     fl_data = FederatedDataSet(train_images, train_labels, valid_images, valid_labels, batch_size=32, num_classes=classes)

Similarly, the :code:`FederatedModel` function takes as an argument your model definition. For the first example, you can wrap a Keras model in a function that outputs the compiled model.

**Example 1:**

    .. code-block:: python

     def build_model(feature_shape,classes):
         #Defines the MNIST model
         model = Sequential()
         model.add(Dense(64, input_shape=feature_shape, activation='relu'))
         model.add(Dense(64, activation='relu'))
         model.add(Dense(classes, activation='softmax'))
         
         model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
         return model 

     fl_model = FederatedModel(build_model, data_loader=fl_data)

For the second example with a PyTorch model, the :code:`FederatedModel` function takes the following parameters: 

- The class that defines the network definition and associated forward function
- The lambda optimizer method that can be set to a newly instantiated network
- The loss function

**Example 2:**

    .. code-block:: python

     class Net(nn.Module):
         def __init__(self):
             super(Net, self).__init__()
             self.conv1 = nn.Conv2d(1, 16, 3)
             self.pool = nn.MaxPool2d(2, 2)
             self.conv2 = nn.Conv2d(16, 32, 3)
             self.fc1 = nn.Linear(32 * 5 * 5, 32)
             self.fc2 = nn.Linear(32, 84)
             self.fc3 = nn.Linear(84, 10)

         def forward(self, x):
             x = self.pool(F.relu(self.conv1(x)))
             x = self.pool(F.relu(self.conv2(x)))
             x = x.view(x.size(0),-1)
             x = F.relu(self.fc1(x))
             x = F.relu(self.fc2(x))
             x = self.fc3(x)
             return F.log_softmax(x, dim=1)
    
     optimizer = lambda x: optim.Adam(x, lr=1e-4)
     
     def cross_entropy(output, target):
         """Binary cross-entropy metric
         """
         return F.binary_cross_entropy_with_logits(input=output,target=target)

     fl_model = FederatedModel(build_model=Net, optimizer=optimizer, loss_fn=cross_entropy, data_loader=fl_data)


Step 5: Define the Collaborators
--------------------------------

Define the collaborators taking part in the experiment. The example below uses the collaborator list, created earlier with the the :code:`fx.init()` command.

    .. code-block:: python

     experiment_collaborators = {col_name:col_model for col_name, col_model \
                                      in zip(collaborator_list, fl_model.setup(len(collaborator_list)))}

This command creates a model for each collaborator with their data shard.

.. note::

	In production deployments of |productName|, each collaborator will have the data on premise. Splitting data into shards is not necessary.

Step 6: Run the Experiment
--------------------------

Run the experiment for five rounds and return the final model once completed.

    .. code-block:: python

     final_fl_model = fx.run_experiment(experiment_collaborators, override_config={"aggregator.settings.rounds_to_train": 5})