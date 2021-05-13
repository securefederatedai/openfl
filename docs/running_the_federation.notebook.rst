.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _running_notebook:

Learning to fly: |productName| tutorials
#######################

New to |productName|? Get familiar with our native Python API using the built-in tutorials. After installing the |productName| package in your virtual environment, simply run :code:`fx tutorial start` from the command line. This will start a Jupyter notebook server and return a URL you can use to access each of our tutorials. We provide several jupyter notebooks for Pytorch and Tensorflow that simulate a federation on a local machine.  These tutorials provide a convient entrypoint for learning about |productName| :ref:`conventions <definitions_and_conventions>`  like FL Plans, aggregators, collaborators and more. 


Starting the tutorials
~~~~~~~~~~~~~~~~~

1. Make sure you have initialized the virtual environment and can run the :code:`fx` command.

2. Run :code:`fx tutorial start`. This will start a jupyter server on your machine. 

3. Copy the URL (including token) to your browser

4. Choose a tutorial from which to start. Each of these represent simulated federated learning training demos. The existing tutorials are:

 - :code:`Federated Keras MNIST Tutorial`: workspace with a simple `Keras <http://keras.io/>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.
 - :code:`Federated Pytorch MNIST Tutorial`: workspace with a simple `PyTorch <https://pytorch.org/>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.

|productName| Python API Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a first step to using the |productName| Python API, add the following lines to your python script:

    .. code-block:: python

     import openfl.native as fx
     from openfl.federated import FederatedModel,FederatedDataSet

This will load the |productName| package and import wrappers that adapt your existing data and models to a (simulated) federated context. To setup |productName| for a basic experiment, just run :code:`fx.init()`. This command will create a new workspace directory containing default plan values for your experiments, and setup a two collaborator experiment (the collaborators are creatively named 'one' and 'two'). If you want to create an experiment with a large number of collaborators, this can be done programmatically as follows:

    .. code-block:: python

     collaborator_list = [str(i) for i in range(NUM_COLLABORATORS)]
     fx.init('keras_cnn_mnist',col_names=collaborator_list)


One last point about :code:`fx.init()`. For Keras models, we recommend starting with the :code:`keras_cnn_mnist` template (by running :code:`fx.init("keras_cnn_mnist")`, and for pytorch models `torch_cnn_mnist` (by running :code:`fx.init("torch_cnn_mnist")`)

At this point you may be wondering what goes into a FL.Plan, and how you can customize it. To see what is part of the FL.Plan that was created with the :code:`fx.init` command, run :code:`fx.get_plan()`:

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

The :code:`fx.get_plan()` command returns all of the plan values that can be set. If you wish to change any of them, these can be provided at experiment runtime in the :code:`override-config` parameter of :code:`fx.run_experiment`, or ahead of time with :code:`fx.update_plan()`. Based on the plan returned above, we see that this experiment will run for 10 rounds. If we wanted to train for 20 rounds instead, we could provide that overriden key value pair as follows:

    .. code-block:: python

     #Set values ahead of time with fx.update_plan() 
     fx.update_plan({"aggregator.settings.rounds_to_train": 20})

     #Or set values at experiment runtime
     fx.run_experiment(experiment_collaborators,override_config={"aggregator.settings.rounds_to_train": 20})


Now that our workspace has been created and know the plan for the experiment, we can actually wrap the data and model. :code:`FederatedDataSet` wraps in-memory numpy datasets and includes a setup function that will split the data into N mutually-exclusive chunks for each collaborator participating in the experiment. 

    .. code-block:: python

     fl_data = FederatedDataSet(train_images,train_labels,valid_images,valid_labels,batch_size=32,num_classes=classes)

Similarly, the :code:`FederatedModel` wrapper takes as an argument your model definition. If you have a Tensorflow/Keras model, wrap it in a function that outputs the fully compiled model (as in the example below):

    .. code-block:: python

     def build_model(feature_shape,classes):
         #Defines the MNIST model
         model = Sequential()
         model.add(Dense(64, input_shape=feature_shape, activation='relu'))
         model.add(Dense(64, activation='relu'))
         model.add(Dense(classes, activation='softmax'))
         
         model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'],)
         return model 

     fl_model = FederatedModel(build_model,data_loader=fl_data)

If you have a Pytorch model, there are three parameters that should be passed to the :code:`FederatedModel`: The class that defines the network definition and associated forward function, lambda optimizer method that can be set to a newly instantiated network, and finally the loss function. See below for an example:

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

     fl_model = FederatedModel(build_model=Net,optimizer=optimizer,loss_fn=cross_entropy,data_loader=fl_data)


Now we just need to define which collaborators (that were created with :code:`fx.init()`) will take part in the experiment. If you want to use the same collaborator list, this can be done in a single line with a dictionary comprehension:

    .. code-block:: python

     experiment_collaborators = {col_name:col_model for col_name,col_model \
                                      in zip(collaborator_list,fl_model.setup(len(collaborator_list)))}

This command will create a model for each collaborator each their data slice. In production deployments of |productName|, each collaborator will have the data on premise, and the splitting of data into shards is not necessary.

We are now ready to run our experiment!

    .. code-block:: python

     final_fl_model = fx.run_experiment(experiment_collaborators,override_config={"aggregator.settings.rounds_to_train": 5})

This will run the experiment for five rounds, and return the final model once it has completed. 
