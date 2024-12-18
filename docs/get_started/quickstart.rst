.. # Copyright (C) 2020-2024 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _quick_start:

=====================
Quick Start
=====================

|productName| has a variety of APIs to choose from when setting up and running a federation. 
In this quick start guide, we will demonstrate how to run a simple federated learning example using the Task Runner API.



.. _creating_a_federation:

********************************
Creating a federation in 5 steps
********************************

To begin we recommend installing the latest OpenFL inside a python virtual environment. This can be done with the following:

.. code-block:: console:

    pip install virtualenv
    virtualenv ~/openfl-quickstart
    source ~/openfl-quickstart/bin/activate
    git clone https://github.com/securefederatedai/openfl.git
    cd openfl
    pip install .


Now you're ready to run your first federation! Copying these commands to your terminal will run a simple federation with an aggregator and two collaborators all on your local machine. These commands can be broken down into five steps, which you can read more about `here <../about/features_index/taskrunner.html#step-1-create-a-workspace>`_

1. Setup Federation Workspace & Certificate Authority (CA) for Secure Communication
2. Setup Aggregator & Initialize Federation Plan + Model
3. Setup Collaborator 1
4. Setup Collaborator 2
5. Run the Federation

.. code-block:: console

   ############################################################################################
   # Step 1: Setup Federation Workspace & Certificate Authority (CA) for Secure Communication #
   ############################################################################################

   # Generate an OpenFL Workspace. This example will train a pytorch
   # CNN model on the MNIST dataset
   fx workspace create --template torch_cnn_mnist --prefix my_workspace
   cd my_workspace
  
   # This will create a certificate authority (CA), so the participants communicate over a secure TLS Channel
   fx workspace certify

   #################################################################
   # Step 2: Setup Aggregator & Initialize Federation Plan + Model #
   #################################################################

   # Generate a Certificate Signing Request (CSR) for the Aggregator
   fx aggregator generate-cert-request --fqdn localhost

   # The CA signs the aggregator's request, which is now available in the workspace
   fx aggregator --fqdn localhost certify --silent

   # Initialize FL Plan and Model Weights for the Federation
   fx plan initialize --aggregator_address localhost

   ################################
   # Step 3: Setup Collaborator 1 #
   ################################

   # Create a collaborator named "collaborator1" that will use data path "1"
   fx collaborator create -n collaborator1 -d 1 

   # Generate a CSR for collaborator1
   fx collaborator generate-cert-request -n collaborator1

   # The CA signs collaborator1's certificate 
   fx collaborator certify -n collaborator1 --silent

   ################################
   # Step 4: Setup Collaborator 2 #
   ################################

   # Create a collaborator named "collaborator2" that will use data path "2"
   fx collaborator create -n collaborator2 -d 2 

   # Generate a CSR for collaborator2
   fx collaborator generate-cert-request -n collaborator2

   # The CA signs collaborator2's certificate 
   fx collaborator certify -n collaborator2 --silent

   ##############################
   # Step 5. Run the Federation #
   ##############################

   # Run the Aggregator
   fx aggregator start &

   # Run Collaborator 1
   fx collaborator start -n collaborator1 &

   # Run Collaborator 2 
   fx collaborator start -n collaborator2

   echo "Congratulations! You've run your first federation with OpenFL"


You should see this output at the end of the experiment:

.. code-block:: console

              INFO     Starting round 9...                                                                                                        aggregator.py:897
   [15:36:28] INFO     Waiting for tasks...                                                                                                     collaborator.py:178
              INFO     Sending tasks to collaborator collaborator2 for round 9                                                                    aggregator.py:329
              INFO     Received the following tasks: [name: "aggregated_model_validation"                                                       collaborator.py:143
                       , name: "train"
                       , name: "locally_tuned_model_validation"
                       ]
   [15:36:30] METRIC   Round 9, collaborator collaborator2 is sending metric for task aggregated_model_validation: accuracy    0.983597         collaborator.py:415
   [15:36:31] INFO     Collaborator collaborator2 is sending task results for aggregated_model_validation, round 9                                aggregator.py:520
              METRIC   Round 9, collaborator validate_agg aggregated_model_validation result accuracy: 0.983597                                   aggregator.py:559
   [15:36:31] INFO     Run 0 epoch of 9 round                                                                                                      runner_pt.py:148
   [15:36:31] INFO     Waiting for tasks...                                                                                                     collaborator.py:178
              INFO     Sending tasks to collaborator collaborator1 for round 9                                                                    aggregator.py:329
              INFO     Received the following tasks: [name: "aggregated_model_validation"                                                       collaborator.py:143
                       , name: "train"
                       , name: "locally_tuned_model_validation"
                       ]
   [15:36:33] METRIC   Round 9, collaborator collaborator1 is sending metric for task aggregated_model_validation: accuracy    0.981000         collaborator.py:415
   [15:36:34] INFO     Collaborator collaborator1 is sending task results for aggregated_model_validation, round 9                                aggregator.py:520
              METRIC   Round 9, collaborator validate_agg aggregated_model_validation result accuracy: 0.981000                                   aggregator.py:559
   [15:36:34] INFO     Run 0 epoch of 9 round                                                                                                      runner_pt.py:148
   [15:36:34] METRIC   Round 9, collaborator collaborator2 is sending metric for task train: cross_entropy     0.059750                         collaborator.py:415
   [15:36:35] INFO     Collaborator collaborator2 is sending task results for train, round 9                                                      aggregator.py:520
              METRIC   Round 9, collaborator metric train result cross_entropy:        0.059750                                                   aggregator.py:559
   [15:36:35] METRIC   Round 9, collaborator collaborator2 is sending metric for task locally_tuned_model_validation: accuracy 0.979596         collaborator.py:415
              INFO     Collaborator collaborator2 is sending task results for locally_tuned_model_validation, round 9                             aggregator.py:520
              METRIC   Round 9, collaborator validate_local locally_tuned_model_validation result accuracy:    0.979596                           aggregator.py:559
              INFO     Waiting for tasks...                                                                                                     collaborator.py:178
   [15:36:37] METRIC   Round 9, collaborator collaborator1 is sending metric for task train: cross_entropy     0.019203                         collaborator.py:415
   [15:36:38] INFO     Collaborator collaborator1 is sending task results for train, round 9                                                      aggregator.py:520
              METRIC   Round 9, collaborator metric train result cross_entropy:        0.019203                                                   aggregator.py:559
   [15:36:38] METRIC   Round 9, collaborator collaborator1 is sending metric for task locally_tuned_model_validation: accuracy 0.977600         collaborator.py:415
              INFO     Collaborator collaborator1 is sending task results for locally_tuned_model_validation, round 9                             aggregator.py:520
              METRIC   Round 9, collaborator validate_local locally_tuned_model_validation result accuracy:    0.977600                           aggregator.py:559
              METRIC   Round 9, aggregator: train <openfl.interface.aggregation_functions.weighted_average.WeightedAverage object at              aggregator.py:838
                       0x7f329a98bee0> cross_entropy:    0.039476
   [15:36:39] METRIC   Round 9, aggregator: aggregated_model_validation <openfl.interface.aggregation_functions.weighted_average.WeightedAverage  aggregator.py:838
                       object at 0x7f329a98bee0> accuracy:   0.982298
              METRIC   Round 9: saved the best model with score 0.982298                                                                          aggregator.py:854
              METRIC   Round 9, aggregator: locally_tuned_model_validation                                                                        aggregator.py:838
                       <openfl.interface.aggregation_functions.weighted_average.WeightedAverage object at 0x7f329a98bee0> accuracy:
                       0.978598
              INFO     Saving round 10 model...                                                                                                   aggregator.py:890
              INFO     Experiment Completed. Cleaning up...                                                                                       aggregator.py:895
   [15:36:39] INFO     Waiting for tasks...                                                                                                     collaborator.py:178
              INFO     Sending signal to collaborator collaborator1 to shutdown...                                                                aggregator.py:283
              INFO     End of Federation reached. Exiting...                                                                                    collaborator.py:150
   
    ✔ OK
   [15:36:46] INFO     Waiting for tasks...                                                                                                     collaborator.py:178
   [15:36:46] INFO     Sending signal to collaborator collaborator2 to shutdown...                                                                aggregator.py:283
              INFO     End of Federation reached. Exiting...                                                                                    collaborator.py:150
   
    ✔ OK
   
   Congratulations! You've run your first federation with OpenFL

***************************
Working with your own model
***************************

Now that you've run your first federation, let's see how to replace the model used in the federation. After copying in the text above, you should be in the :code:`my_workspace` directory. Every workspace has a :code:`src` directory that contains the Task Runner, an OpenFL interface that defines the deep learning model, as well as the training and validation functions that will run on that model. In this case, the Task Runner is defined in :code:`src/taskrunner.py`. After opening it you'll see the following:

.. code-block:: python

    class PyTorchCNN(PyTorchTaskRunner):
        """
        Simple CNN for classification.
        
        PyTorchTaskRunner inherits from nn.module, so you can define your model
        in the same way that you would for PyTorch
        """
    
        def __init__(self, device='cpu', **kwargs):
            """Initialize.
    
            Args:
                device: The hardware device to use for training (Default = "cpu")
                **kwargs: Additional arguments to pass to the function
    
            """
            super().__init__(device=device, **kwargs)
    
            ####################################
            #       Your model goes here       #
            ####################################
            self.conv1 = nn.Conv2d(1, 20, 2, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(800, 500)
            self.fc2 = nn.Linear(500, 10)
            self.to(device)
            ####################################
    
            ######################################################################
            #                    Your optimizer goes here                        #
            #                                                                    # 
            # `self.optimizer` must be set for optimizer weights to be federated #
            ######################################################################
            self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
    
            # Set the loss function
            self.loss_fn = F.cross_entropy
    
    
        def forward(self, x):
            """
            Forward pass of the model.
    
            Args:
                x: Data input to the model for the forward pass
            """
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 800)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

:code:`PyTorchTaskRunner` inherits from :code:`nn.module`, so changing your deep learning model is as easy as modifying the network layers (i.e. :code:`self.conv1`, etc.) into the :code:`__init__` function, and then defining your :code:`forward` function. You'll notice that unlike PyTorch, the optimizer is also defined in this :code:`__init__` function. This is so the model AND optimizer weights can be distributed as part of the federation.  

******************************************
Defining your own train and validate tasks
******************************************

If you continue scrolling down in :code:`src/taskrunner.py`, you'll see two functions: :code:`train_` and :code:`validate_`. These are the primary tasks performed by the collaborators that have access to local data. 

.. code-block:: python

    def train_(self, train_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]) -> Metric:
        """
        Train single epoch.

        Override this function in order to use custom training.

        Args:
            train_dataloader: Train dataset batch generator. Yields (samples, targets) tuples of
            size = `self.data_loader.batch_size`.
        Returns:
            Metric: An object containing name and np.ndarray value.
        """
        losses = []
        for data, target in train_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        loss = np.mean(losses)
        return Metric(name=self.loss_fn.__name__, value=np.array(loss))


    def validate_(self, validation_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]) -> Metric:
        """
        Perform validation on PyTorch Model

        Override this function for your own custom validation function

        Args:
            validation_dataloader: Validation dataset batch generator. Yields (samples, targets) tuples
        Returns:
            Metric: An object containing name and np.ndarray value
        """

        total_samples = 0
        val_score = 0
        with torch.no_grad():
            for data, target in validation_dataloader:
                samples = target.shape[0]
                total_samples += samples
                data, target = data.to(self.device), target.to(self.device, dtype=torch.int64)
                output = self(data)
                # get the index of the max log-probability
                pred = output.argmax(dim=1)
                val_score += pred.eq(target).sum().cpu().numpy()
        
        accuracy = val_score / total_samples
        return Metric(name='accuracy', value=np.array(accuracy))

Each function is passed a dataloader, and returns a :code:`Metric` associated with that task. In this example the :code:`train_` function returns the Cross Entropy Loss for an epoch, and the :code:`validate_` function returns the accuracy. You'll see these metrics reported when running the collaborator locally, and the aggregator will report the average metrics coming from all collaborators. 

*****************************
Defining your own data loader
*****************************

Now let's look at the OpenFL :code:`PyTorchDataLoader` and see how by subclassing it we are able to split the MNIST dataset across collaborators for training. You'll find the following defined in :code:`src/dataloader.py`.


.. code-block:: python

    from openfl.federated import PyTorchDataLoader
    
    class PyTorchMNISTInMemory(PyTorchDataLoader):
        """PyTorch data loader for MNIST dataset."""
    
        def __init__(self, data_path, batch_size, **kwargs):
            """Instantiate the data object.
    
            Args:
                data_path: The file path to the data
                batch_size: The batch size of the data loader
                **kwargs: Additional arguments, passed to super
                 init and load_mnist_shard
            """
            super().__init__(batch_size, **kwargs)
    
            num_classes, X_train, y_train, X_valid, y_valid = load_mnist_shard(
                shard_num=int(data_path), **kwargs)
    
            self.X_train = X_train
            self.y_train = y_train
            self.train_loader = self.get_train_loader()
    
            self.X_valid = X_valid
            self.y_valid = y_valid
            self.val_loader = self.get_valid_loader()
    
            self.num_classes = num_classes

This example uses the classic MNIST dataset for digit recognition. For in-memory datasets, the :code:`data_path` is passed a number to determine which slice of the dataset the collaborator should receive. By initializing the :code:`train_loader` (:code:`self.train_loader = self.get_train_loader()`) and the :code:`val_loader` (:code:`self.val_loader = self.get_valid_loader()`), these dataloader will then be able to be passed into the :code:`train_` and :code:`validate_` functions defined above.

***************************************
Changing the number of federated rounds
***************************************

Now that we've seen how to change the code, let's explore the Federated Learning Plan (FL Plan). The plan, which is defined in :code:`plan/plan.yaml`, is used to configure everything about the federation that can't purely be expressed in python. This includes information like network connectivity details, how different components are configured, and how many rounds the federation should train. Different experiments may take more rounds to train depending on how similar data is between collaborators, the model, and the number of collaborators that participate. To tweak this parameter for your experiment, open :code:`plan/plan.yaml` and modify the following section:

.. code-block:: yaml

    aggregator:
      settings:
        best_state_path: save/torch_cnn_mnist_best.pbuf
        db_store_rounds: 2
        init_state_path: save/torch_cnn_mnist_init.pbuf
        last_state_path: save/torch_cnn_mnist_last.pbuf
        log_metric_callback:
          template: src.utils.write_metric
        rounds_to_train: 10 # Change this value to train for a different number of rounds
        write_logs: true

*****************************************************
Starting a new federation after making custom changes
*****************************************************

Now that you've changed a few things, you can rerun the federation. Copying the below text will reinitialize your plan with new model weights, and relaunch the aggregator and two collaborators:

.. code-block:: console

    fx plan initialize
    fx aggregator start &
    fx collaborator start -n collaborator1 &
    fx collaborator start -n collaborator2

Well done! Now that you know the basics of using the Task Runner API to run OpenFL on a single node, check out some of the other :ref:`openfl_examples` for research purposes and in production.
