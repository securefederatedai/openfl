The taskrunner workspace has the following file structure:

taskrunner
├── requirements.txt      # defines the required software packages
├── cert                  # holds trusted certificates
├── data                  # placeholder for each collaborator’s data set
├── save                  # holds the serialized model
├── logs                  # FL experiment logs
└── plan
    ├── plan.yaml         # the Federated Learning plan declaration
    ├── cols.yaml         # holds the list of authorized collaborators
    ├── data.yaml         # holds the collaborator data set path
    ├── defaults          # path to the default values for the FL plan
├── src
    ├── __init__.py       # treat src as a Python package
    └── cnn_model.py      # CNN model for federated learning.
    ├── dataloader.py     # data loader module
    └── taskrunner.py     # task runner module

## Directory Breakdown:
requirements.txt: This file lists all the Python dependencies required to run the TaskRunner API and its components. Ensure you install these dependencies by running pip install -r requirements.txt.

cert: This folder contains trusted certificates for secure communication between federated collaborators.

data: This is a placeholder for each collaborator’s dataset. Each collaborator should store their data in this folder for federated learning.

save: After training, the serialized model is stored here. This folder ensures that your model is preserved across different training sessions and collaborators.

logs: All logs related to Federated Learning experiments, including training progress and task execution, are stored here. Useful for debugging and monitoring.

plan: Contains configuration files for federated learning:

plan.yaml: The main Federated Learning plan declaration, defining the structure of the federated learning workflow.
cols.yaml: A list of authorized collaborators for the federated learning task.
data.yaml: Specifies the path to the data set for each collaborator.
defaults: Path to the default configuration values for the federated learning plan.
src: This directory contains the Python modules used for federated learning:

init.py: Marks the src directory as a Python package, allowing you to import modules within the directory.
cnn_model.py: Defines the Convolutional Neural Network (CNN) model for federated learning.
dataloader.py: A module responsible for loading and processing datasets for the federated learning task.
taskrunner.py: The core task runner module that manages the execution of federated learning tasks.

## Defining the Data Loader
The data loader in OpenFL is responsible for batching and iterating through the dataset that will be used for local training and validation on each collaborator node. The TemplateDataLoader class in src/dataloader.py is designed to be a starting template for creating a data loader that is tailored to the FL experiment’s data format requirements.

To customize the TemplateDataLoader, we just need to implement the load_dataset() function to process the dataset available at data_path on the local file system. The data_path parameter comes from the data.yaml configuration file, which is populated when the collaborator’s identity is created via fx collaborator create.

## Defining the Task Runner
The Task Runner class defines the actual computational tasks of the FL experiment (such as local training and validation). We can implement the placeholders of the TemplateTaskRunner class (src/taskrunner.py) by importing the DigitRecognizerCNN model, as well as the train_epoch() and validate() helper functions from the centralized ML script. The template also provides placeholders for providing custom optimizer and loss function objects.

## Local Simulation of the Federated Learning Experiment
At this point, the FL workspace is ready to be tested in a locally simulated FL environment, before being distributed to all participating entities.

The fx plan initialize command bootstraps the FL workspace by first setting the initial weights of the aggregate model. It then parses the plan, updates the aggregator address if necessary, and produces a hash of the initialized plan for integrity and auditing purposes.

To help OpenFL calculate the initial model weights, we need to provide the shape of the input tensor as an additional parameter. For the MNIST data set of grayscale (single-channel) 28x28 pixel images, the input tensor shape is [1,28,28]. We will also use a locally deployed aggregator (localhost). Thus, the workspace initialization command for our local federation becomes:

fx plan initialize --input_shape [1,28,28] --aggregator_address localhost

The pre-sharded dataset can be downloaded from mnist_data_shards.tar.gz. Copy the dataset bundle to the root of the FL workspace and unpack it:

cp mnist_data_shards.tar.gz ~/openfl-quickstart/fl_workspace
cd ~/openfl-quickstart/fl_workspace
tar -xvf mnist_data_shards.tar.gz
rm mnist_data_shards.tar.gz
This will populate the data folder of the FL workspace with two shards (data/1 and data/2) of labeled MNIST images of digits (the 0–9 labels being encoded in the sub-folder names). Note that in a real-world federation each of the collaborator nodes would only hold one shard, given the decentralized nature of Federated Learning. To facilitate the local testing of the FL workspace, both shards are made available under the local data/ folder:

data
├── 1
    └── mnist_images
        └── 0
        └── 1
        └── 2
        └── 3
        └── 4
        └── 5
        └── 6
        └── 7
        └── 8
        └── 9
├── 2
    └── mnist_images
        └── 0
        └── 1
        └── 2
        └── 3
        └── 4
        └── 5
        └── 6
        └── 7
        └── 8
        └── 9
We can now perform a test run with the following commands for creating a local PKI setup and starting the aggregator and the collaborators on the same machine:

cd ~/openfl-quickstart/fl_workspace

# This will create a local certificate authority (CA), so the participants communicate over a secure TLS Channel
fx workspace certify

#################################################################
# Step 1: Setup the Aggregator #
#################################################################

# Generate a Certificate Signing Request (CSR) for the Aggregator
fx aggregator generate-cert-request --fqdn localhost

# The CA signs the aggregator's request, which is now available in the workspace
fx aggregator certify --fqdn localhost --silent

################################
# Step 2: Setup Collaborator 1 #
################################

# Create a collaborator named "collaborator1" that will use data path "data/1"
# This command adds the collaborator1,data/1 entry in data.yaml
fx collaborator create -n collaborator1 -d data/1

# Generate a CSR for collaborator1
fx collaborator generate-cert-request -n collaborator1

# The CA signs collaborator1's certificate, adding an entry to the authorized cols.yaml
fx collaborator certify -n collaborator1 --silent

################################
# Step 3: Setup Collaborator 2 #
################################

# Create a collaborator named "collaborator2" that will use data path "data/2"
# This command adds the collaborator2,data/2 entry in data.yaml
fx collaborator create -n collaborator2 -d data/2

# Generate a CSR for collaborator2
fx collaborator generate-cert-request -n collaborator2

# The CA signs collaborator2's certificate, adding an entry to the authorized cols.yaml
fx collaborator certify -n collaborator2 --silent

##############################
# Step 4. Run the Federation #
##############################

fx aggregator start & fx collaborator start -n collaborator1 & fx collaborator start -n collaborator2
A successful local simulation of the FL workspace involves the aggregator and collaborators completing a round of training, saving the best-performing model under save/best.pbuf, and exiting with a unanimous “End of Federation reached…”:

INFO     Round: 1, Collaborators that have completed all tasks: ['collaborator2', 'collaborator1']                                 
    METRIC   {'metric_origin': 'aggregator', 'task_name': 'aggregated_model_validation', 'metric_name': 'accuracy', 'metric_value':
              0.8915090382660382, 'round': 1}
    METRIC   Round 1: saved the best model with score 0.891509                                                                          
    METRIC   {'metric_origin': 'aggregator', 'task_name': 'train', 'metric_name': 'training loss', 'metric_value': 0.2952194180338876,  
              'round': 1}
    METRIC   {'metric_origin': 'aggregator', 'task_name': 'locally_tuned_model_validation', 'metric_name': 'accuracy', 'metric_value':  
              0.9181734901767464, 'round': 1}
INFO     Saving round 1 model...                                                                                                    
INFO     Experiment Completed. Cleaning up...                                                                                       
INFO     Waiting for tasks...                                                                                                     
INFO     Sending signal to collaborator collaborator1 to shutdown...                                                                
INFO     End of Federation reached. Exiting...                                                                                    

INFO     Waiting for tasks...                                                                                                     
INFO     Sending signal to collaborator collaborator2 to shutdown...                                                                
INFO     End of Federation reached. Exiting... 
