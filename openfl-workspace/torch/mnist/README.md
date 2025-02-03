## Instantiating a Workspace from Torch Template
To instantiate a workspace from the torch/mnist template, you can use the fx workspace create command. This allows you to quickly set up a new workspace based on a predefined configuration and template.

1. Ensure the necessary dependencies are installed.
```
pip install virtualenv
mkdir ~/openfl-quickstart
virtualenv ~/openfl-quickstart/venv
source ~/openfl-quickstart/venv/bin/activate
pip install openfl
```
2. Creating the Workspace Folder

```
cd ~/openfl-quickstart
fx workspace create --template torch/mnist --prefix fl_workspace
cd ~/openfl-quickstart/fl_workspace
```

## Directory Structure
The taskrunner workspace has the following file structure:
```
taskrunner
├── requirements.txt      # defines the required software packages
└── plan
    ├── plan.yaml         # the Federated Learning plan declaration
    ├── cols.yaml         # holds the list of authorized collaborators
    ├── data.yaml         # holds the collaborator data set path
    ├── defaults          # path to the default values for the FL plan
├── src
    ├── __init__.py       # treat src as a Python package
    └── cnn_model.py      # centralized CNN model, ready for use in federated learning
    ├── dataloader.py     # data loader module
    └── taskrunner.py     # task runner module
```

## Directory Breakdown:
* requirements.txt: Lists all the Python dependencies required to run the TaskRunner API and its components. Ensure you install these dependencies by running pip install -r requirements.txt.
* plan: Contains configuration files for federated learning:
    - plan.yaml: The main Federated Learning plan declaration, defining the structure of the federated learning workflow.
    - cols.yaml: A list of authorized collaborators for the federated learning task.
    - data.yaml: Specifies the path to the data set for each collaborator.
    - defaults: Path to the default configuration values for the federated learning plan.
* src: Contains the Python modules used for federated learning:
    - init.py: Marks the src directory as a Python package, allowing you to import modules within the directory.
    - cnn_model.py: Defines the Convolutional Neural Network (CNN) model for federated learning.
    - dataloader.py: A module responsible for loading and processing datasets for the federated learning task.
    - taskrunner.py: The core task runner module that manages the execution of federated learning tasks.

## Defining the Data Loader
The data loader in OpenFL is responsible for batching and iterating through the dataset that will be used for local training and validation on each collaborator node. The PyTorchMNISTInMemory class is responsible for batching and iterating through the MNIST data set, additionally sharded "on the fly".

To customize the PyTorchMNISTInMemory class, you need to implement the load_mnist_shard() function to process the dataset available at data_path on the local file system. The data_path parameter represents the data shard number used by the collaborator. This setup allows each collaborator to work with a specific subset of the data, facilitating distributed training.

The load_mnist_shard() function is responsible for loading the MNIST dataset, dividing it into training and validation sets, and applying necessary transformations. The data is then batched and made ready for the training process.

# Modify the dataloader to support "Bring Your Own Data"
You can either try to implement the placeholders by yourself, or get the solution from [dataloader.py](https://github.com/securefederatedai/openfl-contrib/blob/main/openfl_contrib_tutorials/ml_to_fl/federated/src/dataloader.py)
Also, update the data loader class name in plan.yaml accordingly.
```
import numpy as np
from typing import Iterator, Tuple
from openfl.federated import PyTorchTaskRunner
from openfl.utilities import Metric
import torch.optim as optim
import torch.nn.functional as F
from src.cnn_model import DigitRecognizerCNN, train_epoch, validate

class MNISTShardDataLoader(PyTorchDataLoader):

    def __init__(self, data_path, batch_size, **kwargs):
        super().__init__(batch_size, **kwargs)

        # Load the dataset using the provided data_path and any additional kwargs.
        X_train, y_train, X_valid, y_valid = load_dataset(data_path, **kwargs)

        # Assign the loaded data to instance variables.
        self.X_train = X_train
        self.y_train = y_train

        self.X_valid = X_valid
        self.y_valid = y_valid

def load_dataset(data_path, train_split_ratio=0.8, **kwargs):
    dataset = MNISTDataset(
        root=data_path, 
        transform=Compose([Grayscale(num_output_channels=1), ToTensor()])
    )
    n_train = int(train_split_ratio * len(dataset))
    n_valid = len(dataset) - n_train

    ds_train, ds_val = random_split(
        dataset, lengths=[n_train, n_valid], generator=manual_seed(0))

    X_train, y_train = list(zip(*ds_train))
    X_train, y_train = np.stack(X_train), np.array(y_train)

    X_valid, y_valid = list(zip(*ds_val))
    X_valid, y_valid = np.stack(X_valid), np.array(y_valid)

    return X_train, y_train, X_valid, y_valid

class MNISTDataset(ImageFolder):
    """Encapsulates the MNIST dataset"""

    FOLDER_NAME = "mnist_images"
    DEFAULT_PATH = path.join(path.expanduser('~'), '.openfl', 'data')

    def __init__(self, root: str = DEFAULT_PATH, **kwargs) -> None:
        """Initialize."""
        makedirs(root, exist_ok=True)

        super(MNISTDataset, self).__init__(
            path.join(root, MNISTDataset.FOLDER_NAME), **kwargs)

    def __getitem__(self, index):
        """Allow getting items by slice index."""
        if isinstance(index, Iterable):
            return [super(MNISTDataset, self).__getitem__(i) for i in index]
        else:
            return super(MNISTDataset, self).__getitem__(index)
```
Use a pre-sharded version of the standard MNIST dataset (divided among two data owners). The pre-sharded dataset can be downloaded from [mnist_data_shards.tar.gz](https://github.com/securefederatedai/openfl-contrib/blob/main/openfl_contrib_tutorials/ml_to_fl/federated/mnist_data_shards.tar.gz). 

Copy the dataset bundle to the root of the FL workspace and unpack it:
```
cp mnist_data_shards.tar.gz ~/openfl-quickstart/fl_workspace
cd ~/openfl-quickstart/fl_workspace
tar -xvf mnist_data_shards.tar.gz
rm mnist_data_shards.tar.gz
```
This will populate the ```/data``` folder of the FL workspace with two shards (data/1 and data/2) of labeled MNIST images of digits (the 0–9 labels being encoded in the sub-folder names). 
```
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
```

## Defining the Task Runner
The Task Runner class defines the actual computational tasks of the FL experiment (such as local training and validation). We can implement the placeholders of the TemplateTaskRunner class (src/taskrunner.py) by importing the DigitRecognizerCNN model, as well as the train_epoch() and validate() helper functions from the centralized ML script. The template also provides placeholders for providing custom optimizer and loss function objects.

## How to run this tutorial (local simulation):
The fx plan initialize command bootstraps the workspace by first setting the initial weights of the aggregate model. It then parses the plan, updates the aggregator address if necessary, and produces a hash of the initialized plan for integrity and auditing purposes.

To help OpenFL calculate the initial model weights, we need to provide the shape of the input tensor as an additional parameter. For the MNIST data set of grayscale (single-channel) 28x28 pixel images, the input tensor shape is [1,28,28]. We will also use a locally deployed aggregator (localhost). Thus, the workspace initialization command for our local federation becomes:

```
mkdir save
fx plan initialize --input_shape [1,28,28] --aggregator_address localhost
```

We can now perform a test run with the following commands for creating a local PKI setup and starting the aggregator and the collaborators on the same machine:

```
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
fx collaborator create -n collaborator1 -d 1

# Generate a CSR for collaborator1
fx collaborator generate-cert-request -n collaborator1

# The CA signs collaborator1's certificate, adding an entry to the authorized cols.yaml
fx collaborator certify -n collaborator1 --silent

################################
# Step 3: Setup Collaborator 2 #
################################

# Create a collaborator named "collaborator2" that will use data path "data/2"
# This command adds the collaborator2,data/2 entry in data.yaml
fx collaborator create -n collaborator2 -d 2

# Generate a CSR for collaborator2
fx collaborator generate-cert-request -n collaborator2

# The CA signs collaborator2's certificate, adding an entry to the authorized cols.yaml
fx collaborator certify -n collaborator2 --silent

##############################
# Step 4. Run the Federation #
##############################

fx aggregator start & fx collaborator start -n collaborator1 & fx collaborator start -n collaborator2
```

A successful local simulation of the FL workspace involves the aggregator and collaborators completing a round of training, saving the best-performing model under save/best.pbuf, and exiting with a unanimous “End of Federation reached…”:

## Sample output
```
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
```