

## 'Core' TaskRunner interfaces before refactoring

### Initializing interface to TaskRunner:
__init__(self, model_provider, **kwargs) - Currently called from the code, should be initialized properly.
.set_data_loader(data_loader) - Called after initialization by Plan on the collaborator node
.set_framework_adapter(FrameworkAdapterPlugin) - Called after initialization by Plan on the collaborator node

### Collaborator interface to TaskRunner:
* .set_optimizer_treatment(self.opt_treatment.name) - during initialization
* .get_required_tensorkeys_for_function(func_name, **kwargs) - before every task call
* .TASK_REGISTRY[func_name] - obtaining task callable, every task call
* .get_train_data_size() - called every train task end sending results
* .get_valid_data_size() - called every valid task end sending results

### TaskRunner interface to framework adapters:
#### The following work with the model and optimizer whatever they are
* .framework_adapter.get_tensor_dict(*args)
* .set_tensor_dict(*args, **kwargs)

### TaskRunner interface to DataLoader:
* .data_loader.get_train_loader() - called before every train task
* .data_loader.get_valid_loader() - called before every valid task
* .get_train_data_size() - forwarded to the Collaborator
* .get_train_valid_size() - forwarded to the Collaborator

## Refactored TaskRunner interface:


## Workspace
To initialize the workspace, create an empty folder and a Jupyter notebook (or a Python script) inside it. Root folder of the notebook will be considered as the workspace.
If some objects are imported in the notebook from local modules, source code should be kept inside the workspace.
If someone decides to keep local test data inside the workspace, `data` folder should be used as it will not be exported.
If someone decides to keep certificates inside the workspace, `cert` folder should be used as it will not be exported.
Only relevant source code or resources should be kept inside the workspace, since it will be zipped and transferred to collaborator machines.

## Python Environment
Create a virtual Python environment. Please, install only packages that are required for conducting the experiment, since Python environment will be replicated on collaborator nodes.

## Defining a Federated Learning Experiment
Interactive API allows setting up an experiment from a single entrypoint - a Jupyter notebook or a Python script.
Defining an experiment includes setting up several interface entities and experiment parameters.

### Federation API
*Federation* entity is introduced to register and keep information about collaborators settings and their local data, as well as network settings to enable communication inside the federation. 
Each federation is bound to some Machine Learning problem in a sense that all collaborators dataset shards should follow the same annotation format for all samples. Once you created a federation, it may be used in several subsequent experiments.

To set up a federation, use Federation Interactive API.

`from openfl.interface.interactive_api.federation import Federation`
1. Federation API class should be initialized with the aggregator node FQDN and encryption settings. Someone may disable mTLS in trusted environments or provide paths to the certificate chain of CA, aggregator certificate and private key to enable mTLS.

`federation = Federation(central_node_fqdn: str, disable_tls: bool, cert_chain: str, agg_certificate: str, agg_private_key: str)`

2. Federation's `register_collaborators` method should be used to provide an information about collaborators participating in a federation.
It requires a dictionary object - {collaborator name : local data path}.

### Experiment API
*Experiment* entity allows registering training related objects, FL tasks and settings.
To set up an FL experiment someone should use the Experiment interactive API. 

`from openfl.interface.interactive_api.experiment import FLExperiment`

*Experiment* is being initialized by taking federation as a parameter.
`fl_experiment = FLExperiment(federation=federation)`

To start an experiment user must register *DataLoader*, *Federated Learning tasks* and *Model* with *Optimizer*. There are several supplementary interface classes for these purposes.

`from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface`

#### Registering model and optimizer
First, user instantiate and initilize a model and optimizer in their favorite Deep Learning framework. Please, note that for now interactive API supports only *Keras* and *PyTorch* off-the-shelf.
Initialized model and optimizer objects then should be passed to the `ModelInterface` along with the path to correct Framework Adapter plugin inside OpenFL package. If desired DL framework is not covered by existing plugins, someone can implement the plugin's interface and point `framework_plugin` to the implementation inside the workspace.
`from openfl.interface.interactive_api.experiment import ModelInterface`
`MI = ModelInterface(model=model_unet, optimizer=optimizer_adam, framework_plugin=framework_adapter)`

#### Registering FL tasks
We have an agreement on what we consider to be a FL task.
Interactive API currently allows registering only standalone functions defined in the main module or imported from other modules inside the workspace.
We also have requirements on task signature. Task should accept the following objects:
1. model - will be rebuilt with relevant weights for every task by `TaskRunner`
2. data_loader - data loader that will provide local data
3. device - a device to be used for execution on collaborator machines
4. optimizer (optional) - model optimizer, only for training tasks

Moreover FL tasks should return a dictionary object with metrics `{metric name: metric value for this task}`.

`Task Interface` class is designed to register task and accompanying information.
This class must be instantiated, then it's special methods may be used to register tasks.

```
TI = TaskInterface()

task_settings = {
    'batch_size': 32,
    'some_arg': 228,
}
@TI.add_kwargs(**task_settings)
@TI.register_fl_task(model='my_model', data_loader='train_loader',
        device='device', optimizer='my_Adam_opt')
def foo(my_model, train_loader, my_Adam_opt, device, batch_size, some_arg=356)
    ...
````

`@TI.register_fl_task()` needs tasks argument names for (model, data_loader, device, optimizer (optional)) that constitute tasks 'contract'.
It adds the callable and the task contract to the task registry.

`@TI.add_kwargs()` method should be used to set up those arguments that are not included in the contract.

#### Registering Federated DataLoader
`DataInterface` is provided to support a remote DataLoader initialization.

It is initialized with User Dataset class object and all the keyword arguments can be used by dataloaders during training or validation.
User must subclass `DataInterface` and implements several methods.
* `_delayed_init(self, data_path)` is the most important method. It will be called during collaborator initialization procedure with relevant `data_path` (one that corresponds to the collaborator name that user registered in federation).
User Dataset class should be instantiated with local `data_path` here. If dataset initalization procedure differs for some of the  collaborators, the initialization logic must be described here. Dataset sharding procedure for test runs should also be described in this method.
User is free to save objects in class fields for later use.
* `get_train_loader(self, **kwargs)` will be called before training tasks execution. This method must return anything user expects to recieve in the training task with `data_loader` contract argument.
`kwargs` dict holds the same information that was provided during `DataInterface` initialization.
* `get_valid_loader(self, **kwargs)` - see the point above only with validation data
* `get_train_data_size(self)` - return number of samples in local train dataset.
* `get_valid_data_size(self)` - return number of samples in local validation dataset. 

#### Preparing workspace distribution
Now we may use `Experiment` API to prepare a workspace archive for transferring to collaborator's node. In order to run a collaborator, we want to replicate the workspace and the Python environment.

Instances of interface classes `(TaskInterface, DataInterface, ModelInterface)` must be passed to `FLExperiment.prepare_workspace_distribution()` method along with other parameters. 

This method:
* Compiles all provided setings to a Plan object. This is the central place where all actors in federation look up their parameters.
* Saves plan.yaml to the `plan/` folder inside the workspace.
* Serializes interface objects on the disk.
* Prepares `requirements.txt` for remote Python environment setup.
* Compressess the workspace to an archive so it can be coppied to collaborator nodes.

#### Starting the aggregator
As all previous steps done, the experiment is ready to start
`FLExperiment.start_experiment()` method requires `model_interface` object with initialized weights.

It starts a local aggregator that will wait for collaborators to connect.

### Starting collaborators
The process of starting collaborators has not changed.
User must transfer the workspace archive to a remote node and type in console:

`fx workspace import --archive ws.zip`

Please, note that aggregator and all the collaborator nodes should have the same Python interpreter version as the machine used for defining the experiment.

then cd to the workspace and run
`fx collaborator start -d data.yaml -n one`

For more details, please refer to the TaskRunner API section.
