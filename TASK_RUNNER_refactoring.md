

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


## The workspace
To initialize the workspace create an empty folder and a jupyter notebook (or a script) inside it. Root folder of the notebook will be considered the workspace.
If some objects are imported in the notebook from local modules, source code should be keeped inside the workspace.
If one decides to keep local test data inside the workspace, `data` folder should be used as it will not be exported.
If one decides to keep certificates inside the workspace, `cert` folder should be used as it will not be exported.
Only relevant source code or resources should be kept inside the worksapce as it will be zipped and distributed to collaborator machines.

## The python environment
Create a virtual python environment. Please install only packages required for conducting the experiment as python environment will be replicated on collaborator nodes.

## Defining a Federated Learning experiment
Interactive API allows defining an experiment from a single entrypoint - a jupyter notebook or a python script.
Defining an experiment includes setting up several interface entities and experiment parameters.

### Federation API
Federation entity exists to keep information about collaborator related settings, their local data and network setting to enable communication in federation. 
Each federation is bound to some machine learning problem in a sense that all collaborators dataset shards should have the same format and labeling. Created federation may be used in several subsequent experiments.
To set up a federation use Federation interactive API.
`from openfl.interface.interactive_api.federation import Federation`
1. Federation API class should be initialized with the aggregator node FQDN and encryption settings. One may disable mTLS in trusted environments or provide paths to a certificate chain to CA, aggregator certificate and pricate key to enable mTLS.
`federation = Federation(central_node_fqdn: str, disable_tls: bool, cert_chain: str, agg_certificate: str, agg_private_key: str)`
2. Federation's `register_collaborators` method should be used to provide information about collaborators participating federation.
It requres dictionary {collaborator name : local data path}.

### Experiment API
Experiment entity alows registering training related objects, tasks, and settings.
To set up an FL experiment one should use Experiment interactive API. 
`from openfl.interface.interactive_api.experiment import FLExperiment`
Experiment is being initialized taking federation as a parameter.
`fl_experiment = FLExperiment(federation=federation,)`
To start an experiment user must register Dataloader, Federated Learning tasks, and Model with optimizer. For this purpose there are several supplementary interface classes.
`from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface`

#### Registering model and optimizer
First, user instantiate and initilize a model and optimizer in their favorite Deep Learning framework. Please choose framework wisely as interactive API supports only Keras and Pytorch off-the-shelf, as for now.
Initialized model and optimizer objects then should be provided to initializr the `ModelInterface` along with the path to correct Framework Adapter plugin inside OpenFL package. If desired framework is not supported with existing plugins, one is free to implement the plugin's interface and point to the implementation inside the workspace.
`from openfl.interface.interactive_api.experiment import ModelInterface`
`MI = ModelInterface(model=model_unet, optimizer=optimizer_adam, framework_plugin=framework_adapter)`

#### Registering FL tasks


