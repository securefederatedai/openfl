

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

## Defining a Federate Learning experiment
Interactive API allows defining an experiment from a single entrypoint - a jupyter notebook or a python script.
