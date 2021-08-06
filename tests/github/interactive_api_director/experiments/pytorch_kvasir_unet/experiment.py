import logging
from socket import getfqdn

import torch.optim as optim


from tests.github.interactive_api.experiments.pytorch_kvasir_unet.model import UNet
from tests.github.interactive_api.experiments.pytorch_kvasir_unet.data_loader import load_data
from tests.github.interactive_api.experiments.pytorch_kvasir_unet.dataset import KvasirDataset, FedDataset
from openfl.interface.interactive_api.experiment import ModelInterface, FLExperiment
from openfl.interface.interactive_api.federation import Federation
from tests.github.interactive_api.experiments.pytorch_kvasir_unet.tasks import validate, task_interface
# from tests.github.interactive_api.experiment_runner import run_experiment

from copy import deepcopy


logger = logging.getLogger(__name__)

model_unet = UNet()
optimizer_adam = optim.Adam(model_unet.parameters(), lr=1e-4)

load_data()

framework_adapter = 'openfl.plugins.frameworks_adapters.pytorch_adapter.FrameworkAdapterPlugin'
model_interface = ModelInterface(model=model_unet, optimizer=optimizer_adam,
                                 framework_plugin=framework_adapter)

# Save the initial model state
initial_model = deepcopy(model_unet)

fed_dataset = FedDataset(KvasirDataset, train_bs=8, valid_bs=8)
federation = Federation(central_node_fqdn=getfqdn(), tls=False)

# First number which is a collaborators rank is also passed as a cuda device identifier
col_data_paths = {'one': '1,2',
                  'two': '2,2'}
federation.register_collaborators(col_data_paths=col_data_paths)
fl_experiment = FLExperiment(federation=federation)

# If I use autoreload I got a pickling error
arch_path = fl_experiment.prepare_workspace_distribution(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=fed_dataset,
    rounds_to_train=7,
    opt_treatment='CONTINUE_GLOBAL'
)

# run_experiment(col_data_paths, model_interface, arch_path, fl_experiment)
#
# best_model = fl_experiment.get_best_model()
# fed_dataset._delayed_init()
#
# logger.info('Validating initial model')
# validate(initial_model, fed_dataset.get_valid_loader(), 'cpu')
#
# logger.info('Validating trained model')
# validate(best_model, fed_dataset.get_valid_loader(), 'cpu')
