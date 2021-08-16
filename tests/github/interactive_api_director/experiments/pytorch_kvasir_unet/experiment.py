import logging
from socket import getfqdn

import torch.optim as optim


from tests.github.interactive_api_director.experiments.pytorch_kvasir_unet.data_loader import load_data
from tests.github.interactive_api_director.experiments.pytorch_kvasir_unet.model import UNet
from tests.github.interactive_api_director.experiments.pytorch_kvasir_unet.dataset import KvasirDataset, FedDataset
from openfl.interface.interactive_api.experiment import ModelInterface, FLExperiment
from openfl.interface.interactive_api.federation import Federation
from tests.github.interactive_api_director.experiments.pytorch_kvasir_unet.tasks import validate, task_interface
# from tests.github.interactive_api.experiment_runner import run_experiment

from copy import deepcopy


federation = Federation(client_id='frontend', director_node_fqdn='localhost', director_port='50051', tls=False)

shard_registry = federation.get_shard_registry()
shard_registry

dummy_shard_desc = federation.get_dummy_shard_descriptor(size=10)
sample, target = dummy_shard_desc[0]

from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface, FLExperiment

import os
import PIL
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms as tsf

# Now you can implement you data loaders using dummy_shard_desc
class KvasirSD(DataInterface, Dataset):

    def __init__(self, validation_fraction=1/8, **kwargs):
        super().__init__(**kwargs)
        
        self.validation_fraction = validation_fraction
        
        # Prepare transforms
        self.img_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize((332, 332)),
            tsf.ToTensor(),
            tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.mask_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize((332, 332), interpolation=PIL.Image.NEAREST),
            tsf.ToTensor()])
        
    @property
    def shard_descriptor(self):
        return self._shard_descriptor
        
    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        Local shard_descriptor  will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor
        
        validation_size = max(1, int(len(self.shard_descriptor) * self.validation_fraction))
        
        self.train_indeces = np.arange(len(self.shard_descriptor) - validation_size)
        self.val_indeces = np.arange(len(self.shard_descriptor) - validation_size, len(self.shard_descriptor))
        

    def __getitem__(self, index):
        img, mask = self.shard_descriptor[index]
        img = self.img_trans(img).numpy()
        mask = self.mask_trans(mask).numpy()
        return img, mask

    def __len__(self):
        return len(self.shard_descriptor)
    
    
    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        train_sampler = SubsetRandomSampler(self.train_indeces)
        return DataLoader(
            self, num_workers=8, batch_size=self.kwargs['train_bs'], sampler=train_sampler
            )

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        val_sampler = SubsetRandomSampler(self.val_indeces)
        return DataLoader(self, num_workers=8, batch_size=self.kwargs['valid_bs'], sampler=val_sampler)

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        return len(self.train_indeces)

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.val_indeces)

fed_dataset = KvasirSD(train_bs=4, valid_bs=8)
fed_dataset.shard_descriptor = dummy_shard_desc
for i, (sample, target) in enumerate(fed_dataset.get_train_loader()):
    print(sample.shape)

model_unet = UNet()
optimizer_adam = optim.Adam(model_unet.parameters(), lr=1e-4)

from copy import deepcopy

framework_adapter = 'openfl.plugins.frameworks_adapters.pytorch_adapter.FrameworkAdapterPlugin'
MI = ModelInterface(model=model_unet, optimizer=optimizer_adam, framework_plugin=framework_adapter)

# Save the initial model state
initial_model = deepcopy(model_unet)

from tests.github.interactive_api_director.experiments.pytorch_kvasir_unet.tasks import task_interface
# create an experimnet in federation
experiment_name = 'kvasir_test_experiment'
fl_experiment = FLExperiment(federation=federation, experiment_name=experiment_name)

fl_experiment.start(model_provider=MI, 
                    task_keeper=task_interface,
                    data_loader=fed_dataset,
                    rounds_to_train=2,
                    opt_treatment='CONTINUE_GLOBAL')
fl_experiment.stream_metrics()
best_model = fl_experiment.get_best_model()
fl_experiment.remove_experiment_data()
best_model.inc.conv[0].weight
validate(initial_model, fed_dataset.get_valid_loader(), 'cpu')
validate(best_model, fed_dataset.get_valid_loader(), 'cpu')