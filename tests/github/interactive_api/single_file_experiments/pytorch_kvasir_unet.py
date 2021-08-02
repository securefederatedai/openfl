import logging

import torch
import torch.nn as nn
import torch.optim as optim

from tests.github.interactive_api.layers import soft_dice_loss, soft_dice_coef, double_conv, down, \
    up

logger = logging.getLogger(__name__)

"""
UNet model definition
"""


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = double_conv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x


model_unet = UNet()

optimizer_adam = optim.Adam(model_unet.parameters(), lr=1e-4)

import os
import PIL
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tsf
from skimage import io
from openfl.utilities import sha384sum

os.makedirs('data', exist_ok=True)
os.system(
    "wget -nc 'https://datasets.simula.no/hyper-kvasir/hyper-kvasir-segmented-images.zip' -O ./data/kvasir.zip")
ZIP_SHA384 = 'e30d18a772c6520476e55b610a4db457237f151e' \
             '19182849d54b49ae24699881c1e18e0961f77642be900450ef8b22e7'
if sha384sum('./data/kvasir.zip') != ZIP_SHA384:
    raise SystemError('ZIP File hash doesn\'t match expected file hash.')
os.system('unzip -n ./data/kvasir.zip -d ./data')

DATA_PATH = './data/segmented-images/'
import numpy as np


def read_data(image_path, mask_path):
    """
    Read image and mask from disk.
    """
    img = io.imread(image_path)
    assert (img.shape[2] == 3)
    mask = io.imread(mask_path)
    return (img, mask[:, :, 0].astype(np.uint8))


class KvasirDataset(Dataset):
    """
    Kvasir dataset contains 1000 images for all collaborators.
    Args:
        data_path: path to dataset on disk
        collaborator_count: total number of collaborators
        collaborator_num: number of current collaborator
        is_validation: validation option
    """

    def __init__(self, images_path='./data/segmented-images/images/',
                 masks_path='./data/segmented-images/masks/',
                 validation_fraction=1 / 8, is_validation=False):

        self.images_path = images_path
        self.masks_path = masks_path
        self.images_names = [
            img_name
            for img_name in sorted(os.listdir(self.images_path))
            if len(img_name) > 3 and img_name[-3:] == 'jpg'
        ]

        assert (len(self.images_names) > 2), "Too few images"

        validation_size = max(1, int(len(self.images_names) * validation_fraction))

        if is_validation:
            self.images_names = self.images_names[-validation_size:]
        else:
            self.images_names = self.images_names[: -validation_size]

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

    def __getitem__(self, index):
        name = self.images_names[index]
        img, mask = read_data(self.images_path + name, self.masks_path + name)
        img = self.img_trans(img).numpy()
        mask = self.mask_trans(mask).numpy()
        return img, mask

    def __len__(self):
        return len(self.images_names)


def function_defined_in_notebook():
    print('I will cause problems')


def train(unet_model, train_loader, optimizer, device, loss_fn=soft_dice_loss):
    function_defined_in_notebook()

    unet_model.train()
    unet_model.to(device)

    losses = []

    for data, target in train_loader:
        data, target = torch.tensor(data).to(device), torch.tensor(
            target).to(device, dtype=torch.float32)
        optimizer.zero_grad()
        output = unet_model(data)
        loss = loss_fn(output=output, target=target)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())

    return {'train_loss': np.mean(losses), }


def validate(unet_model, val_loader, device):
    unet_model.eval()
    unet_model.to(device)

    val_score = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            samples = target.shape[0]
            total_samples += samples
            data, target = torch.tensor(data).to(device), \
                           torch.tensor(target).to(device, dtype=torch.int64)
            output = unet_model(data)
            val = soft_dice_coef(output, target)
            val_score += val.sum().cpu().numpy()

    return {'dice_coef': val_score / total_samples, }


from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, \
    ModelInterface, FLExperiment

from copy import deepcopy

framework_adapter = 'openfl.plugins.frameworks_adapters.pytorch_adapter.FrameworkAdapterPlugin'
model_interface = ModelInterface(model=model_unet, optimizer=optimizer_adam,
                                 framework_plugin=framework_adapter)

# Save the initial model state
initial_model = deepcopy(model_unet)


class UserDataset:
    def __init__(self, path_to_local_data):
        print(f'User Dataset initialized with {path_to_local_data}')


class OpenflMixin:
    def _delayed_init(self):
        raise NotImplementedError


class FedDataset(OpenflMixin):
    def __init__(self, UserDataset):
        self.user_dataset_class = UserDataset
        print('We implement all abstract methods from mixin in this class')

    def _delayed_init(self, data_path):
        print('This method is called on the collaborator node')
        dataset_obj = self.user_dataset_class(data_path)


fed_dataset = FedDataset(UserDataset)
fed_dataset._delayed_init('data path on the collaborator node')


class FedDataset(DataInterface):
    def __init__(self, UserDatasetClass, **kwargs):
        self.UserDatasetClass = UserDatasetClass
        self.kwargs = kwargs

    def _delayed_init(self, data_path='1,1'):
        # With the next command the local dataset will be loaded on the collaborator node
        # For this example we have the same dataset on the same path, and we will shard it
        # So we use `data_path` information for this purpose.
        self.rank, self.world_size = [int(part) for part in data_path.split(',')]

        validation_fraction = 1 / 8
        self.train_set = self.UserDatasetClass(validation_fraction=validation_fraction,
                                               is_validation=False)
        self.valid_set = self.UserDatasetClass(validation_fraction=validation_fraction,
                                               is_validation=True)

        # Do the actual sharding
        self._do_sharding(self.rank, self.world_size)

    def _do_sharding(self, rank, world_size):
        # This method relies on the dataset's implementation
        # i.e. coupled in a bad way
        self.train_set.images_names = self.train_set.images_names[rank - 1:: world_size]

    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        return DataLoader(
            self.train_set, num_workers=8, batch_size=self.kwargs['train_bs'], shuffle=True
        )

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        return DataLoader(self.valid_set, num_workers=8, batch_size=self.kwargs['valid_bs'])

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        return len(self.train_set)

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.valid_set)


fed_dataset = FedDataset(KvasirDataset, train_bs=8, valid_bs=8)

TI = TaskInterface()

import tqdm


def function_defined_in_notebook(some_parameter):
    print('I will cause problems')
    print(f'Also I accept a parameter and it is {some_parameter}')


# We do not actually need to register additional kwargs, Just serialize them
@TI.add_kwargs(**{'some_parameter': 42})
@TI.register_fl_task(model='unet_model', data_loader='train_loader',
                     device='device', optimizer='optimizer')
def train(unet_model, train_loader, optimizer, device, loss_fn=soft_dice_loss, some_parameter=None):
    if not torch.cuda.is_available():
        device = 'cpu'

    function_defined_in_notebook(some_parameter)

    train_loader = tqdm.tqdm(train_loader, desc="train")

    unet_model.train()
    unet_model.to(device)

    losses = []

    for data, target in train_loader:
        data, target = torch.tensor(data).to(device), torch.tensor(
            target).to(device, dtype=torch.float32)
        optimizer.zero_grad()
        output = unet_model(data)
        loss = loss_fn(output=output, target=target)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())

    return {'train_loss': np.mean(losses), }


@TI.register_fl_task(model='unet_model', data_loader='val_loader', device='device')
def validate(unet_model, val_loader, device):
    unet_model.eval()
    unet_model.to(device)

    val_loader = tqdm.tqdm(val_loader, desc="validate")

    val_score = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            samples = target.shape[0]
            total_samples += samples
            data, target = torch.tensor(data).to(device), \
                           torch.tensor(target).to(device, dtype=torch.int64)
            output = unet_model(data)
            val = soft_dice_coef(output, target)
            val_score += val.sum().cpu().numpy()

    return {'dice_coef': val_score / total_samples, }

    # Create a federation


from openfl.interface.interactive_api.federation import Federation

# 1) Run with TLS disabled (trusted environment)
# will determine fqdn by itself
from socket import getfqdn

federation = Federation(central_node_fqdn=getfqdn(), disable_tls=True)
# First number which is a collaborators rank is also passed as a cuda device identifier
col_data_paths = {'one': '1,2',
                  'two': '2,2'}
federation.register_collaborators(col_data_paths=col_data_paths)

# --------------------------------------------------------------------------------------------------------------------
# 2) Run with aggregator-collaborator mTLS
# If the user wants to enable mTLS their must provide CA root chain, and signed key pair to the federation interface
# cert_chain = 'cert/cert_chain.crt'
# agg_certificate = 'cert/agg_certificate.crt'
# agg_private_key = 'cert/agg_private.key'

# federation = Federation(central_node_fqdn=getfqdn(), disable_tls=True,
#                        cert_chain=cert_chain, agg_certificate=agg_certificate, agg_private_key=agg_private_key)
# col_data_paths = {'one': '1,1',}
# federation.register_collaborators(col_data_paths=col_data_paths)

# create an experimnet in federation
fl_experiment = FLExperiment(federation=federation, )

# If I use autoreload I got a pickling error

# # The following command zips the workspace and python requirements to be transfered to collaborator nodes
# fl_experiment.prepare_workspace_distribution(model_provider=MI, task_keeper=TI, data_loader=fed_dataset, rounds_to_train=7, \
#                               opt_treatment='CONTINUE_GLOBAL')
# # # This command starts the aggregator server
# # fl_experiment.start_experiment(model_provider=MI)


# If I use autoreload I got a pickling error
arch_path = fl_experiment.prepare_workspace_distribution(
    model_provider=model_interface,
    task_keeper=TI,
    data_loader=fed_dataset,
    rounds_to_train=7,
    opt_treatment='CONTINUE_GLOBAL'
)

from tests.github.interactive_api.experiment_runner import run_experiment

run_experiment(col_data_paths, model_interface, arch_path, fl_experiment)

best_model = fl_experiment.get_best_model()
fed_dataset._delayed_init()

logger.info('Validating initial model')
validate(initial_model, fed_dataset.get_valid_loader(), 'cpu')

logger.info('Validating trained model')
validate(best_model, fed_dataset.get_valid_loader(), 'cpu')
