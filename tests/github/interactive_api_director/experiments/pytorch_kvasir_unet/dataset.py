import os
import PIL
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tsf

from tests.github.interactive_api_director.experiments.pytorch_kvasir_unet.data_loader import read_data  # noqa: E501
from openfl.interface.interactive_api.experiment import DataInterface


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


class FedDataset(DataInterface):

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
