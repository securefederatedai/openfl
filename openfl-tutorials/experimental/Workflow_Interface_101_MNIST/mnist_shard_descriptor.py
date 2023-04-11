import yaml
from yaml.loader import SafeLoader
from copy import deepcopy

from torch.utils.data import DataLoader
import torchvision


class MnistShardDescriptor:
    def __init__(self, config_filename):

        self.download()

        idx, n_collaborators, batch_size_train, batch_size_test = self.read_config_file(
            config_filename)
        self.split_dataset(idx, n_collaborators, batch_size_train, batch_size_test)

    def read_config_file(self, config_filename):
        with open(config_filename, "r") as f:
            config = yaml.load(f, Loader=SafeLoader)

        rank_worldsize = config["shard_descriptor"]["params"]["rank_worldsize"]
        batch_size_train = config["shard_descriptor"]["params"]["batch_size_train"]
        batch_size_test = config["shard_descriptor"]["params"]["batch_size_test"]
        idx, total_collaborators = [int(i.strip()) for i in rank_worldsize.split(",")]

        return idx - 1, total_collaborators, batch_size_train, batch_size_test

    def split_dataset(self, idx, n, batch_size_train, batch_size_test):
        train = deepcopy(self.__mnist_train)
        test = deepcopy(self.__mnist_test)
        train.data = self.__mnist_train.data[idx::n]
        train.targets = self.__mnist_train.targets[idx::n]
        test.data = self.__mnist_test.data[idx::n]
        test.targets = self.__mnist_test.targets[idx::n]

        self.__dataset = {
            "train_loader": DataLoader(train, batch_size=batch_size_train, shuffle=True),
            "test_loader": DataLoader(test, batch_size=batch_size_test, shuffle=True),
        }

    def get(self):
        return self.__dataset

    def download(self):
        self.__mnist_train = torchvision.datasets.MNIST('files/', train=True, download=True,
                                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(
                                                                (0.1307,), (0.3081,))
                                                        ]))

        self.__mnist_test = torchvision.datasets.MNIST('files/', train=False, download=True,
                                                       transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize(
                                                               (0.1307,), (0.3081,))
                                                       ]))
