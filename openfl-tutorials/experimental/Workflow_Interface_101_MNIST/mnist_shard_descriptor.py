import yaml
from yaml.loader import SafeLoader
from copy import deepcopy

from torch.utils.data import DataLoader
import torchvision


class ShardDescriptor:
    def __init__(self):

        if not hasattr(self, "dataset_chunks"):
            self.download()

            n_collaborators, batch_size = self.read_config_file()
            self.split_dataset(n_collaborators, batch_size)


    def read_config_file(self):
        with open("config.yaml", "r") as f:
            config = yaml.load(f, Loader=SafeLoader)

        total_collaborators = config["shard_descriptor"]["params"]["total_collaborators"]
        batch_size = config["shard_descriptor"]["params"]["batch_size"]
        return total_collaborators, batch_size


    def split_dataset(self, n, batch_size):
        self.dataset_chunks = []

        for idx in range(n):
            train = deepcopy(self.mnist_train)
            test = deepcopy(self.mnist_test)
            train.data = self.mnist_train.data[idx::n]
            train.targets = self.mnist_train.targets[idx::n]
            test.data = self.mnist_test.data[idx::n]
            test.targets = self.mnist_test.targets[idx::n]

            self.dataset_chunks.append({
                    "train_loader": DataLoader(train, batch_size=batch_size, shuffle=True),
                    "test_loader": DataLoader(train, batch_size=batch_size, shuffle=True),
                }
            )


    def get(self):
        dataset_chunk = self.dataset_chunks[0]
        del self.dataset_chunks[0]
        return dataset_chunk


    def download(self):
        self.mnist_train = torchvision.datasets.MNIST('files/', train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                ]))

        self.mnist_test = torchvision.datasets.MNIST('files/', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                ]))


if __name__ == "__main__":
    ShardDescriptor()
