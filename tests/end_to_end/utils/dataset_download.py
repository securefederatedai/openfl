import torchvision

def download_data(path="files/"):
    train_dataset = torchvision.datasets.MNIST(
        path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    test_dataset = torchvision.datasets.MNIST(
        path,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    return train_dataset, test_dataset
