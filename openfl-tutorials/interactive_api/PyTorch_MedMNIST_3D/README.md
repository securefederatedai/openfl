# MedMNIST 3D Classification Tutorial

![MedMNISTv2_overview](https://raw.githubusercontent.com/MedMNIST/MedMNIST/main/assets/medmnistv2.jpg)

For more details, please refer to the original paper:
**MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification** ([arXiv](https://arxiv.org/abs/2110.14795))


### 1. MedMNIST Installation and Requirements
Setup the required environments and install `medmnist` as a standard Python package from [PyPI](https://pypi.org/project/medmnist/):

    pip install medmnist

Or install from source:

    pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git

Check whether you have installed the latest [version](medmnist/info.py):

    >>> import medmnist
    >>> print(medmnist.__version__)

The code requires only common Python environments for machine learning. Basically, it was tested with
* Python 3 (>=3.8)
* PyTorch\==1.11.0
* Torchvision\==0.12.0
* numpy\==1.23.1, pandas\==1.4.3, scikit-learn\==1.1.1, Pillow\==9.2.0, fire\==0.4.0, scikit-image==0.19.3

Higher (or lower) versions should also work (perhaps with minor modifications). 

### 2. About model and experiments

We use a simple convolutional neural network and settings coming from [the experiments](https://github.com/MedMNIST/experiments) repository.

### 3. How to run this tutorial (without TLC and locally as a simulation):
0. Activate openfl environment (if needed)

1. Run director:

```sh
cd director
./start_director.sh
```

2. Run envoy:

```sh
cd envoy
./start_envoy.sh env_one envoy_config.yaml
```

Optional: start second envoy:

- Copy `envoy_folder` to another place and run from there:

```sh
./start_envoy.sh env_two envoy_config.yaml
```

3. Run `Pytorch_MedMNIST_3D.ipynb` jupyter notebook:

```sh
cd workspace
jupyter lab Pytorch_MedMNIST_3D.ipynb
```

