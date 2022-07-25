# MedMNIST 2D Classification Tutorial

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
* Python 3 (>=3.6)
* PyTorch\==1.3.1
* numpy\==1.18.5, pandas\==0.25.3, scikit-learn\==0.22.2, Pillow\==8.0.1, fire, scikit-image

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

3. Run `Pytorch_MedMNIST_2D.ipynb` jupyter notebook:

```sh
cd workspace
jupyter lab Pytorch_MedMNIST_2D.ipynb
```
