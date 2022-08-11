# Federated FLAX CIFAR-10 CNN Tutorial

### 1. About dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Define the below param in envoy.yaml config to shard the dataset across participants/envoy.
- rank_worldsize

### 2. About model

A simple multi-layer CNN is used with XLA compiled and Auto-grad based parameter updates.
Definition provided in the notebook.

### 3. How to run this tutorial (without TLS and locally as a simulation):

0. Pre-requisites:

    Activate virtual environment (Python - 3.8.10) and install below packages.

```
pip install openfl==1.3 protobuf==3.19.4 --no-cache-dir
```

1. Run director:

```sh
cd director_folder
./start_director.sh
```

2. Run envoy:

Step 1: Activate virtual environment and install packages from the requirements.txt
```
cd envoy_folder
pip install -r requirements.txt --no-cache-dir
```
Step 2: start the envoy
```sh
./start_envoy.sh env_instance_1 envoy_config_1.yaml
```

Optional: start second envoy:

- Copy `envoy_folder` to another place and follow the same process as above:

```sh
./start_envoy.sh env_instance_2 envoy_config_2.yaml
```

3. Run `FLAX_CIFAR10_CNN.ipynb` jupyter notebook:

```sh
cd workspace
jupyter lab FLAX_CIFAR10_CNN.ipynb
```

4. Visualization:

```
tensorboard --logdir logs/
```


### 4. Known issues

1. #### CUDA_ERROR_OUT_OF_MEMORY Exception - JAX XLA pre-allocates 90% of the GPU at start

- set XLA_PYTHON_CLIENT_PREALLOCATE to start with a small memory footprint.
```
%env XLA_PYTHON_CLIENT_PREALLOCATE=false
```
OR

- Below flag to restrict max GPU allocation to 50%
```
%env XLA_PYTHON_CLIENT_MEM_FRACTION=.5
```


2. #### Tensorflow pre-allocates 90% of the GPU (Potential OOM Errors).

- set TF_FORCE_GPU_ALLOW_GROWTH to start with a small memory footprint.
```
%env TF_FORCE_GPU_ALLOW_GROWTH=true
```