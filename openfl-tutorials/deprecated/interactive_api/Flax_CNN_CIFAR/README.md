# Federated FLAX CIFAR-10 CNN Tutorial

### 1. About dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Define the below param in envoy.yaml config to shard the dataset across participants/envoy.
- rank_worldsize

### 2. About model

A simple multi-layer CNN is used with XLA compiled and Auto-grad based parameter updates.
Definition provided in the notebook.

### 3. Notebook Overview

1. Class `CustomTrainState` - Subclasses `flax.training.TrainState`
    - Variable `opt_vars` to keep track of generic optimizer variables.
    - Method `update_state` to update the OpenFL `ModelInterface` registered state with the new_state returned within the `TaskInterface` registered training loop.

2. Method `create_train_state`: Creates a new `TrainState` by encapsulating model layer definitions, random model parameters, and optax optimizer state.

3. Method `apply_model` (`@jax.jit` decorated function): It takes a TrainState, images, and labels as parameters. It computes and returns the gradients, loss, and accuracy. These gradients are applied to a given state in the `update_model` method (`@jax.jit` decorated function) and a new TrainState instance is returned.

### 4. How to run this tutorial (without TLS and locally as a simulation):

0. Pre-requisites:
    
    - Nvidia Driver >= 495.29.05
    - CUDA >= 11.1.105
    - cuDNN >= 8
    
    Activate virtual environment (Python - 3.8.10) and install packages from requirements.txt

    Set the variable `DEFAULT_DEVICE to 'CPU' or 'GPU'` in `start_envoy.sh` and notebook to enforce/control the execution platform.

```sh
cd Flax_CNN_CIFAR
pip install -r requirements.txt
```

1. Run director:

```sh
cd director
./start_director.sh
```

2. Run envoy:

```sh
cd envoy
./start_envoy.sh "envoy_identifier" envoy_config.yaml
```

Optional: start second envoy:

- Copy `envoy` folder to another place and follow the same process as above:

```sh
./start_envoy.sh "envoy_identifier_2" envoy_config_2.yaml
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


### 5. Known issues

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

3. #### DNN library Not found error

- Make sure the jaxlib(cuda version), Nvidia Driver, CUDA and cuDNN versions are specific, relevant and compatible as per the documentation.
- Reference:
    -   CUDA and cuDNN Compatibility Matrix: https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html
    -   Official JAX Compatible CUDA Releases: https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
