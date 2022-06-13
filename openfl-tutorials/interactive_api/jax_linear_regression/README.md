# JAX based Linear Regression Tutorial

### 1. About dataset

Generate a random regression problem using `make_regression` from sklearn.datasets.
Define the below params in envoy.yaml config to control the dataset generation.
- rank (seed)
- n_samples (number of dataset instance)
- n_features (number of features)
- noise (Add noise to sampled dataset)


### 2. About model

Simple Regression Model with XLA compiled and Auto-grad based parameter updates.


### 3. How to run this tutorial (without TLC and locally as a simulation):

1. Run director:

```sh
cd director_folder
./start_director.sh
```

2. Run envoy:

```sh
cd envoy_folder
./start_envoy.sh env_instance_1 envoy_config_1.yaml
```

Optional: start second envoy:

- Copy `envoy_folder` to another place and run from there:

```sh
./start_envoy.sh env_instance_2 envoy_config_2.yaml
```

3. Run `jax_linear_regression.ipybnb` jupyter notebook:

```sh
cd workspace
jupyter lab Mnist_Classification_FL.ipybnb
```

4. Visualization

```
tensorboard --logdir logs/
```


### 4. Known issues

1. ##### CUDA_ERROR_OUT_OF_MEMORY Exception - JAX XLA pre-allocates 90% of the GPU at start

- Below flag to restrict max GPU allocation to 50%
```
%env XLA_PYTHON_CLIENT_MEM_FRACTION=.5
```
OR

- set XLA_PYTHON_CLIENT_PREALLOCATE to start with a small footprint.
```
%env XLA_PYTHON_CLIENT_PREALLOCATE=false
```