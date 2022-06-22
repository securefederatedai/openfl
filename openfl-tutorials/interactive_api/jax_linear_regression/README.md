# JAX based Linear Regression Tutorial

### 1. About dataset

Generate a random regression problem using `make_regression` from sklearn.datasets with pre-defined parameters.

Define the below param in envoy.yaml config to shard the dataset across participants/envoy.
- rank_worldsize


### 2. About model

Simple Regression Model with XLA compiled and Auto-grad based parameter updates.


### 3. How to run this tutorial (without TLC and locally as a simulation):

1. Run director:

```sh
cd director_folder
./start_director.sh
```

2. Run envoy:

Step 1: Activate virtual environment and install packages
```
cd envoy_folder
pip install -r requirements.txt
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

3. Run `jax_linear_regression.ipynb` jupyter notebook:

```sh
cd workspace
jupyter lab jax_linear_regression.ipynb
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