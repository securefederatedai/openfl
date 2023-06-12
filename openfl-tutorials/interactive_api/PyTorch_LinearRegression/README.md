# PyTorch based Linear Regression Tutorial

### 1. About dataset

Generate a random regression problem using `make_regression` from sklearn.datasets with pre-defined parameters.

Define the below param in envoy.yaml config to shard the dataset across participants/envoy.
- rank_worldsize


### 2. About model

Simple Regression Model based on PyTorch.


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
./start_envoy.sh env_instance_1 envoy_config.yaml
```

Optional: start second envoy:

- Copy `envoy_folder` to another place and follow the same process as above:

```sh
./start_envoy.sh env_instance_2 envoy_config.yaml
```

3. Run `torch_linear_regression.ipynb` jupyter notebook:

```sh
cd workspace
jupyter lab torch_linear_regression.ipynb
```

4. Visualization

```
tensorboard --logdir logs/
```
