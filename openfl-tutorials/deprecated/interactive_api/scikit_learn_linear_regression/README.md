# Scikit-learn based Linear Regression Tutorial

### 1. About dataset

Generate 1-dimensional noisy data for linear regression of sinusoid. 

Define the below pamameter in shard_config in the envoy_config.yaml file as the random seed for the dataset generation for a specific Envoy 
- rank

### 2. About model

Linear Regression Lasso Model based on Scikit-learn.


### 3. How to run this tutorial (without TLC and locally as a simulation):

1. Run director:

```sh
cd director folder
./start_director.sh
```

2. Run envoy:

Step 1: Activate virtual environment and install packages
```
cd envoy folder
pip install -r requirements.txt
```
Step 2: start the envoy
```sh
./start_envoy.sh env_instance_1 envoy_config.yaml
```

Optional: start second envoy:

- Copy `envoy_folder` to another place and follow the same process as above:

```sh
./start_envoy.sh env_instance_2 envoy_config_2.yaml
```

3. Run `scikit_learn_linear_regression.ipynb` jupyter notebook:

```sh
cd workspace
jupyter lab scikit_learn_linear_regression.ipynb
```

4. Visualization

```
tensorboard --logdir logs/
```
