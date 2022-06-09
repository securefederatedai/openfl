# CIFAR10 Federated Classification Tutorial

### 1. About the Dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

### 2. About the Model

A simple multi-layer CNN is used. Definition provided in the notebook.

### 3. About the Federation

Data is equally partitioned between envoys/participants.

You can write your own splitting schema in the shard descriptor class.

### 3. How to run this tutorial (without TLC and locally as a simulation):

1. Run director:

```sh
$> cd director_folder
$> ./start_director.sh
```

2. Run envoy:

```sh
$> cd envoy_folder
$> ./start_envoy.sh env_one envoy_config_one.yaml
```

3. [Optional] Start second envoy:

- Copy `envoy_folder` to another place and run from there (alternatively, use the other config for second envoy):

```sh
$> ./start_envoy.sh env_two envoy_config_two.yaml
```

4. Run `Tensorflow_CIFAR.ipynb` jupyter notebook:

```sh
$> cd workspace
$> jupyter lab Tensorflow_CIFAR.ipynb
```
