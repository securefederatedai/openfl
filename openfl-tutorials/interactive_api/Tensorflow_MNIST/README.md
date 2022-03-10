# MNIST Classification Tutorial

![mnist digits](http://i.ytimg.com/vi/0QI3xgXuB-Q/hqdefault.jpg "MNIST Digits")

### 1. About dataset

It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits
between 0 and 9. More info at [wiki](https://en.wikipedia.org/wiki/MNIST_database).

### 2. About model

We use simple fully-connected neural network defined at
[layers.py](./workspace/layers.py) file.

### 3. How to run this tutorial (without TLC and locally as a simulation):

1. Run director:

```sh
cd director_folder
./start_director.sh
```

2. Run envoy:

```sh
cd envoy_folder
./start_envoy.sh env_one envoy_config_one.yaml
```

Optional: start second envoy:

- Copy `envoy_folder` to another place and run from there:

```sh
./start_envoy.sh env_two envoy_config_two.yaml
```

3. Run `Mnist_Classification_FL.ipybnb` jupyter notebook:

```sh
cd workspace
jupyter lab Mnist_Classification_FL.ipybnb
```
