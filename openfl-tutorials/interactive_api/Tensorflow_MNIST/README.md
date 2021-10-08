# MNIST Classification Tutorial

<img src="http://i.ytimg.com/vi/0QI3xgXuB-Q/hqdefault.jpg" alt="alt text" width="whatever" height="whatever">


### 1. About dataset
It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9. For more information, please visit [this](https://en.wikipedia.org/wiki/MNIST_database) web site.


### 2. About model
We use simple fully-connected neural network defined at 
[layers.py](./workspace/layers.py) file.


### 3. How to run this tutorial (without TLC and locally as a simulation):

1. Run director:
```sh
cd director
fx director start --disable-tls -c director_config.yaml
```

2. Run envoy:
```sh
cd envoy
fx envoy start -n env_one --disable-tls --shard-config-path shard_config_one.yaml -dh localhost -dp 50051
```

Optional: start second envoy:
 - Copy `envoy` folder to another place and run from there:
```sh
fx envoy start -n env_two --disable-tls --shard-config-path shard_config_two.yaml -dh localhost -dp 50051
```

3. Run `Mnist_Classification_FL.ipynb` jupyter notebook:
```sh
cd workspace
jupyter notebook Mnist_Classification_FL.ipynb
```
