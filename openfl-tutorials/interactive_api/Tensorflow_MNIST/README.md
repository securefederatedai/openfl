# MNIST Classification Tutorial

![mnist digits](http://i.ytimg.com/vi/0QI3xgXuB-Q/hqdefault.jpg "MNIST Digits")


### 1. About dataset
It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9. More info at [wiki](https://en.wikipedia.org/wiki/MNIST_database).

### 2. About model
We use simple fully-connected neural network defined at 
[layers.py](./workspace/layers.py) file.


### 3. How to run this tutorial (without TLC):

* 1. Run director:
```sh
cd director_folder
./start_director.yaml
```

* 2. Run envoy:
```sh
cd envoy_one_folder
./start_envoy_one.yaml
```

Optional:
start second envoy:
```sh
cd envoy_two_folder
./start_envoy_two.yaml
```

* 3. Run `Mnist_Classification_FL.ipybnb` jupyter notebook:
```sh
cd workspace
jupyter notebook Mnist_Classification_FL.ipybnb
```
