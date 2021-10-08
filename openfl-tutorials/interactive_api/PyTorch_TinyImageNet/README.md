# TinyImagenet Classification Tutorial

<img src="https://production-media.paperswithcode.com/datasets/Tiny_ImageNet-0000001404-a53923c3_XCrVSGm.jpg" alt="alt text" width="300" height="whatever">


### 1. About dataset
This is a miniature of ImageNet classification Challenge. Dataset contains 200 classes for training. Each class has 500 images. The test set contains 10,000 images. All images are 64x64 colored ones. For more information, please visit [this](https://www.kaggle.com/c/tiny-imagenet) web site.


### 2. How to run this tutorial (without TLC and locally as a simulation):

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

3. Run `pytorch_tinyimagenet.ipynb` jupyter notebook:
```sh
cd workspace
jupyter notebook pytorch_tinyimagenet.ipynb
```
