# Person re-identification Tutorial (Market-1501)

<img src="https://production-media.paperswithcode.com/datasets/Market-1501-0000000097-a728ab2d_gyNBlrI.jpg" alt="alt text" width="400" height="whatever">


### 1. About dataset
Market-1501 is a large-scale public benchmark dataset for person re-identification. It contains 1501 identities which are captured by six different cameras, and 32,668 pedestrian image bounding-boxes obtained using the Deformable Part Models pedestrian detector. Each person has 3.6 images on average at each viewpoint. For more information, please visit [this](https://paperswithcode.com/dataset/market-1501) web site.


### 2. How to run this tutorial (without TLC and locally as a simulation):

1. Run director:
```sh
cd director
fx director start --disable-tls -c director_config.yaml
```

2. Run envoy:
```sh
cd envoy
fx envoy start -n env_one --disable-tls -dh localhost -dp 50051  -sc shard_config_one.yaml
```

Optional: start second envoy:
 - Copy `envoy` folder to another place and run from there:
```sh
fx envoy start -n env_two --disable-tls -dh localhost -dp 50051  -sc shard_config_two.yaml
```

3. Run `PyTorch_Market_Re-ID.ipynb` jupyter notebook:
```sh
cd workspace
jupyter notebook PyTorch_Market_Re-ID.ipynb
```
