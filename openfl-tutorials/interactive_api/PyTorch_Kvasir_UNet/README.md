# HyperKvasir Tutorial


### 1. About dataset
The data is collected during real gastro- and colonoscopy examinations at a Hospital in Norway and partly labeled by experienced gastrointestinal endoscopists. For more information, please visit [this](https://datasets.simula.no/hyper-kvasir/) web site.


### 3. How to run this tutorial (without TLC and locally as a simulation):

1. Run director:
```sh
cd director_folder
fx director start --disable-tls -c director_config.yaml
```

2. Run envoy:
```sh
cd envoy_folder
fx envoy start -n env_one --disable-tls --shard-config-path shard_config_one.yaml -dh localhost -dp 50051
```

Optional: start second envoy:
 - Copy `envoy_folder` to another place and run from there:
```sh
fx envoy start -n env_two --disable-tls --shard-config-path shard_config_two.yaml -dh localhost -dp 50051
```

3. Run `PyTorch_Kvasir_UNet.ipynb` jupyter notebook:
```sh
cd workspace
jupyter notebook PyTorch_Kvasir_UNet.ipynb
```
