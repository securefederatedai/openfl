# Anomaly detection and segmentation

![MVTec AD objects](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/dataset_overview_large.png "MVTec AD objects")


### 1. About dataset
MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. Each class contains
60 to 390 normal train images (defect free) and 40 to 167 test images (with various kinds of defects as well as images without defects). More info at [MVTec dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad).
For each object, the data is divided into 3 folders - 'train' (containing defect free training images), 'test'(containing test images, both good and bad), 'ground_truth' (containing the masks of defected images).

### 2. About model
Two neural networks are used: an encoder and a classifier. The encoder is composed of convolutional layers only. The classifier is a two layered MLP model having 128 hidden units per layer, and the input to the classifier is a subtraction of the features of the two patches. The activation function for both networks is a LeakyReLU with a α = 0.1.
The encoder has a hierarchical structure. The receptive field of the encoder is K = 64, and that of the embedded smaller encoder is K = 32. Patch SVDD divides the images into patches with a size K and a stride S. The values for the strides are S = 16 and S = 4 for the encoders with K = 64 and K = 32, respectively.

### 3. Links
[Original paper](https://arxiv.org/abs/2006.16067)
[Original Github code](https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch/tree/934d6238e5e0ad511e2a0e7fc4f4899010e7d892)
[MVTec ad dataset download link](https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz)


### 4. How to run this tutorial (without TLS and locally as a simulation):

Go to example folder:
cd <openfl_folder>/openfl-tutorials/interactive_api/PyTorch_MVTec_PatchSVDD

1. Run director:
```sh
cd director_folder
./start_director.sh
```

2. Run envoy:
```sh
cd envoy_folder
./start_envoy.sh env_one shard_config.yaml
```

Optional: start second envoy:
 - Copy `envoy_folder` to another place and run from there:
```sh
./start_envoy.sh env_two shard_config_two.yaml
```

3. Run `PatchSVDD_with_Director.ipynb` jupyter notebook:
```sh
cd workspace
jupyter notebook PatchSVDD_with_Director.ipynb
```