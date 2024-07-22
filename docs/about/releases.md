# Releases

## 1.5.1
[Full Release Notes](https://github.com/securefederatedai/openfl/releases/tag/v1.5.1)

We are excited to announce the release of OpenFL 1.5.1 - our first since moving to LF AI & Data! This release brings the following changes.

### 1.5.1 Highlights
- **Documentation accessibility improvements**: As part of our [Global Accessibility Awareness Day](https://www.intel.com/content/www/us/en/developer/articles/community/open-fl-project-improve-accessibility-for-devs.html) (GAAD) Pledge, the OpenFL project is making strides towards more accessible documentation. This release includes the integration of [Intel® One Mono](https://www.intel.com/content/www/us/en/company-overview/one-monospace-font.html) font, contrast color improvements, formatting improvements, and [new accessibility focused issues](https://github.com/securefederatedai/openfl/issues?q=is%3Aissue+is%3Aopen+accessibility) to take up in the future. 
- **[Documentation to federate a Generally Nuanced Deep Learning Framework (GaNDLF) model with OpenFL](https://openfl.readthedocs.io/en/latest/running_the_federation_with_gandlf.html)**
- **New OpenFL Interactive API Tutorials**:
    - [Linear regression with SciKit-Learn](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/interactive_api/scikit_learn_linear_regression)
    - [MedMNIST 2D Classification Using FedProx Optimizer](https://github.com/securefederatedai/openfl/blob/develop/openfl-tutorials/interactive_api/PyTorch_FedProx_MNIST/README.md?plain=1)
    - [PyTorch Linear Regression Example](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/interactive_api/PyTorch_LinearRegression)
- **Improvements to workspace export and import**
- **Many documentation improvements and updates**
- **Bug fixes**
- **Fixing dependency vulnerabilities**

## 1.5
[Full Release Notes](https://github.com/securefederatedai/openfl/releases/tag/v1.5)

### 1.5 Highlights
* **New Workflows Interface (Experimental)** - a new way of composing federated learning experiments inspired by [Metaflow](https://github.com/Netflix/metaflow). Enables the creation of custom aggregator and collaborators tasks. This initial release is intended for simulation on a single node (using the LocalRuntime); distributed execution (FederatedRuntime) to be enabled in a future release. 
* **New use cases enabled by the workflow interface**:
    * **[End-of-round validation with aggregator dataset](https://github.com/intel/openfl/blob/develop/openfl-tutorials/experimental/Workflow_Interface_102_Aggregator_Validation.ipynb)** 
    * **[Privacy Meter](https://github.com/intel/openfl/tree/develop/openfl-tutorials/experimental/Privacy_Meter)** - Privacy meter, based on state-of-the-art membership inference attacks, provides a tool to quantitatively audit data privacy in statistical and machine learning algorithms. The objective of a membership inference attack is to determine whether a given data record was in the training dataset of the target model. Measures of success (accuracy, area under the ROC curve, true positive rate at a given false positive rate ...) for particular membership inference attacks against a target model are used to estimate privacy loss for that model (how much information a target model leaks about its training data). Since stronger attacks may be possible, these measures serve as lower bounds of the actual privacy loss. The Privacy Meter workflow example generates privacy loss reports for all party's local model updates as well as the global models throughout all rounds of the FL training. 
    * **[Vertical Federated Learning Examples](https://github.com/intel/openfl/tree/develop/openfl-tutorials/experimental/Vertical_FL)** 
    * **[Federated Model Watermarking](https://github.com/intel/openfl/blob/develop/openfl-tutorials/experimental/Workflow_Interface_301_MNIST_Watermarking.ipynb)** using the [WAFFLE](https://arxiv.org/pdf/2008.07298.pdf) method 
    * **[Differential Privacy](https://github.com/intel/openfl/tree/develop/openfl-tutorials/experimental/Global_DP)** – Global differentially private federated learning using Opacus library to achieve a differentially private result w.r.t the inclusion or exclusion of any collaborator in the training process. At each round, a subset of collaborators are selected using a Poisson distribution over all collaborators, the selected collaborators perform local training with periodic clipping of their model delta (with respect to the current global model) to bound their contribution to the average of local model updates. Gaussian noise is then added to the average of these local models at the aggregator. This example is implemented in two different but statistically equivalent ways – the lower level API utilizes RDPAccountant and DPDataloader Opacus objects to perform privacy accounting and collaborator selection respectively, whereas the higher level API uses PrivacyEngine Opacus object for collaborator selection and internally utilizes RDPAccountant for privacy accounting. 
* **[Habana Accelerator Support](https://github.com/intel/openfl/tree/develop/openfl-tutorials/interactive_api/HPU/PyTorch_TinyImageNet)** 
* **Official support for Python 3.9 and 3.10** 
* **[EDEN Compression Pipeline](https://github.com/intel/openfl/blob/develop/openfl/pipelines/eden_pipeline.py)**: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning ([paper link](https://proceedings.mlr.press/v162/vargaftik22a.html)) 
* **[FLAX Framework Support](https://github.com/intel/openfl/tree/develop/openfl-tutorials/interactive_api/Flax_CNN_CIFAR)**
* **Improvements to the resiliency and security of the director / envoy infrastructure**: 
    * Optional notification to plan participants to agree to experiment sent to their infrastructure 
    * Improved resistance to loss of network connectivity and failure at various stages of execution
* **Windows Support (Experimental)**: Continuous Integration now tests OpenFL on Windows, but certain features may not work as expected. Full Windows support will be added in a future release.  

## 1.4
[Full Release Notes](https://github.com/securefederatedai/openfl/releases/tag/v1.4)

The OpenFL v1.4 release contains the following:

- [Straggler Handling](https://github.com/intel/openfl/pull/465)​
- tf.data [Pipeline Example​](https://github.com/intel/openfl/pull/440)
- [`PrivilegedAggregationFunction`](https://github.com/intel/openfl/pull/417) Interface​
- FeTS Challenge [Task Runner](https://github.com/intel/openfl/pull/419)​
- [JAX Framework Support](https://github.com/intel/openfl/pull/443)
- Bug fixes and other improvements

## 1.3
[Full Release Notes](https://github.com/securefederatedai/openfl/releases/tag/v1.3)

The OpenFL v1.3 release contains the following updates:

* [Task Assigner functionality](https://github.com/intel/openfl/pull/343)
* [OpenFL + Gramine to support workloads within SGX](https://github.com/intel/openfl/pull/339)
* [FedCurv aggregation](https://github.com/intel/openfl/pull/167) algorithm
* [HuggingFace/transformers audio classification example using SUPERB dataset](https://github.com/intel/openfl/pull/340)
* [PyTorch Lightning GAN](https://github.com/intel/openfl/pull/287) example
* NumPy Linear Regression example in [Google Colab](https://github.com/intel/openfl/pull/286)
* [Adaptive Federated Optimization ](https://github.com/intel/openfl/issues/281) algorithms implementation: `FedYogi`, `FedAdagrad`, `FedAdam`
* [MXNet landmarks regression example](https://github.com/intel/openfl/pull/349) as a custom plugin to OpenFL
* Migration to [JupyterLab](https://github.com/intel/openfl/pull/307)
* Bug fixes and other improvements

## 1.2
[Full Release Notes](https://github.com/securefederatedai/openfl/releases/tag/v1.2)

The OpenFL v1.2 release contains the following updates:

- Long-living entities: [Director/Envoy](https://github.com/intel/openfl/issues/120) for supporting multiple experiments within the same `Federation`
- [Scalable PKI](https://github.com/intel/openfl/issues/38): semi-automatic mechanism for certificates distribution via step-ca
- Examples with new Interactive API + Director/Envoy: [TensorFlow Next Word Prediction](https://github.com/intel/openfl/pull/183), [PyTorch Re-ID on Market](https://github.com/intel/openfl/pull/156), [PyTorch MobileNet v2 on TinyImageNet](https://github.com/intel/openfl/pull/170) 
- [3D U-Net TensorFlow workspace for BraTS 2020 for CLI-based workflow](https://github.com/intel/openfl/pull/108)
- `AggregationFunction` interface for custom aggregation functions in new Interactive API
- Autocomplete of `fx` CLI
- Bug fixes and documentation improvements
 
 ## 1.1
[Full Release Notes](https://github.com/securefederatedai/openfl/releases/tag/v1.1)

 The OpenFL v1.1 release contains the following updates:

- New [Interactive Python API](https://github.com/intel/openfl/blob/develop/openfl-tutorials/interactive_api_tutorials_(experimental)/Pytorch_Kvasir_UNET_workspace/new_python_api_UNET.ipynb) (experimental)
- Example FedProx algorithm implementation for PyTorch and Tensorflow
- `AggregationFunctionInterface` for custom aggregation functions 
- Adds a [Keras-based NLP Example](https://github.com/intel/openfl/tree/develop/openfl-workspace/keras_nlp)
- Fixed lossy compression pipelines and added an [example](https://github.com/intel/openfl/tree/develop/openfl-workspace/keras_cnn_with_compression) for usage
- Bug fixes and documentation improvements

 ## 1.0.1
[Full Release Notes](https://github.com/securefederatedai/openfl/releases/tag/v1.0.1)

v1.0.1 is a patch release. It includes the following updates:

- New docker CI tests
- New Pytorch UNet Kvasir tutorial
- Cleanup / fixes to other OpenFL tutorials
- Fixed description for Pypi
- Status/documentation/community badges for README.md

 ## 1.0
[Full Release Notes](https://github.com/securefederatedai/openfl/releases/tag/v1.0.1)

This release includes:
- The official open source release of OpenFL
- Tensorflow 2.0 and PyTorch support
- Examples for classification, segmentation, and adversarial training
- No-install Docker and Singularity* deployments
- Python native API intended for single node federated learning experiments
- `fx` CLI for multi-node production deployments
- Additional test coverage for OpenFL components

\* Singularity supported via DockerHub integration: `singularity shell docker://openfl:latest`
