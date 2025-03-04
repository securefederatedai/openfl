# Tutorials

These tutorials cover a range of frameworks, models, and datasets to help you get started with OpenFL. This directory provides notebooks for the [Workflow API](https://openfl.readthedocs.io/en/latest/about/features_index/workflowinterface.html), one of two ways to run Federated Learning experiments in OpenFL. 

> [!NOTE]
> If you are looking for an enterprise-ready API with support for Trusted Execution Environments (TEEs), refer to the [TaskRunner API](https://openfl.readthedocs.io/en/latest/about/features_index/taskrunner.html), and the corresponding [quickstart](https://openfl.readthedocs.io/en/latest/tutorials/taskrunner.html) guide.

## Getting Started

Install OpenFL following the [installation guide](https://openfl.readthedocs.io/en/latest/installation.html) and activate workflow API.

```bash
fx experimental activate
```

### Beginner

- [MNIST](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/101_MNIST.ipynb): An introductory guide to Workflow Interface for federated learning with a small PyTorch CNN model on the MNIST dataset.
- [Aggregator Validation](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/102_Aggregator_Validation.ipynb): Shows how to perform validation on the aggregator after training using OpenFL Workflow Interface.
- [Cyclic Institutional Incremental Learning](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/103_Cyclic_Institutional_Incremental_Learning.ipynb): Demonstrates cyclic institutional incremental learning using OpenFL Workflow Interface.
- [Keras MNIST with CPU](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/104_Keras_MNIST_with_CPU.ipynb): Trains a CNN on MNIST using Keras on CPU with OpenFL Workflow Interface.
- [Keras MNIST with GPU](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/104_Keras_MNIST_with_GPU.ipynb): Uses Keras to train a CNN on MNIST with GPU support via OpenFL Workflow Interface.
- [MNIST XPU](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/104_MNIST_XPU.ipynb): Trains a CNN on MNIST using Intel Data Center GPU Max Series with OpenFL Workflow Interface.
- [Numpy Linear Regression](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/105_Numpy_Linear_Regression_Workflow.ipynb): Implements linear regression with MSE loss using Numpy and OpenFL Workflow Interface.
- [Scikit Learn Linear Regression](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/106_Scikit_Learn_Linear_Regression_Workflow.ipynb): Trains a linear regression model with Ridge regularization using scikit-learn and OpenFL Workflow Interface.
- [Exclusive GPUs with Ray](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/201_Exclusive_GPUs_with_Ray.ipynb): Utilizes Ray Backend for exclusive GPU allocation in federated learning with OpenFL.
- [MNIST Watermarking](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/301_MNIST_Watermarking.ipynb): Demonstrates embedding watermark in a DL model trained on MNIST in a federated learning setup.
- [FedProx with Synthetic non-IID](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/401_FedProx_with_Synthetic_nonIID.ipynb): Compares FedProx and FedAvg algorithms on a synthetic non-IID dataset using OpenFL Workflow Interface.
- [MNIST Aggregator Validation Ray Watermarking](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/402_MNIST_Aggregator_Validation_Ray_Watermarking.ipynb): Combines aggregator validation, watermarking, and multi-GPU training using Ray Backend in OpenFL.
- [Federated FedProx PyTorch MNIST Workflow Tutorial](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/403_Federated_FedProx_PyTorch_MNIST_Workflow_Tutorial.ipynb): Implements FedProx algorithm for distributed training on MNIST using PyTorch and Workflow API.
- [Keras MNIST with FedProx](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/404_Keras_MNIST_with_FedProx.ipynb): Trains a TensorFlow Keras model on MNIST using OpenFL Workflow Interface with FedProx optimization.
- [Federated Evaluation MNIST Workflow Tutorial](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/405_MNIST_FederatedEvaluation.ipynb): Shows how to implement Federated Evaluation using Workflow API.
- [Workspace Creation from JupyterNotebook](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/1001_Workspace_Creation_from_JupyterNotebook.ipynb): Demonstrates converting a Jupyter Notebook Federated Learning experiment into an OpenFL workspace.

### Advanced

Refer to the respective README files for detailed information on the executing these tutorials.

- [CrowdGuard](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/CrowdGuard): Implementing CrowdGuard to mitigate backdoor attacks in Federated Learning.
- [Federated Runtime](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/FederatedRuntime): Demonstrates use of FederatedRuntime for taking workflow API from simulation to deployment.
- [Differential Privacy](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/Global_DP): Differential Privacy in Federated Learning.
- [LLM Fine-tuning](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/LLM): Fine-tuning a Large Language Model (LLM) using OpenFL Workflow Interface.
- [Privacy Meter](https://github.com/securefederatedai/openfl/tree/develop/openfl-tutorials/experimental/workflow/Privacy_Meter): Demonstrates integration and use of ML Privacy Meter library with OpenFL to audit privacy loss in federated learning models.