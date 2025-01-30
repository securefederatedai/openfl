# Linear Regression with Numpy and OpenFL

This example is devoted to demonstrating several techniques of working with OpenFL.

1. Envoy workspace contains a Shard Descriptor designed to generate 1-dimensional noisy data for linear regression of sinusoid. The random seed for generation for a specific Envoy is parametrized by the `rank` argument in shard_config. 
2. The LinReg frontend jupyter notebook (data scientist's entry point) features a simple numpy-based model for linear regression trained with Ridge regularization.
3. The data scientist's workspace also contains a custom framework adapter allowing extracting and setting weights to the custom model.
4. The start_federation notebook provides shortcut methods to start a Federation with an arbitrary number of Envoys with different datasets. It may save time for people willing to conduct one-node experiments.
5. The SingleNotebook jupyter notebook combines two aforementioned notebooks and allows to run the whole pipeline in Google colab. Besides previously mentioned components, it contains scripts for pulling the OpenFL repo with the example workspaces and installing dependencies.

## How to use this example
### Locally:
1. Start a Federation
Distributed experiments:
Use OpenFL CLI to start the Director and Envoy services from corresponding folders. 
Single-node experiments:
Users may use the same path or benefit from the start_federation notebook in the workspace folder

2. Submit an experiment
Follow LinReg jupyter notebook.

### Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/openfl/blob/develop/openfl-tutorials/interactive_api/numpy_linear_regression/workspace/SingleNotebook.ipynb)
