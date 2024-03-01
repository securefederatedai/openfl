<div align="center">
  <img src="https://github.com/securefederatedai/artwork/blob/main/PNG/OpenFL%20Logo%20-%20color.png?raw=true">
</div>

[![PyPI - Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/openfl/)
[![Ubuntu CI status](https://github.com/intel/openfl/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/intel/openfl/actions/workflows/ubuntu.yml)
[![Windows CI status](https://github.com/intel/openfl/actions/workflows/windows.yml/badge.svg)](https://github.com/intel/openfl/actions/workflows/windows.yml)
[![Documentation Status](https://readthedocs.org/projects/openfl/badge/?version=latest)](https://openfl.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/openfl)](https://pepy.tech/project/openfl)
[![DockerHub](https://img.shields.io/docker/pulls/intel/openfl.svg)](https://hub.docker.com/r/intel/openfl)
[![PyPI version](https://img.shields.io/pypi/v/openfl)](https://pypi.org/project/openfl/)
[<img src="https://img.shields.io/badge/slack-@openfl-blue.svg?logo=slack">](https://join.slack.com/t/openfl/shared_invite/zt-ovzbohvn-T5fApk05~YS_iZhjJ5yaTw) 
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Citation](https://img.shields.io/badge/cite-citation-brightgreen)](https://arxiv.org/abs/2105.06413)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6599/badge)](https://bestpractices.coreinfrastructure.org/projects/6599)
<a href="https://scan.coverity.com/projects/securefederatedai-openfl">
  <img alt="Coverity Scan Build Status"
       src="https://scan.coverity.com/projects/29040/badge.svg"/>
</a>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/openfl/blob/develop/openfl-tutorials/interactive_api/numpy_linear_regression/workspace/SingleNotebook.ipynb)

## Introduction
Open Federated Learning (OpenFL) is a Python 3 framework for Federated Learning. OpenFL is designed to be a _flexible_, _extensible_ and _easily learnable_ tool for data scientists. OpenFL is hosted by The Linux Foundation, aims to be community-driven, and welcomes contributions back to the project.

Landed Looking for the Open Flash Library project also referred to as OpenFL? Find it [here](https://github.com/openfl/openfl)!

| OpenFL | Readme Links | 
| -------------- | :--------------------: | 
| Background | [OpenFL Background](BACKGROUND.md) |
| Supported Aggregation Algorithms |[OpenFL Aggregation Algorithms](AGGREGATION-ALGOS.md) |

### What is Federated Learning
[Federated learning](https://en.wikipedia.org/wiki/Federated_learning) is a distributed machine learning approach that enables collaboration on machine learning projects without having to share sensitive data, such as, patient records, financial data, or classified information. The minimum data movement needed across the federation is solely the model parameters and their updates.

## Quick Start Guide
The Quick Start Guide showcases setting up a federation using the TaskRunner API on a single machine (simulation mode)
|   |  | 
| -------------- | -------- |
| Containing 3 Participants | 1. Aggregator  <br/> 2. Collaborator1  <br/> 3. Collaborator2  |

**Guide in few simple steps.** _(Tip: use fx --help for any command usage details)_
```
0. Setup Prerequisites: Install OpenFL in a conda environment, setup proxies & FQDN.
1. Setup Federation Workspace & Certificate Authority(CA): Create federation workspace with FL Plan & Certify it as CA.
2. Setup Aggregator: Generate Aggregator TLS Certificate signed by CA.
3. Setup Collaborator1: Create & certify collaborator1 workspace.
4. Setup Collaborator2: Create & certify collaborator2 workspace.
5. Run the Federation: Start the Aggregator, Collaborator1 & Collaborator2.
```
### Step 0: Prerequisites
Please be sure you have Miniconda or Anaconda installed.
#### _Setup Miniconda environment_
```bash
conda create -n fedai python=3.10 # officially supports versions (>=3.7, <=3.10) 
conda activate fedai
```
#### _Install OpenFL from source_
```bash
git clone https://github.com/intel/openfl.git
python -m pip install -U pip setuptools wheel
cd openfl/
python -m pip install .
python -c "import openfl; print('Successfully installed OpenFL v{}'.format(openfl.__version__))"
```
#### _Setup fully qualified domain name [(FQDN)](https://en.wikipedia.org/wiki/Fully_qualified_domain_name)_
```bash
echo $(hostname --all-fqdns |  awk '{print tolower($1)}') # Find  the right FQDN
```
```bash
export FQDN=<FQDN> # Enter your FQDN here
export no_proxy=localhost,<local machine IP>,<FQDN>
```
### Step 1: Setup Federation Workspace & Certificate Authority(CA)
#### _Step 1a: Create a federation workspace with necessary dependencies & initialize  [FL Plan](https://openfl.readthedocs.io/en/latest/about/features_index/taskrunner.html#federated-learning-plan-fl-plan-settings)_
```bash
cd ${HOME}
fx workspace create --prefix "my_federation_workspace" --template keras_cnn_mnist #setup workspace with keras_cnn_mnist as model.
cd "my_federation_workspace"
fx plan initialize # initialize the workspace with FLPlan settings.
```
#### _Step 1b: Assign current workspace as the Certificate Authority._
```bash
fx workspace certify    # (Sets up aggregator as certificate authority(CA) populates cert folder)
```
### Step 2: Setup Aggregator
```bash
fx aggregator generate-cert-request --fqdn $FQDN #Generates the csr & key inside cert/server
fx aggregator certify --fqdn $FQDN --silent  #certify the aggregator workspace using CA
```

### Step 3: Setup Collaborator1
```bash
fx collaborator create -n cob_1 -d 1 #create collaborator 1
fx collaborator generate-cert-request -n cob_1 #Generate Certificate request
fx collaborator certify -n cob_1 --silent #Have CA to certify the collaborator
```
### Step 4: Setup Collaborator2
```bash
fx collaborator create -n cob_2 -d 2 #create collaborator 2
fx collaborator generate-cert-request -n cob_2 #Generate Certificate request
fx collaborator certify -n cob_2 --silent #Have CA to certify the collaborator
```

### Step 5: Run the federation (In 3 separate terminals)
_See [Modes and API](#modes-and-associated-api) for supported aggregation algorithms._
#### _Step 5a: Start Aggregator_
```bash
conda activate fedai && cd ${HOME}/my_federation_workspace
fx aggregator start
```
#### _Step 5b: Start Collaborator1_
```bash
conda activate fedai && cd ${HOME}/my_federation_workspace
fx collaborator start -n cob_1
```
#### _Step 5c: Start Collaborator2_
```bash
conda activate fedai && cd ${HOME}/my_federation_workspace
fx collaborator start -n cob_2
```
### Expected output in Aggregator terminal
```bash
INFO     Starting round 9...                                                                                                        aggregator.py:897
INFO     Sending tasks to collaborator cob_2 for round 9                                                                            aggregator.py:329
INFO     Collaborator cob_2 is sending task results for aggregated_model_validation, round 9                                        aggregator.py:520
METRIC   Round 9, collaborator validate_agg aggregated_model_validation result accuracy: 0.989598                                   aggregator.py:559
INFO     Sending tasks to collaborator cob_1 for round 9                                                                            aggregator.py:329
INFO     Collaborator cob_1 is sending task results for aggregated_model_validation, round 9                                        aggregator.py:520
METRIC   Round 9, collaborator validate_agg aggregated_model_validation result accuracy: 0.987200                                   aggregator.py:559
INFO     Collaborator cob_2 is sending task results for train, round 9                                                              aggregator.py:520
METRIC   Round 9, collaborator metric train result loss: 0.026069                                                                   aggregator.py:559
INFO     Collaborator cob_2 is sending task results for locally_tuned_model_validation, round 9                                     aggregator.py:520
METRIC   Round 9, collaborator validate_local locally_tuned_model_validation result accuracy:    0.988198                           aggregator.py:559
INFO     Collaborator cob_1 is sending task results for train, round 9                                                              aggregator.py:520
METRIC   Round 9, collaborator metric train result loss: 0.008339                                                                   aggregator.py:559
INFO     Collaborator cob_1 is sending task results for locally_tuned_model_validation, round 9                                     aggregator.py:520
METRIC   Round 9, collaborator validate_local locally_tuned_model_validation result accuracy:    0.981800                           aggregator.py:559
METRIC   Round 9, aggregator: locally_tuned_model_validation                                                                        aggregator.py:838
<openfl.interface.aggregation_functions.weighted_average.WeightedAverage object at 0x7f6d725509d0> accuracy: 0.984999
METRIC   Round 9, aggregator: train <openfl.interface.aggregation_functions.weighted_average.WeightedAverage object at              aggregator.py:838
0x7f6d725509d0> loss:     0.017204
METRIC   Round 9, aggregator: aggregated_model_validation <openfl.interface.aggregation_functions.weighted_average.WeightedAverage  aggregator.py:838
object at 0x7f6d725509d0> accuracy:   0.988399
INFO     Saving round 10 model...                                                                                                   aggregator.py:890
INFO     Experiment Completed. Cleaning up...                                                                                       aggregator.py:895
INFO     Sending signal to collaborator cob_1 to shutdown...                                                                        aggregator.py:283
INFO     Sending signal to collaborator cob_2 to shutdown...                                                                        aggregator.py:283
✔️ OK
```
### Expected output in Collaborator1 or Collaborator2 terminals
```
 INFO     Using TaskRunner subclassing API                                                                                         collaborator.py:253
157/157 [==============================] - 1s 6ms/step - loss: 0.0416 - accuracy: 0.9872
METRIC   Round 9, collaborator cob_1 is sending metric for task aggregated_model_validation: accuracy    0.987200                 collaborator.py:415
INFO     Using TaskRunner subclassing API                                                                                         collaborator.py:253
INFO     Run 0 epoch of 9 round                                                                                                    runner_keras.py:83
938/938 [==============================] - 11s 11ms/step - loss: 0.0083 - accuracy: 0.9975
METRIC   Round 9, collaborator cob_1 is sending metric for task train: loss      0.008339                                         collaborator.py:415
INFO     Using TaskRunner subclassing API                                                                                         collaborator.py:253
157/157 [==============================] - 1s 7ms/step - loss: 0.0790 - accuracy: 0.9818
METRIC   Round 9, collaborator cob_1 is sending metric for task locally_tuned_model_validation: accuracy 0.981800                 collaborator.py:415
INFO     Waiting for tasks...                                                                                                     collaborator.py:178
INFO     End of Federation reached. Exiting...                                                                                    collaborator.py:150
✔️ OK
```

## How it Works
### Architecture
![Federated Learning](https://raw.githubusercontent.com/intel/openfl/develop/docs/images/diagram_fl_new.png)

## Get Started with detailed guides
For more installation options check out the [online documentation](https://openfl.readthedocs.io/en/latest/install.html). <br>
OpenFL enables data scientists to set up a federated learning experiment following one of the workflows using the associated API, each with its own benefits <br>

| Notes | Links | 
| -------------- | ----- |
Quick Test OpenFL using     | [Quick Start steps for singlenode](#quick-start-guide) |
Quickest Start OpenFL using | [Tutorials](https://github.com/intel/openfl/tree/develop/openfl-tutorials) |
Read                        | [Blog Post](https://towardsdatascience.com/go-federated-with-openfl-8bc145a5ead1) explaining steps to train a model with OpenFL |
Launch Federation using     | [Online Documentation](https://openfl.readthedocs.io/en/latest/index.html) to launch your first federation |

### Modes and associated API
| API | Flexibility | Ease of Use | Simulation Mode <br>(Single Node) |  Distributed Mode <br>(Multiple Nodes) | Production Mode <br>(Real-World Federations) |
| -------------- | :--------------------: | :-----------------------: | :----------------------------: | :----------: | :----------: |
| [ Task Runner](https://openfl.readthedocs.io/en/latest/running_the_federation.html#aggregator-based-workflow "Define an experiment and distribute it manually. All participants can verify model code and FL plan prior to execution. The federation is terminated when the experiment is finished") | ✅ | ❌ | ✅ | ✅ | ✅ | 
| Python Native | ❌ | ✅ | ✅ | ❌ | ❌ |
| [Interative](https://openfl.readthedocs.io/en/latest/running_the_federation.html#director-based-workflow "Setup long-lived components to run many experiments in series. Recommended for FL research when many changes to model, dataloader, or hyperparameters are expected") | ❌ | ✅ | ✅ | ✅ | ❌ |
| [Workflow Interface](https://openfl.readthedocs.io/en/latest/workflow_interface.html "Create complex experiments that extend beyond traditional horizontal federated learning. See the experimental tutorials to learn how to coordinate aggregator validation after collaborator model training, perform global differentially private federated learning, measure the amount of private information embedded in a model after collaborator training with privacy meter, or add a watermark to a federated model")  | ✅ | ✅ | ✅ | ❌ | ❌ |

## Support
**Join us:** bi-monthly community meetings to meet with some team members behind OpenFL for discussions on our roadmap, open Q&A, & idea sharing. <br>
**Calendar and links to Community calls:** [here](https://wiki.lfaidata.foundation/pages/viewpage.action?pageId=70648254) <br>
**Subscribe to the OpenFL mail list:** openfl-announce@lists.lfaidata.foundation

We also always welcome questions, issue reports, and suggestions via:
* [GitHub Issues](https://github.com/intel/openfl/issues)
* [Slack workspace](https://join.slack.com/t/openfl/shared_invite/zt-ovzbohvn-T5fApk05~YS_iZhjJ5yaTw)

## License
This project is licensed under [Apache License Version 2.0](LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.
