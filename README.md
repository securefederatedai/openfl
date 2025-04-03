<div align="center">
  <img src="https://github.com/securefederatedai/artwork/blob/main/PNG/OpenFL%20Logo%20-%20color.png?raw=true" width="70%">
</div>

<div align="center">
	
[![PyPI version](https://img.shields.io/pypi/v/openfl)](https://pypi.org/project/openfl/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/openfl?color=green)](https://anaconda.org/conda-forge/openfl)
[![Downloads](https://pepy.tech/badge/openfl)](https://pepy.tech/project/openfl)
[![Documentation Status](https://readthedocs.org/projects/openfl/badge/?version=latest)](https://openfl.readthedocs.io/en/latest/?badge=latest)
[<img src="https://img.shields.io/badge/slack-@openfl-blue.svg?logo=slack">](https://join.slack.com/t/openfl/shared_invite/zt-ovzbohvn-T5fApk05~YS_iZhjJ5yaTw) 
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6599/badge)](https://bestpractices.coreinfrastructure.org/projects/6599)
<a href="https://scan.coverity.com/projects/securefederatedai-openfl">
  <img alt="Coverity Scan Build Status"
       src="https://scan.coverity.com/projects/29040/badge.svg"/>
</a>

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Features**](#features)
| [**Changelog**](https://openfl.readthedocs.io/en/latest/releases.html)
| [**Documentation**](https://openfl.readthedocs.io/en/latest/)

</div>

OpenFL is a Python framework for Federated Learning. It enables organizations to train and validate machine learning models on sensitive data. It increases privacy by allowing collaborative model training or validation across local private datasets without ever sharing that data with a central server. OpenFL is hosted by The Linux Foundation.

## Overview

Federated Learning is a distributed machine learning approach that enables collaborative training and evaluation of models without sharing sensitive data such as, personal information, patient records, financial data, or classified information. The minimum data movement needed across a Federated Training experiment, is solely the model parameters and their updates. This is in contrast to a Centralized Learning regime, where all data needs to be moved to a central server or a datacenter for massively parallel training.

![Federated Learning](https://openfl.readthedocs.io/en/latest/_images/ct_vs_fl.png)

OpenFL builds on a collaboration between Intel and the Bakas lab at the University of Pennsylvania (UPenn) to develop the [Federated Tumor Segmentation (FeTS)](https://www.fets.ai/) platform (grant award number: U01-CA242871).

The grant for FeTS was awarded from the [Informatics Technology for Cancer Research (ITCR)](https://itcr.cancer.gov/) program of the National Cancer Institute (NCI) of the National Institutes of Health (NIH), to Dr. Spyridon Bakas (Principal Investigator) when he was affiliated with the [Center for Biomedical Image Computing and Analytics (CBICA)](https://www.cbica.upenn.edu/) at UPenn and now heading the [Division of Computational Pathology at Indiana University (IU)](https://medicine.iu.edu/pathology/research/computational-pathology).

FeTS is a real-world medical federated learning platform with international collaborators. The original OpenFederatedLearning project and OpenFL are designed to serve as the backend for the FeTS platform, and OpenFL developers and researchers continue to work very closely with IU on the FeTS project. An example is the [FeTS-AI/Front-End](https://github.com/FETS-AI/Front-End), which integrates the group’s medical AI expertise with OpenFL framework to create a federated learning solution for medical imaging. 

Although initially developed for use in medical imaging, OpenFL designed to be agnostic to the use-case, the industry, and the machine learning framework.

For more information, here is a list of relevant [publications](https://openfl.readthedocs.io/en/latest/about/blogs_publications.html).

## Installation

Install via PyPI (latest stable release):

```
pip install -U openfl
```
Or via conda:
```
conda install conda-forge::openfl
```
For more installation options, checkout the [installation guide](https://openfl.readthedocs.io/en/latest/installation.html).

## Features

### Ways to set up an FL experiment
OpenFL supports two ways to set up a Federated Learning experiment:

- [TaskRunner API](https://openfl.readthedocs.io/en/latest/about/features_index/taskrunner.html): This API uses short-lived components like the `Aggregator` and `Collaborator`, which terminate at the end of an FL experiment. TaskRunner supports mTLS-based secure communication channels, and TEE-based confidential computing environments.

- [Workflow API](https://openfl.readthedocs.io/en/latest/about/features_index/workflowinterface.html): This API allows for experiments beyond the traditional horizontal federated learning paradigm using a pythonic interface. It allows an experiment to be simulated locally, and then to be seamlessly scaled to a federated setting by switching from a local runtime to a distributed, federated runtime.
	> **Note:** This is experimental capability.

### Framework Compatibility

OpenFL is backend-agnostic. It comes with support for popular NumPy-based ML frameworks like TensorFlow, PyTorch and Jax which should be installed separately. Users may extend the list of supported backends if needed.

### Aggregation Algorithms
OpenFL supports popular aggregation algorithms out-of-the-box, with more algorithms coming soon.
|  | Reference | PyTorch backend | TensorFlow backend | NumPy backend |
| -------------- | ----- | :--------------------: | :-----------------------: | :----------------------------: |
| FedAvg | [McMahan et al., 2017](https://arxiv.org/pdf/1602.05629.pdf) | yes | yes | yes |
| FedOpt | [Reddi et al., 2020](https://arxiv.org/abs/2003.00295) | yes | yes | yes |
| FedProx | [Li et al., 2020](https://arxiv.org/pdf/1812.06127.pdf) | yes | yes | - |
| FedCurv | [Shoham et al., 2019](https://arxiv.org/pdf/1910.07796.pdf) | yes | - | - |

## Contributing
We welcome contributions! Please refer to the [contributing guidelines](https://openfl.readthedocs.io/en/latest/contributing.html).

The OpenFL community is expanding, and we encourage you to join us. Connect with other enthusiasts, share knowledge, and contribute to the advancement of federated learning by joining our [Slack channel](https://join.slack.com/t/openfl/shared_invite/zt-ovzbohvn-T5fApk05~YS_iZhjJ5yaTw).

Stay updated by subscribing to the OpenFL mailing list: [openfl-announce@lists.lfaidata.foundation](mailto:openfl-announce@lists.lfaidata.foundation).


## License
This project is licensed under [Apache License Version 2.0](LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.


## Citation

```
@article{openfl_citation,
	author={Foley, Patrick and Sheller, Micah J and Edwards, Brandon and Pati, Sarthak and Riviera, Walter and Sharma, Mansi and Moorthy, Prakash Narayana and Wang, Shi-han and Martin, Jason and Mirhaji, Parsa and Shah, Prashant and Bakas, Spyridon},
	title={OpenFL: the open federated learning library},
	journal={Physics in Medicine \& Biology},
	url={http://iopscience.iop.org/article/10.1088/1361-6560/ac97d9},
	year={2022},
	doi={10.1088/1361-6560/ac97d9},
	publisher={IOP Publishing}
}
```