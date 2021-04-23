
# Welcome to Intel&reg; Open Federated Learning

[![PyPI - Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://pypi.org/project/openfl/)
[![Jenkins](https://img.shields.io/jenkins/build?jobUrl=http%3A%2F%2F213.221.44.203%2Fjob%2FFederated-Learning%2Fjob%2Fnightly%2F)](http://213.221.44.203/job/Federated-Learning/job/nightly/)
[![Documentation Status](https://readthedocs.org/projects/openfl/badge/?version=latest)](https://openfl.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/openfl)](https://pypi.org/project/openfl/)
[<img src="https://img.shields.io/badge/slack-@openfl-blue.svg?logo=slack">](https://join.slack.com/t/openfl/shared_invite/zt-ovzbohvn-T5fApk05~YS_iZhjJ5yaTw) 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[Federated learning](https://en.wikipedia.org/wiki/Federated_learning) is a distributed machine learning approach that
enables organizations to collaborate on machine learning projects
without sharing sensitive data, such as, patient records, financial data,
or classified secrets 
([Sheller MJ,  et al., 2020](https://www.nature.com/articles/s41598-020-69250-1);
[Sheller MJ, et al., 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6589345);
[Yang Y, et al., 2019](https://arxiv.org/abs/1902.04885);
[McMahan HB, et al., 2016](https://arxiv.org/abs/1602.05629)).


The basic premise behind federated learning
is that the model moves to meet the data rather than the data moving
to meet the model. Therefore, the minimum data movement needed
across the federation is solely the model parameters and their updates.


Open Federated Learning (OpenFL) is a Python 3 project developed by Intel Labs and 
Intel Internet of Things Group. 

![Federated Learning](https://raw.githubusercontent.com/intel/openfl/master/docs/images/diagram_fl.png)

## Getting started

Check out our [online documentation](https://openfl.readthedocs.io/en/latest/index.html) to launch your first federation.  The quickest way to test OpenFL is through our [Jupyter Notebook tutorials](https://openfl.readthedocs.io/en/latest/running_the_federation.notebook.html).

For more questions, please consider joining our [Slack channel](https://openfl.slack.com).


## Requirements

- OS: Tested on Ubuntu Linux 16.04 and 18.04.
- Python 3.6+ with a Python virtual environment (e.g. [conda](https://docs.conda.io/en/latest/))
- TensorFlow 2+ or PyTorch 1.6+ (depending on your training requirements). OpenFL is designed to easily support other frameworks as well.

![fx commandline interface](https://raw.githubusercontent.com/intel/openfl/master/docs/images/fx_help.png)

## License
This project is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

## Resources:
* Docs and Tutorials: https://openfl.readthedocs.io/en/latest/index.html
* Issue tracking: https://github.com/intel/openfl/issues
* [Slack channel](https://openfl.slack.com)

## Support
Please report questions, issues and suggestions using:

* [GitHub* Issues](https://github.com/intel/openfl/issues)
* [Slack channel](https://openfl.slack.com)

### Relation to OpenFederatedLearning and the Federated Tumor Segmentation (FeTS) Initiative

This project builds on the [Open Federated Learning](https://github.com/IntelLabs/OpenFederatedLearning) framework that was 
developed as part of a collaboration between Intel
and the University of Pennsylvania (UPenn) for federated learning. 
It describes Intel’s commitment in 
supporting the grant awarded to the [Center for Biomedical Image Computing and Analytics (CBICA)](https://www.cbica.upenn.edu/) 
at UPenn (PI: S. Bakas) from the [Informatics Technology for Cancer Research (ITCR)](https://itcr.cancer.gov/) program of 
the National Cancer Institute (NCI) of the National Institutes of Health (NIH), 
for the development of the [Federated Tumor Segmentation (FeTS, www.fets.ai)](https://www.fets.ai/) 
platform (grant award number: U01-CA242871). 

FeTS is an exciting, real-world 
medical FL platform, and we are honored to be collaborating with UPenn in 
leading a federation of international collaborators. The original OpenFederatedLearning
project and OpenFL are designed to serve as the backend for the FeTS platform, 
and OpenFL developers and researchers continue to work very closely with UPenn on 
the FeTS project. The 
[FeTS-AI/Front-End](https://github.com/FETS-AI/Front-End) shows how UPenn 
and Intel have integrated UPenn’s medical AI expertise with Intel’s framework 
to create a federated learning solution for medical imaging. 

Although initially developed for use in medical imaging, this project is
designed to be agnostic to the use-case, the industry, and the 
machine learning framework.

