
# Open Federated Learning (OpenFL)

[![PyPI - Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://pypi.org/project/openfl/)
[![Jenkins](https://img.shields.io/jenkins/build?jobUrl=http%3A%2F%2F213.221.44.203%2Fjob%2FFederated-Learning%2Fjob%2Fnightly%2F)](http://213.221.44.203/job/Federated-Learning/job/nightly/)
[![Documentation Status](https://readthedocs.org/projects/openfl/badge/?version=latest)](https://openfl.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/openfl)](https://pepy.tech/project/openfl)
[![PyPI version](https://img.shields.io/pypi/v/openfl)](https://pypi.org/project/openfl/)
[<img src="https://img.shields.io/badge/slack-@openfl-blue.svg?logo=slack">](https://join.slack.com/t/openfl/shared_invite/zt-ovzbohvn-T5fApk05~YS_iZhjJ5yaTw) 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Citation](https://img.shields.io/badge/cite-citation-blue)](https://arxiv.org/abs/2105.06413)

## About

Open Federated Learning (OpenFL) enables data scientists to set up a federated learning experiment:

- [aggregator-based workflow](https://openfl.readthedocs.io/en/docs_correction/source/workflow/running_the_federation.agg_based.html)
Creates a short-lived federation that runs one experiment

- [director-based workflow](https://openfl.readthedocs.io/en/docs_correction/source/workflow/director_based_workflow.html)
Sets up a long-lived federation with a single entry point to distribute experiments in series

OpenFL is a Python 3 project developed by Intel Labs and Intel Internet of Things Group.

### What is Federated Learning

[Federated learning](https://en.wikipedia.org/wiki/Federated_learning) is a distributed machine learning approach that enables collaboration on machine learning projects without having to share sensitive data, such as, patient records, financial data, or classified information. The minimum data movement needed across the federation is solely the model parameters and their updates.

![Federated Learning](https://raw.githubusercontent.com/intel/openfl/master/docs/images/diagram_fl.png)

### References
- [Sheller MJ,  et al., 2020](https://www.nature.com/articles/s41598-020-69250-1) 
- [Sheller MJ, et al., 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6589345)
- [Yang Y, et al., 2019](https://arxiv.org/abs/1902.04885)
- [McMahan HB, et al., 2016](https://arxiv.org/abs/1602.05629)


## Getting Started

Check out the [online documentation](https://openfl.readthedocs.io/en/latest/index.html) to launch your first federation. The quickest way to test OpenFL is to follow our [tutorials](https://openfl.readthedocs.io/en/docs_correction/source/workflow/running_the_federation.tutorial.html).

Consider joining our [Slack workspace](https://join.slack.com/t/openfl/shared_invite/zt-ovzbohvn-T5fApk05~YS_iZhjJ5yaTw).


## Requirements

- Tested on Ubuntu* Linux 16.04 and 18.04.
- Python* 3.6+ with a Python virtual environment (recommendation is [Virtualenv](https://virtualenv.pypa.io/en/latest/)).
- TensorFlow* 2+ or PyTorch* 1.6+ (depending on your training requirements). OpenFL is designed to support other frameworks as well.


## License
This project is licensed under [Apache License Version 2.0](LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.


## Citation

```
@misc{reina2021openfl,
      title={OpenFL: An open-source framework for Federated Learning}, 
      author={G Anthony Reina and Alexey Gruzdev and Patrick Foley and Olga Perepelkina and Mansi Sharma and Igor Davidyuk and Ilya Trushkin and Maksim Radionov and Aleksandr Mokrov and Dmitry Agapov and Jason Martin and Brandon Edwards and Micah J. Sheller and Sarthak Pati and Prakash Narayana Moorthy and Shih-han Wang and Prashant Shah and Spyridon Bakas},
      year={2021},
      eprint={2105.06413},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Support
We welcome questions, issue reports, and suggestions:

* [GitHub* Issues](https://github.com/intel/openfl/issues)
* [Slack channel](https://join.slack.com/t/openfl/shared_invite/zt-ovzbohvn-T5fApk05~YS_iZhjJ5yaTw)


## Background
OpenFL builds on the [OpenFederatedLearning](https://github.com/IntelLabs/OpenFederatedLearning) framework, which was a collaboration between Intel and the University of Pennsylvania (UPenn) to develop the [Federated Tumor Segmentation (FeTS, www.fets.ai)](https://www.fets.ai/) platform (grant award number: U01-CA242871). 

The grant for FeTS was awarded to the [Center for Biomedical Image Computing and Analytics (CBICA)](https://www.cbica.upenn.edu/) at UPenn (PI: S. Bakas) from the [Informatics Technology for Cancer Research (ITCR)](https://itcr.cancer.gov/) program of the National Cancer Institute (NCI) of the National Institutes of Health (NIH). 

FeTS is a real-world medical federated learning platform with international collaborators. The original OpenFederatedLearning project and OpenFL are designed to serve as the backend for the FeTS platform, 
and OpenFL developers and researchers continue to work very closely with UPenn on the FeTS project. An example is the [FeTS-AI/Front-End](https://github.com/FETS-AI/Front-End), which integrates UPenn’s medical AI expertise with Intel’s framework to create a federated learning solution for medical imaging. 

Although initially developed for use in medical imaging, OpenFL designed to be agnostic to the use-case, the industry, and the machine learning framework.

