
## Differential Privacy (DP)
Neural networks when trained adequately, pose a risk of learning labels and unique patterns in training data. In a distributed training setup like federated learning, such behaviors cause a significant security compromise since the model owner is now privy to some patterns in the training data. Hence, a privacy protection criterion that provides a mathematically provable guarantee of privacy protection is required. Differential Privacy is one such method which rules out memorization of sensitive information. For our case, we use collaborator level privacy, meaning that the final model is epsilon, delta differentially private with respect to inclusion or exclusion of any one collaborator.

## Global DP example
Global DP implementation uses the [Opacus library](https://opacus.ai/) to perform global differentially private federated learning. At each round, a subset of collaborators are selected using a Poisson distribution over all collaborators, the selected collaborators perform local training with periodic clipping of their model delta (with respect to the current global model) to bound their contribution to the average of local model updates. Gaussian noise is then added to the average of these local models at the aggregator. The result of this federated training is then differentially private with respect to the inclusion (or exclusion) of any one collaborator into the training. A Renyi differential privacy (RDP) accountant is used at the aggregator to determine the epsilon, delta for which the final model is epsilon, delta differentially private with respect to inclusion or exclusion of any one collaborator. This tutorial has two implementations which are listed below. The two implementations are statistically equivalent, but incorporate Opacus interfaces at two different levels. Both implementations use the same code for local training and periodic clipping.

Prerequisites:

`pip install -r ../workflow_interface_requirements.txt`
`pip install -r requirements_global_dp.txt`

1. `Workflow_Interface_Mnist_Implementation_1.py` uses lower level RDPAccountant and DPDataLoader Opacus objects to perform the privacy accounting and collaborator selection respectively. Local model aggregation and noising is implemented independent of Opacus, and final accounting is calculated by the RDPAccountant, using information about how many rounds of training was performed. To run with this version:

`python Workflow_Interface_Mnist_Implementation_1.py --config_path test_config.yml`

2. `Workflow_Interface_Mnist_Implementation_2.py` uses the higher level PrivacyEngine Opacus object to wrap (using the 'make_private' method) a global data loader (serving up collaborator names), and a global optimizer (performing the local model aggregation and noising), with RDP accounting being performed internally by PrivacyEngine utilizing the fact that it tracks the usage of these wrapped objects over the course of training. To run with this version:

`python Workflow_Interface_Mnist_Implementation_2.py --config_path test_config.yml`

