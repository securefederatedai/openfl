# On the Integration of CrowdGuard into OpenFL
Federated Learning (FL) is a promising approach enabling multiple clients to train Deep Neural Networks (DNNs) collaboratively without sharing their local training data. However, FL is susceptible to backdoor (or targeted poisoning) attacks. These attacks are initiated by malicious clients who seek to compromise the learning process by introducing specific behaviors into the learned model that can be triggered by carefully crafted inputs. Existing FL safeguards have various limitations: They are restricted to specific data distributions or reduce the global model accuracy due to excluding benign models or adding noise, are vulnerable to adaptive defense-aware adversaries, or require the server to access local models, allowing data inference attacks.

This tutorial implements CrowdGuard [1], which effectively mitigates backdoor attacks in FL and overcomes the deficiencies of existing techniques. It leverages clients' feedback on individual models, analyzes the behavior of neurons in hidden layers, and eliminates poisoned models through an iterative pruning scheme. CrowdGuard employs a server-located stacked clustering scheme to enhance its resilience to rogue client feedback. The experiments that were conducted in the paper show a 100% True-Positive-Rate and True-Negative-Rate across various scenarios, including IID and non-IID data distributions. Additionally, CrowdGuard withstands adaptive adversaries while preserving the original performance of protected models. To ensure confidentiality, CrowdGuard requires a secure and privacy-preserving architecture leveraging Trusted Execution Environments (TEEs) on both client and server sides. Full instructions to set up CrowdGuard's workflows inside TEEs using the OpenFL Workflow API will be made available in a future release of OpenFL.



## Threat Model
Following this, we consider two threat models.
- Backdoor Attacks: Malicious clients aim to inject a backdoor by uploading manipulated model updates.
- Privacy Attacks: The attacker aims to infer information about the clients' data from their local models. Thus, the server tries to gain access to the local models before their aggregation. The clients try to gain access to other clients' local models.


## Workflow
We provide a demo code in `cifar10_crowdguard.py` as well as an interactive version as notebook. In the following, we briefly describe the workflow.
In each FL training round, each client trains the global model using its local dataset. Afterward, the server collects the local models and sends them to the clients for the local validation. The clients report the identified suspicious models to the server, which combines these votes using the stacked-clustering scheme to identify the poisoned models. At the end of each round, the identified benign models are aggregated using FedAVG.

## Methodology
We implemented a simple scaling-based poisoning attack to demonstrate the effectiveness of CrowdGuard.

For the local validation in CrowdGuard, each client uses its local dataset to obtain the hidden layer outputs for each local model. Then it calculates the Euclidean and Cosine Distance, before applying a PCA. Based on the first principal component, CrowdGuard employs several statistical tests to determine whether poisoned models remain and removes the poisoned models using clustering. This process is repeated until no more poisoned models are detected before sending the detected poisoned models to the server. On the server side, the votes of the individual clients are aggregated using a stacked-clustering scheme to prevent malicious clients from manipulating the aggregation process through manipulated votes. The client-side validation as well as the server-side operations, are executed with SGX to prevent privacy attacks.

[1] Rieger, P., Krauß, T., Miettinen, M., Dmitrienko, A., & Sadeghi, A. R. CrowdGuard: Federated Backdoor Detection in Federated Learning. NDSS 2024.

## Running the CIFAR-10 demo script
The demo script requires a dedicated allocation of at least 18GB of RAM to run without issues.

1) Create a Python virtual environment for better isolation
```shell
python -m venv venv
source venv/bin/activate
```
2) Install OpenFL from the latest sources
```shell
git clone https://github.com/securefederatedai/openfl.git && cd openfl
pip install -e .
```
3) Install the requirements for Workflow API
```shell
cd openfl-tutorials/experimental/workflow
pip install -r workflow_interface_requirements.txt
```
4) Start the training script<br/>
Note that the number of training rounds can be adjusted via the `--comm_round` parameter:
```shell
cd CrowdGuard
python cifar10_crowdguard.py --comm_round 5
```

## Running the demo script in a true federated runtime

You will need five terminals. One for the director and four for the envoys.

1st Terminal

```sh
fx experimental activate
cd director
./start_director.sh
```

2nd Terminal

```sh
cd Amsterdam
./start_envoy.sh Amsterdam Amsterdam_config.yaml
```

3rd Terminal

```sh
cd Bangalore
./start_envoy.sh Bangalore Bangalore_config.yaml
```

4th Terminal

```sh
cd Chandler
./start_envoy.sh Chandler Chandler_config.yaml
```

5th Terminal

```sh
cd Detroit
./start_envoy.sh Detroit Detroit_config.yaml
```

Now that your director and envoy terminals are set up, run the Jupyter Notebook `FederatedCrowdGuard.ipynb`. For this I am using the Jupyter Extension for VS Code.
