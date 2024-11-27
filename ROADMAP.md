# OpenFL Project Roadmap

This document is intended to give users and contributors an idea of OpenFL product team's current priorities, features we plan to incorporate over the short, medium, and long term, and call out opportunities for the community to get involved.

## When will this document be updated?
At a minimum once each product release - which we expect to be on a cadence of every 3-4 months. 

## 1. Features and interfaces

### 1.1 Workflows
All interfaces in OpenFL support the standard horizontal FL training workflow today:
1. The collaborator downloads the latest model from the aggregator
2. The collaborator performs validation with their local validation dataset on the aggregated model, and sends these metrics to the aggregator (aggregated_model_validation task)
3. The collaborator trains the model on their local training data set, and sends the local model weights and metrics to the aggregator (train task)
4. The collaborator performs validation with their local validation dataset on their locally trained model, and sends their validation metrics to the aggregator (locally_tuned_model_validation task)
5. The aggregator applies an aggregation function (weighted average, FedCurv, FedProx, etc.) to the model weights, and reports the aggregate metrics.

The [Task Assigner](https://github.com/securefederatedai/openfl/blob/develop/openfl-workspace/workspace/plan/defaults/assigner.yaml#L7-L9) determines the list of collaborator tasks to be performed, 
and both in the task runner API as well as the interactive API these tasks can be modified (to varying degrees).
For example, to perform federated evaluation of a model, only the `aggregated_model_validation` task would be listed for the assigner's block of the federated plan.
Equivalently for the interactive API, this can be done by only registering a single validation task.
But there are *many* other types of workflows that can't be easily represented purely by training / validation tasks performed on a collaborator with a single model.
An example is training a Federated Generative Adversarial Network (GAN); because this may be represented by separate generative and discriminator models, and could leak information about a collaborator dataset,
the interface we provide should allow for better control over what gets sent over the network and how. 
Another common request we get is for validation with an aggregator's dataset after training. Today there is not a great way to enable this in OpenFL. 

For these reasons, we are planning to add *experimental support* for complex distributed workflows in OpenFL 1.5, with the following goals: 

In the process of thinking about federated workflows, and the properties that are important, these are our goals:

1. Simplify the federated workflow representation
2. Clean separation of workflow from runtime infrastructure
4. Help users better understand the steps in federated learning (weight extraction, tensor compression, etc.)
5. Interface makes it clear what is sent across the network
6. The placement of tasks and how they connect should be straightforward
7. Don't reinvent unless absolutely necessary

### 1.2 Security, Privacy, and Governance
OpenFL is designed for security and privacy, and soon we will be releasing some exciting extensions that allow running OpenFL experiments within TEE environments.

### 1.3 Decoupling interface from infrastructure
The task runner interface is coupled with the the single experiment aggregator / collaborator infrastructure, and the interactive API is tied to the director / envoy infrastructure. 
The interactive API was originally designed to be a high-level API for OpenFL, but for the cases when more control is required by users, access to lower level interfaces is necessary.

### 1.4 Consolidating interfaces
Today we support three interfaces: TaskRunner, native Python API, and interactive API. These are all distinct APIs, and are not particularly interoperable.
By the time we reach OpenFL 2.0, our intention is to deprecate the original native [Python API](https://openfl.readthedocs.io/en/latest/get_started/examples.html#python-native-api) used for simulations, 
bring consistency to the remaining interfaces with a high level, middle level, and low level API that are **fully interoperable**. This will result in being able to use the interface you're most comfortable with for a simulation,
single experiment, or experiment session (with the director / envoy infrastructure).

### 1.5 Component standardization and framework interoperability

Federated Learning is a [burgeoning space](https://github.com/weimingwill/awesome-federated-learning#frameworks).
Most core FL infrastructure (model weight extraction, network protocols, and serialization designs) must be reimplemented ad hoc by each framework. 
This causes community fragmentation and distracts from some of the bigger problems to be solved in federated learning. In the short term, we want to collaborate on standards for FL,
 first at the communication and storage layer, and make these components modular across other frameworks. Our aim is also to provide a library for FL algorithms, compression methods,
 that can both be applied and interpreted easily.

## Upcoming OpenFL releases

### OpenFL 1.6 (Q4 2023)
1. Use the OpenFL Workflow Interface on distributed infrastructure with the [FederatedRuntime](https://openfl.readthedocs.io/en/latest/about/features_index/workflowinterface.html#runtimes-future-plans)
2. LLM Support
3. New use cases enabled by custom workflows
    * Standard ML Models (i.e. Tree-based algorithms)
4. Federated evaluation documentation and examples
6. Significantly improved documentation

### OpenFL 2.0 (2024)
1. Interface Cohesion
    * High level interface: Interactive API
    * Low level interface: Workflow API
2. Decoupling interfaces from infrastructure
3. Well defined interfaces intended for building higher level projects on top of OpenFL
