# Roadmap

This document is intended to give users and contributors an idea of the OpenFL team's current priorities, features we plan to incorporate over the short, medium, and long term, and call out opportunities for the community to get involved.

### When will this document be updated?
We expect to update this document at least once every quarter.

## Long-term directions

### Decoupling the FL specification interface from the infrastructure
The task runner interface is coupled with the the single experiment aggregator / collaborator infrastructure, and the Interactive API is tied to the director / envoy infrastructure. 
The Interactive API was originally designed to be a high-level API for OpenFL, but for the cases when more control is required by users, access to lower level interfaces is necessary.
In OpenFL 1.5, we introduced the Workflow API as an experimental feature, which can be used to specify the federated learning flow, independently of the underlying computing infrastructure. The Workflow API facilitates a seamless transition from local simulation to a federated setting. Additionally, this approach offers greater control over the sequence and content of the FL experiment steps, which enables more complex experiments beyond just horizontal FL. Workflow API also provides more granular privacy controls, allowing the model owner to explicitly permit or forbid the transfer of specific attributes over the network.

### Consolidating interfaces
OpenFL has supported multiple ways of running FL experiments for a long time, many of which are not interoperable: TaskRunner API, Workflow API, Python Native API, and Interactive API. The strategic vision is to consolidate OpenFL around the Workflow API, as it focuses on meeting the needs of the data scientist, who is the main user of the framework. Over the upcoming 1.x releases, we plan to gradually deprecate and eliminate the legacy Python Native API and Interactive API. OpenFL 2.0 will be centered around the Workflow API, facilitating a seamless transition from local simulations to distributed FL experiments, and even enabling the setup of permanent federations, which is currently only possible through the Interactive API.

### Component standardization and framework interoperability

Federated Learning is a [burgeoning space](https://github.com/weimingwill/awesome-federated-learning#frameworks).
Most core FL infrastructure (model weight extraction, network protocols, and serialization designs) must be reimplemented ad hoc by each framework. 
This causes community fragmentation and distracts from some of the bigger problems to be solved in federated learning. In the short term, we want to collaborate on standards for FL, first at the communication and storage layer, and make these components modular across other frameworks. Our aim is also to provide a library for FL algorithms, compression methods, that can both be applied and interpreted easily.

### Confidential computing support
Although OpenFL currently relies on Intel® SGX for trusted execution, the long term vision is towards broader confidential computing ecosystem support. This can be achieved by packaging OpenFL workspaces and workflows as Confidential Containers (CoCo), which supports a spectrum of TEE backends, including Intel® SGX and TDX, Arm TrustZone, and AMD SEV.

## Upcoming OpenFL releases

The roadmap for the upcoming releases is provided for informational purposes only. It is intended to offer visibility into our current planning and priorities. However, please note that the features and timelines outlined here are not commitments and are subject to change. We are continuously evaluating and adjusting our plans to best meet the needs of our users and the evolving technological landscape.

### 1.9 (May '25)
In the upcoming 1.9 release, our focus shifts to improving the resilience and scalability of the core OpenFL framework. Key initiatives include:
- Improved gRPC connection resilience
- Preparations for scaling to 10-s/100-s of collaborators
- First-class support for federated LLM fine-tuning
- Comprehensive FL plan consistency verifications to prevent incompatible configurations
- Experimental support for REST API as an alternative to the existing gRPC communication layer
- Support for data loading from object storage (S3)
  * This also includes enhanced dataset abstractions, with emphasis on integrity, composition and reuse
- Support for Federated Analytics via TaskRunner API
- Additional TaskRunner API utilities for validating the Aggregator/Collaborator infrastructure before executing the FL plan:
  * A new `fx collaborator ping` command to test collaborator/aggregator connectivity without starting any FL tasks or accessing private data
  * A [`no-op`](https://github.com/securefederatedai/openfl/tree/develop/openfl-workspace/no-op) workspace template that can be configured and distributed just for the purposes of establishing and testing connectivity and PKI

As a stretch goal, we are beginning preparations for the production-readiness of Workflow API (FederatedRuntime) via:
  * Improved controls of the types of data allowed across the network
  * Plan agreement mechanism for all experiment participants
  * Branching support in Workflow API (in line with the Metaflow API)
  * Streamlined TLS setup for distributed deployments (via FederatedRuntime)
  * Enhanced handling of straggler collaborators

### 1.10 (TBA)
We expect Workflow API to be the "star" of the 1.10 release, along with several other interoperability and security enhancements:
- Promote Workflow API as a core OpenFL feature, removing the experimental tag
- Integrate custom changes to support Flower workloads to core Aggregator and Collaborator components
- Support for semi-automated remote attestation of OpenFL nodes running in a TEE (starting with TaskRunner API)
- Design proposal for a SecureFederatedRuntime for Workflow API
- PoC for running OpenFL federations with CoCo for broader TEE frameworks support, beyond SGX
- ... (more details to be shared soon)