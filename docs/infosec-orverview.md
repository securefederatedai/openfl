# InfoSec Overview

_Last updated: 19 May 2025_

### Purpose
This document provides the information needed when evaluating OpenFL for real world deployment in highly sensitive environments. The target audience is InfoSec reviewers who need detailed information about code contents, communication traffic, and potential exploit vectors.

### Overview: Network Connectivity
OpenFL federations use a hub-and-spoke topology between `collaborator` clients that generate model parameter updates from their data and the `aggregator` server that combines their training updates into new models[^1]. Key details about this functionality are:

* Connections are made using request/response gRPC[^2] connections.
* The `aggregator` listens for connections on a single port (usually decided by the experiment admin), and is explicitly defined in the FL plan (f.e. `50051`), so all `collaborator`s must be able to send outgoing traffic to this port.
* All connections are initiated by the `collaborator`, i.e., a [`pull`](https://karlchris.github.io/data-engineering/data-ingestion/push-pull/#pull) architecture.
* The `collaborator` does not open any listening sockets.
* Connections are secured using mTLS[^3].
* Each request response pair is done on a new TLS connection.
* The PKI for federations is created using the [`fx aggregator/collaborator certify`](https://openfl.readthedocs.io/en/latest/fx.html) CLI command. OpenFL internally leverages Python's cryptography module. The organization hosting the `aggregator` usually acts as the Certificate Authority (CA) and verifies each identity before signing.
* Currently, the `collaborator` polls the `aggregator` at a fixed interval. We have had a request to enable client-side configuration of this interval and hope to support that feature soon.
* Connection timeouts are set to gRPC defaults.
* If the `aggregator` is not available, the `collaborator` will retry connections indefinitely. This is currently useful so that we can take the aggregator down for bugfixes without `collaborator` processes exiting.

### Contents of Network Messages
Network messages are well defined protobufs which can be found in the following files:
- [`aggregator.proto`](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/aggregator.proto)
- [`base.proto`](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/base.proto)

Key points about the network messages/protocol:
* No executable code is ever sent to the collaborator. All code to be executed is contained within the OpenFL package and the custom FL workspace. The code, along with the FL plan file that specifies the classes and initial parameters to be used, is available for review prior to the FL plans execution. This ensures that all potential operations are understood before they take place.
* The `collaborator` typically requests the FL tasks to execute from the aggregator via a [`GetTasksRequest`](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/aggregator.proto#L34) message.
* The `aggregator` based on the FL plan, returns a [`GetTasksResponse`](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/aggregator.proto#L45) which includes [`Tasks`](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/aggregator.proto#L38) - metadata about the Python functions to be invoked by the collaborator. All code is available locally to each collaborator as part of a pre-distributed workspace bundle.
* The `collaborator` then uses its TaskRunner framework to execute the FL tasks on the locally available data, producing output tensors such as model weights or metrics.
* During task execution, the `collaborator` may require certain tensors for task execution that are not available locally. For example, a federated training task requires globally averaged model weights from the `aggregator`. Collaborators gather a list of tensor keys that need to be fetched from the aggregator and download them via the [`GetAggregatedTensors`](https://openfl.readthedocs.io/en/latest/reference/_autosummary/openfl.transport.grpc.aggregator_server.AggregatorGRPCServer.html#openfl.transport.grpc.aggregator_server.AggregatorGRPCServer.GetAggregatedTensor) RPC method.
* Upon task completion, the `collaborator` transmits the results by emitting a [`SendLocalTaskResults`](https://openfl.readthedocs.io/en/latest/reference/_autosummary/openfl.transport.grpc.aggregator_server.AggregatorGRPCServer.html#openfl.transport.grpc.aggregator_server.AggregatorGRPCServer.SendLocalTaskResults) RPC method which contains [`NamedTensor`](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/base.proto#L11) objects that encode results (like model weight updates or metrics such as loss or accuracy).

### Testing a Collaborator
There is a "no-op" workspace template in OpenFL (available in versions `>=1.9`) which can be used to test the network connection between the `aggregator` and each `collaborator` without performing any computational task. More details can be found [here](https://github.com/securefederatedai/openfl/tree/develop/openfl-workspace/no-op#overview).


[^1]: [OpenFL TaskRunner Overview](https://openfl.readthedocs.io/en/latest/about/features_index/taskrunner.html)
[^2]: [gRPC Overview](https://grpc.io/docs/what-is-grpc/core-concepts/)
[^3]: [mTLS Overview](https://www.cloudflare.com/learning/access-management/what-is-mutual-tls/)