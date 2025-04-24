# InfoSec Overview

## Purpose
This document provides the information needed when evaluating OpenFL for real world deployment in highly sensitive environments. The target audience is InfoSec reviewers who need detailed information about code contents, communication traffic, and potential exploit vectors.

## Network Connectivity Overview
OpenFL federations use a hub-and-spoke topology between _collaborator_ clients that generate model parameter updates from their data and the _aggregator_ server that combines their training updates into new models [[ref](https://openfl.readthedocs.io/en/latest/about/features_index/taskrunner.html)]. Key details about this functionality are:
* Connections are made using request/response gRPC connections [[ref](https://grpc.io/docs/what-is-grpc/core-concepts/)].
* The _aggregator_ listens for connections on a single port (usually decided by the experiment admin), and is explicitly defined in the FL plan (f.e. `50051`), so all _collaborators_ must be able to send outgoing traffic to this port.
* All connections are initiated by the _collaborator_, i.e., a `pull` architecture [[ref](https://karlchris.github.io/data-engineering/data-ingestion/push-pull/#pull)].
* The _collaborator_ does not open any listening sockets.
* Connections are secured using mutually-authenticated TLS [[ref](https://www.cloudflare.com/learning/access-management/what-is-mutual-tls/)].
* Each request response pair is done on a new TLS connection.
* The PKI for federations can be created using the [OpenFL CLI](https://openfl.readthedocs.io/en/latest/about/features_index/taskrunner.html#step-2-configure-the-federation). OpenFL internally leverages Python's cryptography module. The organization hosting the _aggregator_ usually acts as the Certificate Authority (CA) and verifies each identity before signing.
* Currently, the _collaborator_ polls the _aggregator_ at a fixed interval. We have had a request to enable client-side configuration of this interval and hope to support that feature soon.
* Connection timeouts are set to gRPC defaults.
* If the _aggregator_ is not available, the _collaborator_ will retry connections indefinitely. This is currently useful so that we can take the aggregator down for bugfixes without _collaborator_ processes exiting.

## Overview of Contents of Network Messages
Network messages are well defined protobufs which can be found in the following files:
- [aggregator.proto](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/aggregator.proto)
- [base.proto](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/base.proto)

Key points about the network messages/protocol:
* No executable code is ever sent to the collaborator. All code to be executed is contained within the OpenFL package and the custom FL workspace. The code, along with the FL plan file that specifies the classes and initial parameters to be used, is available for review prior to the FL plans execution. This ensures that all potential operations are understood before they take place.
* The _collaborator_ typically requests the FL tasks to execute from the aggregator via the `GetTasksRequest` message [[ref](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/aggregator.proto#L34)]
* The _aggregator_ reads the FL plan and returns a `GetTasksResponse` [[ref](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/aggregator.proto#L45)] which includes metadata (`Tasks`) [[ref](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/aggregator.proto#L38)] about the Python functions to be invoked by the collaborator (the code being installed locally as part of a pre-distributed workspace bundle)
* The _collaborator_ then uses its TaskRunner framework to execute the FL tasks on the locally available data, producing output tensors such as model weights or metrics
* During task execution, the _collaborator_ may additionally request tensors from the aggregator via the `GetAggregatedTensor` RPC method [[ref](https://openfl.readthedocs.io/en/latest/reference/_autosummary/openfl.transport.grpc.aggregator_server.AggregatorGRPCServer.html#openfl.transport.grpc.aggregator_server.AggregatorGRPCServer.GetAggregatedTensor)]
* Upon task completion, the _collaborator_ transmits the results by emitting a `SendLocalTaskResults` call [[ref](https://openfl.readthedocs.io/en/latest/reference/_autosummary/openfl.transport.grpc.aggregator_server.AggregatorGRPCServer.html#openfl.transport.grpc.aggregator_server.AggregatorGRPCServer.SendLocalTaskResults)] which contains `NamedTensor` [[ref](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/base.proto#L11)] objects that encode model weight updates or ML metrics such as loss or accuracy (among others).

## Testing a Collaborator
There is a "no-op" workspace template in OpenFL (available in versions `>=1.9`) which can be used to test the network connection between the _aggregator_ and each _collaborator_ without performing any computational task. More details can be found [here](https://github.com/securefederatedai/openfl/tree/develop/openfl-workspace/no-op#overview).
