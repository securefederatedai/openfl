### Purpose of this document
This document is intended to help InfoSec analysis processes of the collaborators involved in the initial stages of OpenFL federations by summarizing key points of the code functionality.

#### Network Connectivity Overview
OpenFL federations use a hub-and-spoke topology between _collaborator_ clients that generate model parameter updates from their data and the _aggregator_ server that combines their training updates into new models. Key details about this functionality are:
* Connections are made using request/response gRPC connections.
* The _aggregator_ listens for connections on a single port, explicitly defined in the FL plan (f.e. 50051), so all _collaborators_ must be able to send outgoing traffic to this port.
* All connections are initiated by the _collaborator_.
* The _collaborator_ does not open any listening sockets.
* Connections are secured using mutually-authenticated TLS.
* Each request response pair is done on a new TLS connection.
* The PKI for OpenFL federations can be created using openssl tools, with the organization hosting the _aggregator_ usually acting as the Certificate Authority (CA) that verifies each identity before signing.
* Currently, the _collaborator_ polls the _aggregator_ at a fixed interval. We have had a request to enable client-side configuration of this interval and hope to support that feature soon.
* Connection timeouts are set to gRPC defaults.
* If the _aggregator_ is not available, the _collaborator_ will retry connections indefinitely. This is currently useful so that we can take the aggregator down for bugfixes without _collaborator_ processes exiting.

#### Overview of Contents of Network Messages
Network messages are well defined protobufs which can be found in [aggregator.proto](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/aggregator.proto) and [base.proto](https://github.com/securefederatedai/openfl/blob/develop/openfl/protocols/base.proto).
Key points about the network messages/protocol:
* No private data is ever sent to the _aggregator_
* No executable code is ever sent to the _collaborator_. All algorithms come with the OpenFL package and the custom FL workspace.
* The _collaborator_ typically requests the FL tasks to execute from the aggregator via the `GetTasksRequest` message
* The _aggregator_ reads the FL plan and returns a `GetTasksResponse` which includes metadata (`Tasks`) about the Python functions to be invoked by the collaborator (the code being installed locally as part of a pre-distributed workspace bundle)
* The _collaborator_ then uses its TaskRunner framework to execute the FL tasks on the locally available data, producing output tensors such as model weights or metrics
* During task execution, the _collaborator_ may additionally request tensors from the aggregator via the `GetAggregatedTensor` RPC method
* Upon task completion, the _collaborator_ transmits the results by emitting a `SendLocalTaskResults` call which contains `NamedTensor` objects that encode model weight updates or ML metrics such as loss or accuracy (among others).

#### Testing a Collaborator
In order to test a _collaborator_, the _aggregator_ can be set to not aggregate updates from specific _collaborators_ so that they can test functionality without impacting the running model training/validation. We hope this enables InfoSec runtime analysis against the live _aggregator_ without the need to access any private data on the _collaborator_ machine running the test. To do this, please coordinate with the OpenFL aggregator admins.
