## Overview
The "no-op" workspace template serves as a tool for testing the PKI setup and connectivity among the participants in a federation. It can be particularly useful in the initial phases of establishing a Federated Learning environment, as it allows for the separation of the infrastructure setup from the actual FL plan and task definitions. Moreover, a "no-op" workspace offers the benefit of requiring minimal code review, making it easy for collaborators to agree to run it locally. Additionally:
* it does not access any data;
* minimizes compute resource utilization for initial tests;
* only the `openfl` package is required (no 3rd party ML frameworks need to be installed).

## Configuring a no-op workspace
Once openfl has been [installed](https://openfl.readthedocs.io/en/latest/installation.html), a no-op workspace can be instantiated like any other OpenFL workspace by running:
```bash
fx workspace create --prefix ./no-op --template no-op
```

To configure a local connectivity experiment, follow the [quickstart guide](https://openfl.readthedocs.io/en/latest/tutorials/taskrunner.html). For a distributed infrastructure test, follow the [Task Runner API documentation](https://openfl.readthedocs.io/en/latest/about/features_index/taskrunner.html#bare-metal-approach).

**NB!** When configuring the collaborators, omit the `-d` or `--data_path` parameter, as the no-op workspace does not require any data:
```bash
fx collaborator create -n collaborator-name
```

## Connectivity test
Once the workspace has been configured and distributed to all participants:
1. Start the aggregator:
```bash
fx aggregator start
```

2. From each collaborator machine, ping the aggregator:
```bash
fx collaborator ping -n collaborator-name
```

If network connectivity is available and the PKI setup is correct, you should see a confirmation message resembling the following:
```bash
The Aggregator is reachable at agg-host:52019
TLS connection established.
```