# Federated Runtime: CrowdGuard

This work is based on [CrowdGuard demo code](https://github.com/securefederatedai/openfl/blob/develop/openfl-tutorials/experimental/workflow/CrowdGuard), the [FederatedRuntime](https://github.com/securefederatedai/openfl/blob/develop/openfl-tutorials/experimental/workflow/FederatedRuntime) and the [SecAgg](https://github.com/securefederatedai/openfl/blob/develop/openfl-tutorials/experimental/workflow/SecAgg/workspace/MNIST_SecAgg.ipynb) examples given in this repo. It has been adapted to demonstrate CrowdGuard in the new `FederatedRuntime`.

CrowdGuard is a defense mechanism against backdoor attacks in federated learning environments. This example contains four envoys named Amsterdam, Bangalore, Chandler and Detroit. Detroit is the malicious actor that submits poisoned model updates. CrowdGuard is able to detect and filter out poisoned model updates.

## How to run it

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

Now that your director and envoy terminals are set up, run the Jupyter Notebook in the workspace folder. I am using the Jupyter Extension for VS Code.
