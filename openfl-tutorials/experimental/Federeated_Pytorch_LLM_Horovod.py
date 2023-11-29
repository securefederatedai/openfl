
import openfl.native as fx
import sys

sys.path.append("openfl/openfl-workspace/torch_llm")
from src.pt_model import LLMTaskRunner
from src.ptglue_inmemory import GlueMrpcFederatedDataLoader
import openfl.interface.workspace as workspace
import os
import subprocess

WORKSPACE_PREFIX = os.path.join(os.path.expanduser("~"), ".local", "workspace")

# Run openfl-tutorials/experimental/setup_env.shsetup_env.sh in your venv to setup horovod dependancies
# set up the venv in each node
# make dir ~/.local/workspace in each node
# horovod requires password less ssh login, you can learn how to set it up here: http://www.linuxproblem.org/art_9.html

# You should set the following ENVIROMENTAL VARIABLES for horovod
#OPENFL_HOROVOD_DEMO_NP=STR with number of processes to run eg. "4"
#OPENFL_HOROVOD_DEMO_NICS=STR with the common network interface name to use with all nodes eg. "en01"
#OPENFL_HOROVOD_DEMO_LOCALHOSTIP=STR with the IP address of the local node eg. "ip1"
#OPENFL_HOROVOD_DEMO_HOSTS=STR with the IP address of the each node and number of slots eg. "ip1:2,ip2,2"

NP = os.environ.get('OPENFL_HOROVOD_DEMO_NP','4')
NETWORK_INTERFACES = os.environ.get('OPENFL_HOROVOD_DEMO_NICS','localhost')
LOCAL_HOST = os.environ.get('OPENFL_HOROVOD_DEMO_LOCALHOSTIP','localhost')
HOSTS = os.environ.get('OPENFL_HOROVOD_DEMO_HOSTS','localhost:4')

print('NP:', NP)
print('NETWORK_INTERFACES:', NETWORK_INTERFACES)
print('LOCAL_HOST:', LOCAL_HOST)
print('HOSTS:', HOSTS)

def propogate_workspace():
    remote_hosts = [
        i.split(":")[0] for i in HOSTS.split(",") if i.split(":")[0] != LOCAL_HOST
    ]
    for rem_host in remote_hosts:
        result = subprocess.run(
            [
                "scp",
                "-r",
                WORKSPACE_PREFIX,
                rem_host
                + ":" +
                WORKSPACE_PREFIX.replace('workspace',''),
            ],
            capture_output=True,
        )
        print([
                "scp",
                "-r",
                WORKSPACE_PREFIX,
                rem_host
                + ":" +
                WORKSPACE_PREFIX,
            ])
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

def main():
    print(WORKSPACE_PREFIX)
    log_level = "INFO"
    log_file = None
    workspace.create(WORKSPACE_PREFIX, "torch_llm")
    os.chdir(WORKSPACE_PREFIX)
    sys.path.append(WORKSPACE_PREFIX)
    propogate_workspace()
    fx.setup_logging(level=log_level, log_file=log_file)
    num_collaborators = 1

    collaborator_models = [
        LLMTaskRunner(
            data_loader=GlueMrpcFederatedDataLoader(
                data_slice, 32, collaborator_count=num_collaborators
            )
        )
        for data_slice in range(num_collaborators)
    ]
    collaborators = {
        "one": collaborator_models[0],
    }

    # Collaborator one's data
    for i, model in enumerate(collaborator_models):
        print(
            f"Collaborator {i}'s training data size: {len(model.data_loader.train_set)}"
        )
        print(
            f"Collaborator {i}'s validation data size: {len(model.data_loader.valid_set)}\n"
        )
    final_fl_model = fx.run_experiment(
        collaborators,
        {"aggregator.settings.rounds_to_train": 5, "tasks.train.kwargs.epochs": 1},
    )


if __name__ == "__main__":
    main()
