import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

mnist_train = torchvision.datasets.MNIST(
    "files/",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_test = torchvision.datasets.MNIST(
    "files/",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def inference(network, test_loader):
    if torch.cuda.is_available():
        network = network.to("cuda")
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data = data.to("cuda")
                target = target.to("cuda")
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    accuracy = float(correct / len(test_loader.dataset))
    return accuracy


from copy import deepcopy

from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator


def FedAvg(models, weights=None):
    models = [model.to("cpu") for model in models]
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():
        state_dict[key] = torch.from_numpy(
            np.average(
                [state[key].numpy() for state in state_dicts], axis=0, weights=weights
            )
        )
    new_model.load_state_dict(state_dict)
    return new_model

frac_gpu = 0.9
class FederatedFlow(FLSpec):
    def __init__(self, model=None, optimizer=None, rounds=3, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            self.optimizer = optimizer
        else:
            self.model = Net()
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=learning_rate, momentum=momentum
            )
        self.rounds = rounds

    @aggregator
    def start(self):
        print(f"Performing initialization for model")
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.current_round = 0
        self.next(
            self.aggregated_model_validation,
            foreach="collaborators",
            exclude=["private"],
        )

    @collaborator(num_gpus=frac_gpu)
    def aggregated_model_validation(self):
        print(f"Performing aggregated model validation for collaborator {self.input}")
        self.agg_validation_score = inference(self.model, self.test_loader)
        print(f"{self.input} value of {self.agg_validation_score}")
        self.next(self.train)

    @collaborator(num_gpus=frac_gpu)
    def train(self):
        if torch.cuda.is_available():
            print('train model on gpu')
            self.model = self.model.to("cuda")
        else:
            print('train model on cpu')
        self.model.train()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum
        )
        train_losses = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if torch.cuda.is_available():
                data = data.to("cuda")
                target = target.to("cuda")
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch {}: 1 [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        self.input,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item(),
                    )
                )
                self.loss = loss.item()
                torch.save(self.model.state_dict(), "model.pth")
                torch.save(self.optimizer.state_dict(), "optimizer.pth")
        self.training_completed = True
        self.next(self.local_model_validation)

    @collaborator(num_gpus=frac_gpu)
    def local_model_validation(self):
        self.local_validation_score = inference(self.model, self.test_loader)
        print(
            f"Doing local model validation for collaborator {self.input}: {self.local_validation_score}"
        )
        self.next(self.join, exclude=["training_completed"])

    @aggregator
    def join(self, inputs):
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs
        ) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs
        ) / len(inputs)
        print(
            f"Average aggregated model validation values = {self.aggregated_model_accuracy}"
        )
        print(f"Average training loss = {self.average_loss}")
        print(f"Average local model validation values = {self.local_model_accuracy}")
        self.model = FedAvg([input.model for input in inputs])
        self.optimizer = [input.optimizer for input in inputs][0]
        self.current_round += 1
        if self.current_round < self.rounds:
            self.next(
                self.aggregated_model_validation,
                foreach="collaborators",
                exclude=["private"],
            )
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(f"This is the end of the flow")

if torch.cuda.is_available():
    print('model will run on gpu')
    print(f'available devices:',torch.cuda.device_count())
else:
    print('model will run on cpu')
    
aggregator = Aggregator()
aggregator.private_attributes = {}

# Setup collaborators with private attributes
collaborator_names = ["Portland", "Seattle", "Chandler", "Bangalore"]
collaborators = [Collaborator(name=name, ) for name in collaborator_names]
for idx, col in enumerate(collaborators):
    local_train = deepcopy(mnist_train)
    local_test = deepcopy(mnist_test)
    local_train.data = mnist_train.data[idx :: len(collaborators)]
    local_train.targets = mnist_train.targets[idx :: len(collaborators)]
    local_test.data = mnist_test.data[idx :: len(collaborators)]
    local_test.targets = mnist_test.targets[idx :: len(collaborators)]
    col.private_attributes = {
        "train_loader": torch.utils.data.DataLoader(
            local_train, batch_size=batch_size_train, shuffle=True
        ),
        "test_loader": torch.utils.data.DataLoader(
            local_test, batch_size=batch_size_train, shuffle=True
        ),
    }

local_runtime = LocalRuntime(
    aggregator=aggregator, collaborators=collaborators, backend="ray"
)
print(f"Local runtime collaborators = {local_runtime.collaborators}")

model = None
best_model = None
optimizer = None
flflow = FederatedFlow(model, optimizer)
flflow.runtime = local_runtime
flflow.run()

print(
    f'Sample of the final model weights: {flflow.model.state_dict()["conv1.weight"][0]}'
)

print(
    f"\nFinal aggregated model accuracy for {flflow.rounds} rounds of training: {flflow.aggregated_model_accuracy}"
)

flflow2 = FederatedFlow(model=flflow.model, optimizer=flflow.optimizer, checkpoint=True)
flflow2.runtime = local_runtime
flflow2.run()

run_id = flflow2._run_id

# %%
import metaflow

# %%
from metaflow import Metaflow, Flow, Task, Step

# %%
m = Metaflow()
list(m)

# %% [markdown]
# For existing users of Metaflow, you'll notice this is the same way you would examine a flow after completion. Let's look at the latest run that generated some results:

# %%
f = Flow("FederatedFlow").latest_run

# %%
f

# %% [markdown]
# And its list of steps

# %%
list(f)

# %% [markdown]
# This matches the list of steps executed in the flow, so far so good...

# %%
s = Step(f"FederatedFlow/{run_id}/train")

# %%
s

# %%
list(s)

# %% [markdown]
# Now we see **12** steps: **4** collaborators each performed **3** rounds of model training

# %%
t = Task(f"FederatedFlow/{run_id}/train/9")

# %%
t

# %% [markdown]
# Now let's look at the data artifacts this task generated

# %%
t.data

# %%
t.data.input

# %% [markdown]
# Now let's look at its log output (stdout)

# %%
print(t.stdout)

# %% [markdown]
# And any error logs? (stderr)

# %%
print(t.stderr)

# %% [markdown]
# # Congratulations!
# Now that you've completed your first workflow interface quickstart notebook, see some of the more advanced things you can do in our [other tutorials](broken_link), including:
#
# - Using the LocalRuntime Ray Backend for dedicated GPU access
# - Vertical Federated Learning
# - Model Watermarking
# - Differential Privacy
# - And More!
