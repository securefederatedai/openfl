
# %%
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

mnist_train = torchvision.datasets.MNIST('files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

mnist_test = torchvision.datasets.MNIST('files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

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
        return F.log_softmax(x)
    
def inference(network,test_loader):
    print('1')
    if torch.cuda.is_available():
        network = network.to('cuda:0')
    network.eval()
    test_loss = 0
    correct = 0
    print('2')
    with torch.no_grad():
      for data, target in test_loader:
        if torch.cuda.is_available():
          data = data.to('cuda:0')
          target = target.to('cuda:0')
        output = network(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    print('3')
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
    accuracy = float(correct / len(test_loader.dataset))
    return accuracy
#%%
from copy import deepcopy

from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator


def FedAvg(models, weights=None):
    models = [model.to('cpu') for model in models]
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():
        state_dict[key] = torch.from_numpy(np.average([state[key].numpy() for state in state_dicts],
                                                      axis=0, 
                                                      weights=weights))
    new_model.load_state_dict(state_dict)
    return new_model



# %% [markdown]
# Now we come to the updated flow definition.

# %%
class CollaboratorGPUFlow(FLSpec):

    def __init__(self, model = None, optimizer = None, rounds=3, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            self.optimizer = optimizer
        else:
            self.model = Net()
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                   momentum=momentum)
        self.rounds = rounds

    @aggregator
    def start(self):
        print(f'Performing initialization for model')
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.current_round = 0
        self.next(self.aggregated_model_validation,foreach='collaborators',exclude=['private'])

    @collaborator
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input}')
        self.agg_validation_score = inference(self.model,self.test_loader)
        print(f'{self.input} value of {self.agg_validation_score}')
        self.next(self.train)

    @collaborator
    def train(self):
        """
        Train the model.
        """
        self.model.train()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                              momentum=momentum)
        train_losses = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if torch.cuda.is_available():
                data = data.to("cuda:0")
                target = target.to("cuda:0")
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch {}: 1 [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(self.input,
                    batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
                self.loss = loss.item()
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_validation_score = inference(self.model,self.test_loader)
        print(f'Doing local model validation for collaborator {self.input}: {self.local_validation_score}')
        self.next(self.join)

    @aggregator
    def join(self,inputs):
        self.average_loss = sum(input.loss for input in inputs)/len(inputs)
        self.aggregated_model_accuracy = sum(input.agg_validation_score for input in inputs)/len(inputs)
        self.local_model_accuracy = sum(input.local_validation_score for input in inputs)/len(inputs)
        print(f'Average aggregated model validation values = {self.aggregated_model_accuracy}')
        print(f'Average training loss = {self.average_loss}')
        print(f'Average local model validation values = {self.local_model_accuracy}')
        self.model = FedAvg([input.model for input in inputs])
        self.optimizer = [input.optimizer for input in inputs][0]
        self.current_round += 1
        if self.current_round < self.rounds:
            self.next(self.aggregated_model_validation, foreach='collaborators', exclude=['private'])
        else:
            self.next(self.end)
        
    @aggregator
    def end(self):
        print(f'This is the end of the flow')

# %% [markdown]
# In this step we define entities necessary to run the flow and create a function which returns dataset as private attributes of collaborator. As described in [quickstart](https://github.com/securefederatedai/openfl/blob/develop/openfl-tutorials/experimental/Workflow_Interface_101_MNIST.ipynb) we define entities necessary for the flow.
# 
# To request GPU(s) with ray-backend, we specify `num_gpus=0.3` as the argument while instantiating Aggregator and Collaborator, this will reserve 0.3 GPU for each of the 2 collaborators and the aggregator and therefore require a dedicated GPU for the experiment. Tune this based on your use case, for example `num_gpus=0.4` for an experiment with 4 collaborators and the aggregator will require 2 dedicated GPUs. **NOTE:** Collaborator cannot span over multiple GPUs, for example `num_gpus=0.4` with 5 collaborators will require 3 dedicated GPUs. In this case collaborator 1 and 2 use GPU#1, collaborator 3 and 4 use GPU#2, and collaborator 5 uses GPU#3.

# %%
# Setup Aggregator private attributes via callable function
#aggregator = Aggregator(num_gpus=0.3)
aggregator = Aggregator()

collaborator_names = ['Portland', 'Seattle','col1','col2','col3']

def callable_to_initialize_collaborator_private_attributes(index, n_collaborators,
        train_dataset, test_dataset, batch_size_train):
    local_train = deepcopy(train_dataset)
    local_test = deepcopy(test_dataset)
    local_train.data = train_dataset.data[index::n_collaborators]
    local_train.targets = train_dataset.targets[index::n_collaborators]
    local_test.data = test_dataset.data[index::n_collaborators]
    local_test.targets = test_dataset.targets[index::n_collaborators]

    return {
        'train_loader': torch.utils.data.DataLoader(local_train,batch_size=batch_size_train, shuffle=True),
        'test_loader': torch.utils.data.DataLoader(local_test, batch_size=batch_size_train, shuffle=True)
    }

# Setup collaborators private attributes via callable function
collaborators = []
for idx, collaborator_name in enumerate(collaborator_names):
    collaborators.append(
        Collaborator(
            #name=collaborator_name, num_cpus=0, num_gpus=0.5,
            name=collaborator_name, num_cpus=0,
            private_attributes_callable=callable_to_initialize_collaborator_private_attributes,
            index=idx, n_collaborators=len(collaborator_names),
            train_dataset=mnist_train, test_dataset=mnist_test, batch_size_train=batch_size_train
        )
    )
    
# The following is equivalent to
# local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators, **backend='ray'**)
local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators, backend='ray_grouped', collaborators_per_group=3)
#local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators)
print(f'Local runtime collaborators = {local_runtime.collaborators}')

# %% [markdown]
# Now that we have our flow and runtime defined, let's run the experiment! 
# 
# (If you run this example on Google Colab with the GPU Runtime, you should see two task executing at a time.)

# %%
model = None
best_model = None
optimizer = None
flflow = CollaboratorGPUFlow(model, optimizer, checkpoint=True)
flflow.runtime = local_runtime
flflow.run()

# %% [markdown]
# Now that the flow has completed, let's get the final model and accuracy:

# %%
print(f'Sample of the final model weights: {flflow.model.state_dict()["conv1.weight"][0]}')

print(f'\nFinal aggregated model accuracy for {flflow.rounds} rounds of training: {flflow.aggregated_model_accuracy}')

# %% [markdown]
# Now that the flow is complete, let's dig into some of the information captured along the way

# %%
run_id = flflow._run_id

# %%
from metaflow import Metaflow, Flow, Task, Step

# %%
m = Metaflow()
list(m)

# %% [markdown]
# For existing users of Metaflow, you'll notice this is the same way you would examine a flow after completion. Let's look at the latest run that generated some results:

# %%
f = Flow('CollaboratorGPUFlow').latest_run

# %%
f

# %% [markdown]
# And its list of steps

# %%
list(f)

# %% [markdown]
# This matches the list of steps executed in the flow, so far so good...

# %%
s = Step(f'CollaboratorGPUFlow/{run_id}/train')

# %%
s

# %%
list(s)

# %% [markdown]
# Now we see **6** steps: **2** collaborators each performed **3** rounds of model training  

# %%
