import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def cross_entropy(output, target):
    return F.binary_cross_entropy_with_logits(input=output, target=target)

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def init_network(self,
                     device='cpu',
                     print_model=True,
                     num_classes=10,
                     channel_in=1,
                     **kwargs):
        """Create the network (model)."""
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=20, \
            kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1, 1)
        self.conv3 = nn.Conv2d(50, 500, 1)
        self.fc = nn.Linear(500, num_classes)

        if print_model:
            print(self)
        self.to(device)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Data input to the model for the forward pass
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



model_inst = Model()
optimizer_inst = optim.Adam(model_inst.parameters(), lr=1e-4)


def train_epoch(model, train_loader, device, optimizer, loss_fn):
    model.train()
    model.to(device)

    losses = []

    for data, target in train_loader:
        data, target = torch.tensor(data).to(device), \
            torch.tensor(target).to(device, dtype=torch.float32)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output=output, target=target)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())

    return {'train_loss': np.mean(losses),}


def validation(model, val_loader, device):
    model.eval()
    model.to(device)

    val_score = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            samples = target.shape[0]
            total_samples += samples
            data, target = torch.tensor(data).to(device), \
                torch.tensor(target).to(device, dtype=torch.int64)
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            target_categorical = target.argmax(dim=1, keepdim=True)
            val_score += pred.eq(target_categorical).sum().cpu().numpy()

    return {'accuracy': val_score / total_samples,}