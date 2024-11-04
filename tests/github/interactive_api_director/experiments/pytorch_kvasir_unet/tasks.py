import tqdm
import torch
import numpy as np

from openfl.interface.interactive_api.experiment import TaskInterface
from tests.github.interactive_api_director.experiments.pytorch_kvasir_unet.layers import soft_dice_loss, soft_dice_coef


task_interface = TaskInterface()


def function_defined_in_notebook(some_parameter):
    print('I will cause problems')
    print(f'Also I accept a parameter and it is {some_parameter}')


# We do not actually need to register additional kwargs, Just serialize them
@task_interface.add_kwargs(**{'some_parameter': 42})
@task_interface.register_fl_task(model='unet_model', data_loader='train_loader',
                                 device='device', optimizer='optimizer')
def train(unet_model, train_loader, optimizer, device, loss_fn=soft_dice_loss, some_parameter=None):
    if not torch.cuda.is_available():
        device = 'cpu'

    function_defined_in_notebook(some_parameter)

    train_loader = tqdm.tqdm(train_loader, desc="train")

    unet_model.train()
    unet_model.to(device)

    losses = []

    for data, target in train_loader:
        data, target = torch.tensor(data).to(device), torch.tensor(
            target).to(device, dtype=torch.float32)
        optimizer.zero_grad()
        output = unet_model(data)
        loss = loss_fn(output=output, target=target)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())

    return {'train_loss': np.mean(losses), }


@task_interface.register_fl_task(model='unet_model', data_loader='val_loader', device='device')
def validate(unet_model, val_loader, device):
    unet_model.eval()
    unet_model.to(device)

    val_loader = tqdm.tqdm(val_loader, desc="validate")

    val_score = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            samples = target.shape[0]
            total_samples += samples
            data, target = torch.tensor(data).to(device), \
                torch.tensor(target).to(device, dtype=torch.int64)
            output = unet_model(data)
            val = soft_dice_coef(output, target)
            val_score += val.sum().cpu().numpy()

    return {'dice_coef': val_score / total_samples, }
