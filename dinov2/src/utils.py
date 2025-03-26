import torch
import numpy as np
from transformers import Trainer, get_scheduler
from peft.config import PeftConfig
from peft import PeftModel


class MetricAccumulator:
    def __init__(self, metric, num_labels, ignore_index=None, reduce_labels=False):
        self.metric = metric
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.reduce_labels = reduce_labels
        self.accum = []

    def compute(self, p, compute_result):
        self.accum.append(
            self.metric.compute(
                predictions=p.predictions.argmax(1),
                references=p.label_ids.argmax(1),
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                reduce_labels=self.reduce_labels,
            )
        )
        if compute_result:
            out = {
                met: np.mean(np.stack([np.nan_to_num(m[met]) for m in self.accum]), axis=0).tolist()
                for met in self.accum[0].keys()
            }
            self.accum = []
            return out


class PeftModelForVit(PeftModel):

    def __init__(
        self,
        model: torch.nn.Module,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        **kwargs,
    ):
        super().__init__(model, peft_config, adapter_name, **kwargs)

    def forward(
        self,
        pixel_values=None,
        output_hidden_states=None,
        output_attentions=None,
        **kwargs,
    ):
        return self.base_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )


class FederatedTrainer(Trainer):
    def __init__(self, total_rounds=10, **kwargs):
        self.total_rounds = total_rounds
        super().__init__(**kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        train_dataloader = self.get_train_dataloader()
        max_steps = len(train_dataloader) * self.total_rounds
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(max_steps),
                num_training_steps=max_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler


def FedAvg(models, weights=None):
    new_model = models[0]
    state_dicts = models
    state_dict = new_model
    for key in state_dict:
        state_dict[key] = torch.from_numpy(
            np.asarray(
                np.average([state[key].numpy() for state in state_dicts], axis=0, weights=weights)
            )
        )
    return state_dict
