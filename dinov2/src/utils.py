import numpy as np
import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, Union
from peft import PeftModel
from peft.config import PeftConfig
from transformers import Trainer, get_scheduler
from evaluate import Metric
from transformers.trainer_utils import PredictionOutput


class MetricAccumulator:
    def __init__(
        self,
        metric: Metric,
        num_labels: int,
        ignore_index: Optional[int] = None,
        reduce_labels: bool = False,
    ) -> None:
        self.metric = metric
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.reduce_labels = reduce_labels
        self.accumulator: List[Dict[str, Any]] = []
        self.keys = set()

    def compute(
        self, p: PredictionOutput, compute_result: bool
    ) -> Optional[Dict[str, Union[float, List[float]]]]:
        out = self.metric.compute(
            predictions=p.predictions.argmax(1),
            references=p.label_ids.argmax(1),
            num_labels=self.num_labels,
            ignore_index=self.ignore_index,
            reduce_labels=self.reduce_labels,
        )
        self.keys.update(out.keys())
        self.accumulator.append(out)
        if compute_result:
            out = {
                metric_name: np.mean(
                    np.stack([np.nan_to_num(metric[metric_name]) for metric in self.accumulator]),
                    axis=0,
                ).tolist()
                for metric_name in self.keys
            }
            self.accumulator = []
            self.keys = set()
            return out
        return None


class PeftModelForVit(PeftModel):
    def __init__(
        self,
        model: torch.nn.Module,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.base_model(
            pixel_values=pixel_values,
            head_mask=head_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FederatedTrainer(Trainer):
    def __init__(self, total_rounds: int = 10, **kwargs: Any) -> None:
        self.total_rounds = total_rounds
        super().__init__(**kwargs)

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Any:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before
          this method is called or
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


def FedAvg(
    models: List[Dict[str, torch.Tensor]], weights: Optional[List[float]] = None
) -> Dict[str, torch.Tensor]:
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
