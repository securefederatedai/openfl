import numpy as np
import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, Union
from peft import PeftModel
from peft.config import PeftConfig
from transformers import Trainer, get_scheduler
from evaluate import Metric, EvaluationModule
import evaluate
from transformers.trainer_utils import PredictionOutput


class Metric_Computer:
    def __init__(
        self,
        num_labels: int,
        ignore_index: Optional[int] = None,
        reduce_labels: bool = False,
        task_type: str = "segmentation",
        batched_compute: bool = False,
    ) -> None:
        if task_type not in ["classification", "segmentation"]:
            raise ValueError(
                f"Unsupported task type: {task_type}. Supported types are 'classification' and 'segmentation'."
            )
        if task_type == "classification":
            self.metric: EvaluationModule = evaluate.load("accuracy")
        else:
            self.metric: EvaluationModule = evaluate.load("mean_iou")

        if batched_compute:
            self.accumulator: List[Dict[str, Any]] = []
            self.keys = set()

        self.batched_compute = batched_compute
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.reduce_labels = reduce_labels
        self.task_type = task_type

    def compute(
        self, p: PredictionOutput, compute_result: bool = True
    ) -> Optional[Dict[str, Union[float, List[float]]]]:
        """
        Computes the metric based on the predictions and labels from the PredictionOutput.

        Args:
            p (PredictionOutput): The prediction output containing predictions and label_ids.
            compute_result (bool): Whether to compute the final result or just update the accumulator.

        Returns:
            Optional[Dict[str, Union[float, List[float]]]]: The computed metrics if compute_result is True, otherwise None.
        """
        return (
            self._compute(p, compute_result)
            if not self.batched_compute
            else self._batched_compute(p, compute_result)
        )

    def _compute(
        self, p: PredictionOutput, compute_result: bool
    ) -> Optional[Dict[str, Union[float, List[float]]]]:
        if self.task_type == "classification":
            out: Dict = self.metric.compute(
                predictions=p.predictions.argmax(1),
                references=p.label_ids.argmax(1),
            )
        else:
            out: Dict = self.metric.compute(
                predictions=p.predictions.argmax(1),
                references=p.label_ids.argmax(1),
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                reduce_labels=self.reduce_labels,
            )
        return out

    def _batched_compute(
        self, p: PredictionOutput, compute_result: bool
    ) -> Optional[Dict[str, Union[float, List[float]]]]:
        if self.task_type == "classification":
            out: Dict = self.metric.compute(
                predictions=p.predictions.argmax(1),
                references=p.label_ids.argmax(1),
            )
        else:
            out: Dict = self.metric.compute(
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
        **kwargs: Any,
    ):
        return self.base_model(
            **kwargs,
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
    base_model: torch.nn.Module,
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    grads: bool = False,
    lr: float = 1e-3,
    peft_prefix: str = "default.",
    modules_to_save_prefix: str = "modules_to_save.",
) -> Dict[str, torch.Tensor]:
    state_dict = base_model.state_dict()
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            param.data = torch.from_numpy(
                np.asarray(
                    np.average(
                        [
                            state[
                                name.replace(peft_prefix, "").replace(modules_to_save_prefix, "")
                            ].numpy()
                            for state in state_dicts
                        ],
                        axis=0,
                        weights=weights,
                    )
                )
            )
            if grads:
                param.grad = (param.data - state_dict[name]) / lr


def get_param_counts_log(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    log_str = (
        f"Parameter Counts:\n"
        f"  Total:     {total:,}\n"
        f"  Trainable: {trainable:,}\n"
        f"  Frozen:    {frozen:,}"
    )
    return log_str