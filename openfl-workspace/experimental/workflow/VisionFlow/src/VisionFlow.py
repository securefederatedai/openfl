import logging

import evaluate
import torch
from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from transformers import TrainingArguments
from datasets import Dataset

from src.utils import FedAvg, FederatedTrainer, Metric_Computer
from src.modeling.VisionModel import VisionModel, TASK_DICT
from src.dataset.img_classification import image_classification_collate_fn, DEFAULT_LABEL_FEATURE
from src.dataset.img_pretraining import image_pretraining_collate_fn


WRITER = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLATE_FN = {
    "classification": image_classification_collate_fn,
    "pretraining": image_pretraining_collate_fn,
}


class VisionFlow(FLSpec):

    def __init__(
        self,
        model=None,  # can be None to use default VisionModel or use your own model
        rounds=3,  # rounds to train
        task_type="classification",  # task type
        global_validation_dataset: Dataset = None,  #
        training_args: TrainingArguments | dict = None,
        move_to_cpu_end_of_training=False,
        writer=None,
        pretrained_model_path=None,
        **kwargs,
    ):
        logger.info("Initializing VisionFlow...")

        if task_type not in TASK_DICT:
            raise ValueError(
                f"Unsupported task type: {task_type}. Supported types are 'classification' and 'segmentation'."
            )
        model_config_kwargs = kwargs.pop("model_config_kwargs", {})
        peft_config = kwargs.pop("peft_config", None)
        use_peft = kwargs.pop("use_peft", False)
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
        else:
            self.model = VisionModel(
                task_type=task_type,
                use_peft=use_peft,
                peft_config=peft_config,
                pretrained_model_path=pretrained_model_path,
                **model_config_kwargs,
            )

        self.collate_fn = COLLATE_FN[task_type]
        self.rounds = rounds
        self.global_validation_dataset = global_validation_dataset

        training_args = self.set_training_args(training_args)
        if training_args.label_names is None:
            if task_type in ["classification", "segmentation"]:
                training_args.label_names = [DEFAULT_LABEL_FEATURE]
        self.training_args = training_args
        self.move_to_cpu_at_end_of_training = move_to_cpu_end_of_training
        self.task_type = task_type

        self.training_dicts = {}
        self.validation_dicts = {}

        self.lr_scheduler = None

        if self.move_to_cpu_at_end_of_training:
            self.model.to("cpu")
            torch.cuda.empty_cache()
        logger.info("VisionFlow initialized with %d rounds.", self.rounds)

        if writer is not None:
            global WRITER
            WRITER = writer

    def set_training_args(self, training_args):
        default_training_args_dict = {
            "output_dir": "./output",
            "num_train_epochs": 1,
            "weight_decay": 0.01,
        }

        if isinstance(training_args, dict):
            default_training_args_dict.update(training_args)
            training_args = TrainingArguments(**default_training_args_dict)
        elif isinstance(training_args, TrainingArguments):
            training_args = training_args
        return training_args

    @aggregator
    def start(self):
        logger.info("Starting federated learning workflow...")
        self.collaborators = self.runtime.collaborators
        self.current_round = 0
        logger.info("Workflow started.")
        self.next(self.aggregated_model_validation, foreach="collaborators")

    @collaborator
    def aggregated_model_validation(self):
        logger.info("Performing aggregated model validation on collaborator: %s", self.input)
        if torch.cuda.is_available():
            self.model.to("cuda")

        if self.task_type in ["classification", "segmentation"]:
            metric_computer = self.get_metric_computer()
        else:
            metric_computer = None

        self.train_dataset = self.train_dataset.shuffle()

        self.trainer = FederatedTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.collate_fn,
            compute_metrics=metric_computer.compute if metric_computer else None,
            optimizers=(None, self.lr_scheduler),
            total_rounds=self.rounds,
        )
        self.data_count = len(self.train_dataset)

        eval_dict = self.trainer.evaluate()
        self.validation_dicts["agg_validation_dict"] = eval_dict
        logger.info("Aggregated model validation completed on collaborator: %s", self.input)
        self.next(self.train)

    def get_metric_computer(self):
        metric_accumulator = None
        if self.task_type in ["classification", "segmentation"]:
            metric_accumulator = Metric_Computer(
                self.model.num_labels,
                task_type=self.task_type,
                batched_compute=self.training_args.batch_eval_metrics,
            )
            return metric_accumulator
        else:

            class dummy_metric:
                def compute(self, eval_preds):
                    return {"dummy_metric": 0.0}

            return dummy_metric()

    @collaborator
    def train(self):
        logger.info("Starting training on collaborator: %s", self.input)
        training_dict = self.trainer.train()
        self.training_completed = True
        self.training_dicts["train_dict"] = training_dict
        logger.info("Training completed on collaborator: %s", self.input)
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        logger.info("Performing local model validation on collaborator: %s", self.input)
        self.trainer.evaluate()
        eval_dict = self.trainer.evaluate()
        self.validation_dicts["local_validation_dict"] = eval_dict

        if self.move_to_cpu_at_end_of_training:
            self.model.to("cpu")
            torch.cuda.empty_cache()

        self.lr_scheduler = self.trainer.lr_scheduler
        self.weights = self.model.get_weights()
        logger.info("Local model validation completed on collaborator: %s", self.input)
        self.next(self.join, exclude=["training_completed", "trainer"])

    @aggregator
    def join(self, inputs):
        logger.info("Joining collaborator inputs...")
        # Log global metrics using PyTorch
        self.lr_scheduler = inputs[0].lr_scheduler
        if self.lr_scheduler.get_last_lr()[0]:
            self.last_lr = self.lr_scheduler.get_last_lr()[0]
        total_data_count = sum(input.data_count for input in inputs)
        data_weights = [input.data_count / total_data_count for input in inputs]
        FedAvg(
            self.model.model,
            [input.weights for input in inputs],
            data_weights,
            grads=True,
            lr=self.last_lr,
        )

        self.aggregate_evaluation_metrics = {}
        if self.global_validation_dataset is not None:
            self.perform_global_evaluation()

        self.aggregate_training_and_validation_metrics(inputs)

        self.write_to_tensorboard()

        model_path = f"{self.training_args.output_dir}/round_{self.current_round}.pt"
        torch.save(self.model.state_dict(), model_path)

        self.current_round += 1
        logger.info("Collaborator inputs joined. Metrics logged.")
        if self.current_round < self.rounds:
            self.next(
                self.aggregated_model_validation,
                foreach="collaborators",
            )
        else:
            self.next(self.end)

    def write_to_tensorboard(self):
        if WRITER is not None:
            for key, value in self.aggregate_training_results.items():
                tag = "losses/" + key
                WRITER.add_scalar(tag, value, self.current_round)

            for key, value in self.aggregated_eval_metrics.items():
                tag = key
                for val, s_class in enumerate(value):
                    WRITER.add_scalar(f"{tag}/{s_class}", value[s_class], self.current_round)

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    WRITER.add_histogram(
                        f"gradients/{name}", param.grad.cpu().data.numpy(), self.current_round
                    )

    def aggregate_training_and_validation_metrics(self, inputs):
        self.aggregate_training_results = {}
        for key in inputs[0].training_dicts.keys():
            self.aggregate_training_results[key] = sum(
                input.training_dicts[key].training_loss for input in inputs
            ) / len(inputs)

        self.aggregated_eval_metrics = {}
        for val_step_key in inputs[0].validation_dicts.keys():
            self.aggregated_eval_metrics[val_step_key] = {}
            for key in inputs[0].validation_dicts[val_step_key].keys():
                if isinstance(inputs[0].validation_dicts[val_step_key][key], list):
                    label_names = (
                        self.model.config.id2label
                        if hasattr(self.model.config, "id2label")
                        else [str(i) for i in range(self.model.num_labels)]
                    )
                    for val, s_class in enumerate(label_names):
                        self.aggregated_eval_metrics[val_step_key][key + "_" + s_class] = sum(
                            input.validation_dicts[val_step_key][key][val] for input in inputs
                        ) / len(inputs)
                else:
                    self.aggregated_eval_metrics[val_step_key][key] = sum(
                        input.validation_dicts[val_step_key][key] for input in inputs
                    ) / len(inputs)

        for val_step_key in self.validation_dicts.keys():
            self.aggregated_eval_metrics[val_step_key] = {}
            for key in self.validation_dicts[val_step_key].keys():
                if isinstance(self.validation_dicts[val_step_key][key], list):
                    label_names = (
                        self.model.config.id2label
                        if hasattr(self.model.config, "id2label")
                        else [str(i) for i in range(self.model.num_labels)]
                    )
                    for val, s_class in enumerate(label_names):
                        self.aggregated_eval_metrics[val_step_key][key + "_" + s_class] = (
                            self.validation_dicts[val_step_key][key][val]
                        )
                else:
                    self.aggregated_eval_metrics[val_step_key][key] = self.validation_dicts[
                        val_step_key
                    ][key]

    def perform_global_evaluation(self):
        if torch.cuda.is_available():
            self.model.to("cuda")
        if self.task_type in ["classification", "segmentation"]:
            metric_computer = self.get_metric_computer()
        else:
            metric_computer = None
        trainer = FederatedTrainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=self.global_validation_dataset,
            data_collator=self.collate_fn,
            compute_metrics=metric_computer.compute if metric_computer else None,
        )
        eval_results = trainer.evaluate()
        self.validation_dicts["global_eval_metrics"] = eval_results

        if self.move_to_cpu_at_end_of_training:
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()

    @aggregator
    def end(self):
        logger.info("Ending federated learning workflow...")
        if WRITER is not None:
            WRITER.close()  # Close the SummaryWriter
        logger.info("Workflow ended.")
