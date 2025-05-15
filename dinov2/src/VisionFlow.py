# %%
import os
import logging

import evaluate
import torch
from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from transformers import TrainingArguments
from datasets import Dataset
from tictoc import bench_dict

from src.Brats2020_dataloader import SEGMENT_CLASSES, collate_fn
from src.utils import FedAvg, FederatedTrainer, MetricAccumulator
from src.modeling.VisionModel import VisionModel


WRITER = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_writer(writer):
    global WRITER
    WRITER = writer


class VisionFlow(FLSpec):

    def __init__(
        self,
        model=None,# can be None to use default VisionModel or use your own model
        rounds=3,
        global_validation_dataset: Dataset = None,
        training_args: TrainingArguments = None,
        use_lora=False,
        move_to_cpu_end_of_training=False,
        **kwargs,
    ):
        logger.info("Initializing VisionFlow...")
        model_config_kwargs = kwargs.pop("model_config_kwargs", {})
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
        else:
            self.model = VisionModel(
                task_type="segmentation", use_peft=use_lora, num_labels=4, image_size=224, **model_config_kwargs
            )

        self.rounds = rounds
        self.global_validation_dataset = global_validation_dataset

        self.training_args = training_args
        self.use_lora = use_lora
        self.move_to_cpu_end_of_training = move_to_cpu_end_of_training

        self.lr_scheduler = None

        if self.move_to_cpu_end_of_training:
            self.model.to("cpu")
            torch.cuda.empty_cache()
        logger.info("VisionFlow initialized with %d rounds.", self.rounds)

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
        metric = evaluate.load("mean_iou")
        metric_accumulator = MetricAccumulator(metric, len(SEGMENT_CLASSES))
        self.train_dataset = self.train_dataset.shuffle()

        self.trainer = FederatedTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=collate_fn,
            compute_metrics=metric_accumulator.compute,
            optimizers=(None, self.lr_scheduler),
            total_rounds=self.rounds,
        )

        eval_dict = self.trainer.evaluate()
        self.agg_validation_dict = eval_dict
        logger.info("Aggregated model validation completed on collaborator: %s", self.input)
        self.next(self.train)

    @collaborator
    def train(self):
        logger.info("Starting training on collaborator: %s", self.input)
        training_dict = self.trainer.train()
        self.training_completed = True
        self.train_dict = training_dict
        logger.info("Training completed on collaborator: %s", self.input)
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        logger.info("Performing local model validation on collaborator: %s", self.input)
        self.trainer.evaluate()
        eval_dict = self.trainer.evaluate()
        self.local_validation_dict = eval_dict

        if self.move_to_cpu_end_of_training:
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
        if self.use_lora:
            peft_weights = [input.weights[0] for input in inputs]
            class_weights = [input.weights[1] for input in inputs]
            peft_weights = FedAvg(peft_weights)
            class_weights = FedAvg(class_weights)
            weights = (peft_weights, class_weights)
        else:
            weights = FedAvg([input.weights for input in inputs])

        self.model.set_weights(weights)

        metric = evaluate.load("mean_iou")

        if torch.cuda.is_available():
            self.model.to("cuda")

        metric_accumulator = MetricAccumulator(metric, len(SEGMENT_CLASSES))
        trainer = FederatedTrainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=self.global_validation_dataset,
            data_collator=collate_fn,
            compute_metrics=metric_accumulator.compute,
        )
        eval_results = trainer.evaluate()

        if self.move_to_cpu_end_of_training:
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()

        self.global_eval_metrics = {}
        for key in eval_results.keys():
            if isinstance(eval_results[key], list):
                for val, s_class in SEGMENT_CLASSES.items():
                    self.global_eval_metrics[key + "_" + s_class] = eval_results[key][val]
            else:
                self.global_eval_metrics[key] = eval_results[key]

        self.average_loss = sum(input.train_dict.training_loss for input in inputs) / len(inputs)
        self.aggregated_eval_metrics = {}
        for key in inputs[0].agg_validation_dict.keys():
            if isinstance(inputs[0].agg_validation_dict[key], list):
                for val, s_class in SEGMENT_CLASSES.items():
                    self.aggregated_eval_metrics[key + "_" + s_class] = sum(
                        input.agg_validation_dict[key][val] for input in inputs
                    ) / len(inputs)
            else:
                self.aggregated_eval_metrics[key] = sum(
                    input.agg_validation_dict[key] for input in inputs
                ) / len(inputs)

        self.local_eval_metrics = {}
        for key in inputs[0].local_validation_dict.keys():
            if isinstance(inputs[0].local_validation_dict[key], list):
                for val, s_class in SEGMENT_CLASSES.items():
                    self.local_eval_metrics[key + "_" + s_class] = sum(
                        input.local_validation_dict[key][val] for input in inputs
                    ) / len(inputs)
            else:
                self.local_eval_metrics[key] = sum(
                    input.local_validation_dict[key] for input in inputs
                ) / len(inputs)

        WRITER.add_scalar("losses/average_training_loss", self.average_loss, self.current_round)
        for key, value in self.aggregated_eval_metrics.items():
            tag = "aggregated/" + key
            WRITER.add_scalar(tag, value, self.current_round)

        for key, value in self.local_eval_metrics.items():
            tag = "local/" + key
            WRITER.add_scalar(tag, value, self.current_round)

        for key, value in self.global_eval_metrics.items():
            tag = "global/" + key
            WRITER.add_scalar(tag, value, self.current_round)

        trainer.save_model()

        self.current_round += 1
        logger.info("Collaborator inputs joined. Metrics logged.")
        if self.current_round < self.rounds:
            self.next(
                self.aggregated_model_validation,
                foreach="collaborators",
            )
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        logger.info("Ending federated learning workflow...")
        WRITER.close()  # Close the SummaryWriter
        logger.info("Workflow ended.")
