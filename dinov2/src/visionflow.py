# %%
import evaluate
import torch
import os

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator

os.chdir(PATH)
from src.dataloader import SEGMENT_CLASSES, collate_fn
from src.utils import MetricAccumulator, FederatedTrainer, FedAvg
from tictoc import bench_dict

benchmarker = bench_dict["workflow"]
benchmarker.enable_memory_tracking()
benchmarker.enable_memory_tracking_in_step()
benchmarker.memory_benchmaker.enable_cuda_memory_tracking()
benchmarker.save_on_gstop = 1
benchmarker.save_on_step = True
benchmarker.disable()

WRITER = None


def set_writer(writer):
    global WRITER
    WRITER = writer


class FederatedFlow(FLSpec):

    def __init__(
        self, model=None, rounds=3, val_set=None, training_args=None, use_lora=False, **kwargs
    ):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
        self.rounds = rounds
        self.val_set = val_set
        self.training_args = training_args
        self.use_lora = use_lora

        self.model.to("cpu")
        torch.cuda.empty_cache()
        self.lr_scheduler = None

    @aggregator
    def start(self):
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.current_round = 0
        benchmarker.gstep()
        self.next(self.aggregated_model_validation, foreach="collaborators", exclude=["private"])

    @collaborator
    def aggregated_model_validation(self):
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
        print("aggregated_model_validation", input, eval_dict["eval_per_category_iou"][1])
        benchmarker.step("aggregated_model_validation")
        self.next(self.train)

    @collaborator
    def train(self):
        training_dict = self.trainer.train()
        self.training_completed = True
        self.train_dict = training_dict
        benchmarker.step("train")
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.trainer.evaluate()
        eval_dict = self.trainer.evaluate()
        self.local_validation_dict = eval_dict
        self.model.to("cpu")
        torch.cuda.empty_cache()
        self.lr_scheduler = self.trainer.lr_scheduler
        self.weights = self.model.get_weights()
        benchmarker.step("local_model_validation")
        print("local_model_validation", input, eval_dict["eval_per_category_iou"][1])
        self.next(self.join, exclude=["training_completed", "trainer"])

    @aggregator
    def join(self, inputs):
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
        metric_accumulator = MetricAccumulator(metric, len(SEGMENT_CLASSES))
        trainer = FederatedTrainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=self.val_set,
            data_collator=collate_fn,
            compute_metrics=metric_accumulator.compute,
        )
        benchmarker.step("join")
        eval_results = trainer.evaluate()
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()
        self.global_eval_metrics = {}
        for key in eval_results.keys():
            if isinstance(eval_results[key], list):
                for val, s_class in SEGMENT_CLASSES.items():
                    self.global_eval_metrics[key + "_" + s_class] = eval_results[key][val]
            else:
                self.global_eval_metrics[key] = eval_results[key]

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

        benchmarker.step("write_to_tensorboard")
        benchmarker.gstop()

        trainer.save_model()

        self.current_round += 1
        if self.current_round < self.rounds:
            benchmarker.gstep()
            print("Starting round ", self.current_round)
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
        WRITER.close()  # Close the SummaryWriter
        benchmarker.save_data()
