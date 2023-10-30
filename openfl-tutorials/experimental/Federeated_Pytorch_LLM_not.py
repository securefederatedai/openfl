
from typing import Any, Mapping
import numpy as np
import openfl.native as fx
import torch
import torch as pt
from datasets import Dataset, load_dataset, load_metric
from openfl.federated import PyTorchTaskRunner
from openfl.federated.task.runner_pt import change_tags
from openfl.utilities import Metric, TensorKey
from openfl.utilities.data_splitters import EqualNumPyDataSplitter
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from transformers.trainer_pt_utils import get_parameter_names
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding, get_scheduler)
import argparse

def get_glue_mrpc_dataset(tokenizer):
    dataset = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=None,
        )
        return outputs

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    return data_collator, tokenized_datasets

class GlueMrpc(Dataset):
    """
    Has 5.8k pairs of sentences with annotations if the two sentences are equivalent
    """    
    def get_shape(self):
        
        if not hasattr(self, 'saved_shape'):
            self.saved_shape = max([len(i) for i in self.data['input_ids']])
        return self.saved_shape

class GlueMrpcFederatedDataset(DataLoader):
    def __init__(self, train_set, valid_set, batch_size, data_collator=None):
        self.data_splitter = EqualNumPyDataSplitter(shuffle=True)
        if isinstance(train_set,Dataset):
            self.train_set = GlueMrpc.from_dict(train_set.to_dict())
        else:
            self.train_set = train_set
            
        if isinstance(valid_set,Dataset):
            self.valid_set = GlueMrpc.from_dict(valid_set.to_dict())
        else:
            self.valid_set = valid_set            
            
        self.batch_size = batch_size
        self.data_collator = data_collator
    
    def split(self, num_collaborators):
        train_split = self.data_splitter.split(self.train_set, num_collaborators)
        valid_split = self.data_splitter.split(self.valid_set, num_collaborators)
        return [
            GlueMrpcFederatedDataset(
                self.train_set.select(train_split[i]),
                self.valid_set.select(valid_split[i]),
                self.batch_size
            )
            for i in range(num_collaborators)
        ]
    
    def get_feature_shape(self):
        return self.train_set.get_shape()
    
    def get_train_loader(self, num_batches=None):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=self.data_collator)
    
    def get_valid_loader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, collate_fn=self.data_collator)
    
    def get_train_data_size(self):
        return len(self.train_set)
    
    def get_valid_data_size(self):
        return len(self.valid_set)
    

class LLMTaskRunner(PyTorchTaskRunner):
    def __init__(
        self, base_model_name, data_loader, device=None, metric=None, args=None, **kwargs
    ):
        kwargs["data_loader"] = data_loader
        super().__init__(device, **kwargs)
        self.base_model_name = base_model_name
        self.metric = metric
        self._init_model()
        self._init_optimizer()
        self.save_models = []

    def _init_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name, return_dict=True
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="lora_only",
        )
        self.model = get_peft_model(model, peft_config)

    def _init_optimizer(self):
        ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=0.001)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.data_loader.train_set) * 5,
        )

        self.training_round_completed = False
        self.initialize_tensorkeys_for_functions()

    def train(self):
        return self.model.train()

    def state_dict(self):
        return get_peft_model_state_dict(self.model)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return set_peft_model_state_dict(self.model, state_dict)

    def validate(
        self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs
    ):
        """Validate.

        Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB

        """
        self.save_models.append(input_tensor_dict.copy())
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.model.eval()
        

        self.model.to(self.device)
        val_score = 0
        total_samples = 0

        loader = self.data_loader.get_valid_loader()
        if use_tqdm:
            loader = tqdm(loader, desc="validate")

        with pt.no_grad():
            for sample in loader:
                samples = sample["input_ids"].shape[0]
                total_samples += samples
                output = self.model(**sample)
                # get the index of the max log-probability
                logits = output.logits
                predictions = torch.argmax(logits, dim=-1)
                self.metric.add_batch(predictions=predictions, references=sample["labels"])
        val_score = self.metric.compute()["accuracy"]

        origin = col_name
        suffix = "validate"
        if kwargs["apply"] == "local":
            suffix += "_local"
        else:
            suffix += "_agg"
        tags = ("metric",)
        tags = change_tags(tags, add_field=suffix)
        # TODO figure out a better way to pass in metric for this pytorch
        #  validate function
        output_tensor_dict = {
            TensorKey("acc", origin, round_num, True, tags): np.array(val_score)
        }

        # Empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def train_epoch(self, batch_generator) -> Metric:
        """Train single epoch.

        Override this function in order to use custom training.

        Args:
            batch_generator: Train dataset batch generator. Yields (samples, targets) tuples of
            size = `self.data_loader.batch_size`.
        Returns:
            Metric: An object containing name and np.ndarray value.
        """
        losses = []
        for sample in batch_generator:
            self.model.zero_grad()
            output = self.model(**sample)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            losses.append(loss.detach().cpu().numpy())
        loss = np.mean(losses)
        if self.model.config.problem_type == "regression":
            loss_fct = MSELoss()
        elif self.model.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
        elif self.model.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
        return Metric(name=loss_fct._get_name(), value=np.array(loss))

    def save_native(
        self,
        filepath,
        model_state_dict_key="model_state_dict",
        optimizer_state_dict_key="optimizer_state_dict",
        **kwargs,
    ):
        """
        Save model and optimizer states in a picked file specified by the \
        filepath. model_/optimizer_state_dicts are stored in the keys provided. \
        Uses pt.save().

        Args:
            filepath (string)                 : Path to pickle file to be
                                                created by pt.save().
            model_state_dict_key (string)     : key for model state dict
                                                in pickled file.
            optimizer_state_dict_key (string) : key for optimizer state
                                                dict in picked file.
            kwargs                            : unused

        Returns:
            None
        """
        pickle_dict = {
            model_state_dict_key: get_peft_model_state_dict(self.model),
            optimizer_state_dict_key: self.optimizer.state_dict(),
        }
        pt.save(pickle_dict, filepath)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    # Include DeepSpeed configuration arguments

    args = parser.parse_args()

    return args

def get_dataset(base_model_name = "roberta-base", padding_side = "right"):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    data_collator, tokenized_datasets = get_glue_mrpc_dataset(tokenizer)

    train_set = GlueMrpc.from_dict(tokenized_datasets['train'].to_dict())
    valid_set = GlueMrpc.from_dict(tokenized_datasets['test'].to_dict())
        
    fl_data = GlueMrpcFederatedDataset(train_set, valid_set, batch_size=32, data_collator=data_collator)
    return fl_data

def main():
    args = get_arguments()
    fx.init('torch_llm')
    base_model_name = "roberta-base"
    padding_side = "right"
    fl_data = get_dataset(base_model_name, padding_side)
    metric = load_metric('glue', "mrpc")

    num_collaborators = 2
    collaborator_models = [
                LLMTaskRunner(
                    base_model_name,
                    data_loader=data_slice,
                    metric=metric, args=args
                )
                for data_slice in fl_data.split(num_collaborators)]
    collaborators = {'one':collaborator_models[0],'two':collaborator_models[1]}#, 'three':collaborator_models[2]}

    # %%
    #Original TinyImageNet dataset
    print(f'Original training data size: {len(fl_data.train_set)}')
    print(f'Original validation data size: {len(fl_data.valid_set)}\n')

    #Collaborator one's data
    for i, model in enumerate(collaborator_models):
        print(f'Collaborator {i}\'s training data size: {len(model.data_loader.train_set)}')
        print(f'Collaborator {i}\'s validation data size: {len(model.data_loader.valid_set)}\n')
    final_fl_model = fx.run_experiment(collaborators,{'aggregator.settings.rounds_to_train':10,"tasks.train.kwargs.epochs":2})
    

if __name__ == "__main__":
    main()