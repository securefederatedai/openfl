# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from logging import getLogger

from datasets import Dataset, load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, DataCollatorWithPadding

logger = getLogger(__name__)

writer = None


def get_writer():
    """Create global writer object."""
    global writer
    if not writer:
        writer = SummaryWriter("./logs/llm", flush_secs=5)


def write_metric(node_name, task_name, metric_name, metric, round_number):
    """Write metric callback."""
    get_writer()
    writer.add_scalar(f"{node_name}/{task_name}/{metric_name}", metric, round_number)


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
        if not hasattr(self, "saved_shape"):
            self.saved_shape = max([len(i) for i in self.data["input_ids"]])
        return self.saved_shape


def get_dataset(base_model_name="roberta-base", padding_side="right"):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, padding_side=padding_side
    )
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    data_collator, tokenized_datasets = get_glue_mrpc_dataset(tokenizer)

    train_set = GlueMrpc.from_dict(tokenized_datasets["train"].to_dict())
    valid_set = GlueMrpc.from_dict(tokenized_datasets["test"].to_dict())
    return train_set, valid_set, data_collator
