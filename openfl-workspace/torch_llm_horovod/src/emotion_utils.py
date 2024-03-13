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


def get_emotion_dataset(tokenizer):
    dataset = load_dataset("dair-ai/emotion", cache_dir="dataset", revision="9ce6303")

    assert (
        dataset["train"]._fingerprint == "cbee262593e53bad"
    ), "The hash for the train dataset doesn't match"
    assert (
        dataset["test"]._fingerprint == "742f5ef1ea3947e2"
    ), "The hash for the test dataset doesn't match"
    assert (
        dataset["validation"]._fingerprint == "7e98c3be2fbf8e9e"
    ), "The hash for the validation dataset doesn't match"

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=None,
        )
        return outputs

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    return data_collator, tokenized_datasets


class EmotionDataset(Dataset):
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
    data_collator, tokenized_datasets = get_emotion_dataset(tokenizer)

    train_set = EmotionDataset.from_dict(tokenized_datasets["train"].to_dict())
    valid_set = EmotionDataset.from_dict(tokenized_datasets["test"].to_dict())
    return train_set, valid_set, data_collator
