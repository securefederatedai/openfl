# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from logging import getLogger, Logger

from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import hashlib
from typing import Any, Optional

logger: Logger = getLogger(__name__)

writer: Optional[SummaryWriter] = None


def file_checksum(file_path: str, algorithm: str = "sha256") -> str:
    hash_func: hashlib._Hash = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def get_writer() -> None:
    """Create global writer object."""
    global writer
    if not writer:
        writer = SummaryWriter("./logs/llm", flush_secs=5)


def write_metric(
    node_name: str,
    task_name: str,
    metric_name: str,
    metric: Any,
    round_number: int,
) -> None:
    """Write metric callback."""
    get_writer()
    writer.add_scalar(
        f"{node_name}/{task_name}/{metric_name}", metric, round_number
    )


def preprocess_dataset(
    raw_dataset, model_name, sequence_max_length=512, val_set_size=2000
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.model_max_length = 2048
    tokenizer.pad_token = (
        tokenizer.unk_token  # unk. we want this to be different from the eos token
    )
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
        tokenizer.pad_token
    )
    tokenizer.padding_side = "right"

    def apply_chat_template(
        example,
        tokenizer,
    ):
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return example

    dataset_train_test = raw_dataset["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    column_names = list(dataset_train_test["train"].features)

    processed_train_dataset = (
        dataset_train_test["train"]
        .shuffle()
        .map(
            apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=1,
            remove_columns=column_names,
        )
        .select(range(4))
    )
    processed_test_dataset = (
        dataset_train_test["test"]
        .shuffle()
        .map(
            apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=1,
            remove_columns=column_names,
        )
        .select(range(4))
    )
    return processed_train_dataset, processed_test_dataset, tokenizer


def get_dataset(sequence_max_length=512, val_set_size=2000):
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    raw_dataset = load_dataset("json", data_files="medquad_alpaca_train.json")
    processed_train_dataset, processed_test_dataset, tokenizer = (
        preprocess_dataset(
            raw_dataset, model_name, sequence_max_length, val_set_size
        )
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    return (
        processed_train_dataset,
        processed_test_dataset,
        data_collator,
        tokenizer,
    )
