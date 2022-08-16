"""Copyright (C) 2020-2021 Intel Corporation
   SPDX-License-Identifier: Apache-2.0

Licensed subject to the terms of the separately executed evaluation
license agreement between Intel Corporation and you.
"""
from logging import getLogger
from os import getcwd
from os import path
from os import remove
from zipfile import ZipFile

import numpy as np
import requests

logger = getLogger(__name__)


def download_data_():
    """Download data.

    Returns:
      string: relative path to data file
    """
    pkg = 'fra-eng.zip'   # Language file: change this to change the language
    data_dir = 'data'
    url = 'https://www.manythings.org/anki/' + pkg
    filename = pkg.split('-')[0] + '.txt'

    workspace_dir = getcwd()
    default_path = path.join(workspace_dir, data_dir)
    pkgpath = path.join(default_path, pkg)       # path to downloaded zipfile
    filepath = path.join(default_path, filename)  # path to extracted file

    if path.isfile(filepath):
        return path.join(data_dir, filename)
    try:
        response = requests.get(url, headers={'User-Agent': 'openfl'})
        if response.status_code == 200:
            with open(pkgpath, 'wb') as f:
                f.write(response.content)
        else:
            print(f'Error while downloading {pkg} from {url}: Aborting!')
            exit()
    except Exception:
        print(f'Error while downloading {pkg} from {url}: Aborting!')
        exit()

    try:
        with ZipFile(pkgpath, 'r') as z:
            z.extract(filename, default_path)
    except Exception:
        print(f'Error while extracting {pkgpath}: Aborting!')
        exit()

    if path.isfile(filepath):
        remove(pkgpath)
        return path.join(data_dir, filename)
    else:
        return ''


def import_raw_data_(data_path='', num_samples=0):
    """Import data.

    Returns:
      dict: variable details
      numpy.ndarray: encoder input data
      numpy.ndarray: decoder input data
      numpy.ndarray: decoder labels
    """
    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text, _ = line.split('\t')
        # We use 'tab' as the 'start sequence' character
        # for the targets, and '\n' as 'end sequence' character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(input_characters)
    target_characters = sorted(target_characters)
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    details = {'num_samples': len(input_texts),
               'num_encoder_tokens': num_encoder_tokens,
               'num_decoder_tokens': num_decoder_tokens,
               'max_encoder_seq_length': max_encoder_seq_length,
               'max_decoder_seq_length': max_decoder_seq_length}

    input_token_index = {char: i for i, char in enumerate(input_characters)}
    target_token_index = {char: i for i, char in enumerate(target_characters)}

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')

    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.0
        decoder_target_data[i, t:, target_token_index[' ']] = 1.0

    logger.info(f'[DL]-import_raw_data: Number of samples = {len(input_texts)}')
    logger.info(f'[DL]-import_raw_data: Number of unique input tokens = {num_encoder_tokens}')
    logger.info(f'[DL]-import_raw_data: '
                f'Number of unique decoder tokens = {num_decoder_tokens}')

    logger.info(f'[DL]-import_raw_data: '
                f'Max sequence length for inputs = {max_encoder_seq_length}')

    logger.info(f'[DL]-import_raw_data: '
                f'Max sequence length for outputs = {max_decoder_seq_length}')

    logger.info(f'[DL]-import_raw_data: encoder_input_data = {encoder_input_data.shape}')
    logger.info(f'[DL]-import_raw_data: decoder_input_data = {decoder_input_data.shape}')
    logger.info(f'[DL]-import_raw_data: decoder_target_data = {decoder_target_data.shape}')

    return details, encoder_input_data, decoder_input_data, decoder_target_data


def get_datasets_(encoder_input_data, decoder_input_data,
                  decoder_target_data, num_samples, split_ratio):
    """Create train/val.

    Returns:
      dict: Results, containing the train-valid split of the dataset (split_ratio = 0.2)
    """
    import random

    random.seed(42)
    train_indexes = random.sample(range(num_samples), int(num_samples * (1 - split_ratio)))
    valid_indexes = np.delete(range(num_samples), train_indexes)

    # Dataset creation (2 inputs <encoder,decoder>, 1 output <decoder_target>)
    encoder_train_input = encoder_input_data[train_indexes, :, :]
    decoder_train_input = decoder_input_data[train_indexes, :, :]
    decoder_train_labels = decoder_target_data[train_indexes, :, :]

    encoder_valid_input = encoder_input_data[valid_indexes, :, :]
    decoder_valid_input = decoder_input_data[valid_indexes, :, :]
    decoder_valid_labels = decoder_target_data[valid_indexes, :, :]

    results = {'encoder_train_input': encoder_train_input,
               'decoder_train_input': decoder_train_input,
               'decoder_train_labels': decoder_train_labels,
               'encoder_valid_input': encoder_valid_input,
               'decoder_valid_input': decoder_valid_input,
               'decoder_valid_labels': decoder_valid_labels}

    logger.info(f'[DL]get_datasets: encoder_train_input = {encoder_train_input.shape}')
    logger.info(f'[DL]get_datasets: decoder_train_labels= {decoder_train_labels.shape}')

    return results


def load_shard(collaborator_count, shard_num, data_path, num_samples, split_ratio):
    """Load data-shards.

    Returns:
      Tuple: ( numpy.ndarray: X_train_encoder,
               numpy.ndarray: X_train_decoder,
               numpy.ndarray: y_train)
      Tuple: ( numpy.ndarray: X_valid_encoder,
               numpy.ndarray: X_valid_decoder,
               numpy.ndarray: y_valid)
      Dict: details, from DataLoader_utils.get_datasets
    """
    details, encoder_input_data, decoder_input_data, decoder_target_data = import_raw_data_(
        data_path,
        num_samples
    )

    train_val_dataset = get_datasets_(encoder_input_data, decoder_input_data,
                                      decoder_target_data, num_samples, split_ratio)
    # Get the data shards
    shard_num = int(shard_num)
    X_train_encoder = train_val_dataset['encoder_train_input'][shard_num::collaborator_count]
    X_train_decoder = train_val_dataset['decoder_train_input'][shard_num::collaborator_count]
    y_train = train_val_dataset['decoder_train_labels'][shard_num::collaborator_count]

    X_valid_encoder = train_val_dataset['encoder_valid_input'][shard_num::collaborator_count]
    X_valid_decoder = train_val_dataset['decoder_valid_input'][shard_num::collaborator_count]
    y_valid = train_val_dataset['decoder_valid_labels'][shard_num::collaborator_count]

    logger.info(f'[DL]load_shard: X_train_encoder = {X_train_encoder.shape}')
    logger.info(f'[DL]load_shard: y_train = {y_train.shape}')

    return (
        (X_train_encoder, X_train_decoder, y_train),
        (X_valid_encoder, X_valid_decoder, y_valid),
        details
    )
