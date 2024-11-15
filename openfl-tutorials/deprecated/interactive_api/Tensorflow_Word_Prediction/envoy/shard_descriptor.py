# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shard descriptor for text."""

import re
import urllib.request
import zipfile
from pathlib import Path

import gdown
import numpy as np
import pandas as pd

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class NextWordShardDataset(ShardDataset):
    """Shard Dataset for text."""

    def __init__(self, X, y):
        """Initialize NextWordShardDataset."""
        self.X = X
        self.y = y

    def __len__(self):
        """Count number of sequences."""
        return len(self.X)

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.X[index], self.y[index]


class NextWordShardDescriptor(ShardDescriptor):
    """Data is any text."""

    def __init__(self, title: str = '', author: str = '') -> None:
        """Initialize NextWordShardDescriptor."""
        super().__init__()

        self.title = title
        self.author = author

        dataset_dir = self.download_data(title)
        data = self.load_data(dataset_dir)  # list of words
        self.X, self.y = self.get_sequences(data)

    def get_dataset(self, dataset_type='train', train_val_split=0.8):
        """Return a dataset by type."""
        train_size = round(len(self.X) * train_val_split)
        if dataset_type == 'train':
            X = self.X[:train_size]
            y = self.y[:train_size]
        elif dataset_type == 'val':
            X = self.X[train_size:]
            y = self.y[train_size:]
        else:
            raise Exception(f'Wrong dataset type: {dataset_type}.'
                            f'Choose from the list: [train, val]')
        return NextWordShardDataset(X, y)

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        length, n_gram, vector_size = self.X.shape
        return [str(n_gram), str(vector_size)]  # three vectors

    @property
    def target_shape(self):
        """Return the target shape info."""
        length, vocab_size = self.y.shape
        return [str(vocab_size)]  # row at one-hot matrix with n = vocab_size

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return f'Dataset from {self.title} by {self.author}'

    @staticmethod
    def load_data(path):
        """Load text file, return list of words."""
        file = open(path, 'r', encoding='utf8').read()
        data = re.findall(r'[a-z]+', file.lower())
        return data

    @staticmethod
    def get_sequences(data):
        """Transform words to sequences, for X transform to vectors as well."""
        # spacy en_core_web_sm vocab_size = 10719, vector_size = 96
        x_seq = []
        y_seq = []

        # created with make_vocab.py from
        # https://gist.github.com/katerina-merkulova/e351b11c67832034b49652835b14adb0
        NextWordShardDescriptor.download_vectors()
        vectors = pd.read_feather('keyed_vectors.feather')
        vectors.set_index('index', inplace=True)

        for i in range(len(data) - 3):
            x = data[i:i + 3]  # make 3-grams
            y = data[i + 3]
            cur_x = [vectors.vector[word] for word in x if word in vectors.index]
            if len(cur_x) == 3 and y in vectors.index:
                x_seq.append(cur_x)
                y_seq.append(vectors.index.get_loc(y))

        x_seq = np.array(x_seq)
        y_seq = np.array(y_seq)
        y = np.zeros((y_seq.size, 10719))
        y[np.arange(y_seq.size), y_seq] = 1
        return x_seq, y

    @staticmethod
    def download_data(title):
        """Download text by title form Github Gist."""
        url = ('https://gist.githubusercontent.com/katerina-merkulova/e351b11c67832034b49652835b'
               '14adb0/raw/5b6667c3a2e1266f3d9125510069d23d8f24dc73/' + title.replace(' ', '_')
               + '.txt')
        filepath = Path.cwd() / f'{title}.txt'
        if not filepath.exists():
            with urllib.request.urlopen(url) as response:
                content = response.read().decode('utf-8')
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(content)
        return filepath

    @staticmethod
    def download_vectors():
        """Download vectors."""
        if Path('keyed_vectors.feather').exists():
            return None

        output = 'keyed_vectors.zip'
        if not Path(output).exists():
            url = 'https://drive.google.com/uc?id=1QfidtkJ9qxzNLs1pgXoY_hqnBjsDI_2i'
            gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(Path.cwd())

        Path(output).unlink()  # remove zip
