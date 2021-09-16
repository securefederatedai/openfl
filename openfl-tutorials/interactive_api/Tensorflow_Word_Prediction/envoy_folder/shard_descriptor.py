# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Market shard descriptor."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class NextWordShardDescriptor(ShardDescriptor):
    """"Data - any text."""

    def __init__(self, title: str = '', author: str = '') -> None:
        """Initialize NextWordShardDescriptor."""
        super().__init__()

        self.title = title
        self.author = author
        self.dataset_dir = list(Path.cwd().rglob(f'{title}.txt'))[0]
        self.data = self.load_data(self.dataset_dir)  # list of words
        self.X, self.y = self.get_sequences(self.data)

    def __len__(self):
        """Number of sequences."""
        return len(self.X)

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.X[index], self.y[index]

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['3', '96']  # three vectors

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['10719']  # row at one-hot matrix with n = vocab_size

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
        """
        Transform words to sequences, for X transform to vectors as well.

        To make vocab, clean it, sort by frequency (pip install wordfreq) and get keyed vectors:
            if not spacy.util.is_package('en_core_web_sm'):
                spacy.cli.download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')

            clean_vocab_list = [word for word in nlp.vocab.strings
                                if re.fullmatch(r'[a-z]+', word) and
                                if wordfreq.zipf_frequency(word, 'en', wordlist='small') > 3.7]

            word_to_vector = pd.Series([], name='vector')
            for word in clean_vocab_list:
                word_to_vector[word] = nlp(word).vector
            word_to_vector.to_pickle('keyed_vectors.pkl')
        """
        # spacy en_core_web_sm vocab_size = 10719, vector_size = 96
        x_seq = []
        y_seq = []
        vectors = pd.read_pickle('keyed_vectors.pkl')
        for i in range(len(data) - 3):
            x = data[i:i + 3]  # make 3-grams
            y = data[i + 3]
            cur_x = [vectors[word] for word in x if word in vectors]
            if len(cur_x) == 3 and y in vectors:
                x_seq.append(cur_x)
                y_seq.append(vectors.index.get_loc(y))

        x_seq = np.array(x_seq)
        y_seq = to_categorical(y_seq, num_classes=vectors.shape[0])
        return x_seq, y_seq