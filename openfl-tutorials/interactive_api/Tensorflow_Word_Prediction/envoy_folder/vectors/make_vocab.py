# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Vocab extracting, cleaning, and matching with vectors."""

import re

import pandas as pd
import spacy
from wordfreq import zipf_frequency

if not spacy.util.is_package('en_core_web_sm'):
    spacy.cli.download('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

clean_vocab_list = [word for word in nlp.vocab.strings
                    if re.fullmatch(r'[a-z]+', word) and
                    zipf_frequency(word, 'en', wordlist='small') > 3.7]

word_to_vector = pd.Series([], name='vector')
for word in clean_vocab_list:
    word_to_vector[word] = nlp(word).vector
word_to_vector.to_frame().reset_index().to_feather('keyed_vectors.feather')
