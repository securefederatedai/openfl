#!/bin/bash

rm -rf MedQuAD
rm medquad_alpaca_test.json
rm medquad_alpaca_train.json
rm data_counts.txt
git clone https://github.com/abachaa/MedQuAD.git
python preprocess_dataset.py
rm -rf MedQuAD