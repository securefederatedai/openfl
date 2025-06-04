#!/bin/bash
curl -L -o data/brats2020-training-data.zip\
  https://www.kaggle.com/api/v1/datasets/download/awsaf49/brats20-dataset-training-validation
unzip data/brats2020-training-data.zip -d data/
python src/setup_data.py --dataset_path data
rm -rf data/brats2020-training-data.zip