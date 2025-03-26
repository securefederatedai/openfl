#!/bin/bash
curl -L -o data/brats2020-training-data.zip\
  https://www.kaggle.com/api/v1/datasets/download/awsaf49/brats20-dataset-training-validation
unzip data/brats2020-training-data.zip -d data/
python src/preprocess_dataset.py --dataset_path data