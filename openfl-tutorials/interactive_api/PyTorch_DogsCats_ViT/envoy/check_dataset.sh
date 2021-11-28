#!/bin/bash
set -e 

mkdir -p data

cd data

if [ -s dataset.md5 ] && tar -cf - train test | md5sum --quiet -c dataset.md5
then
    echo "Dataset is OK"
else
    echo "Your dataset is absent or damaged. Downloading ... "
    rm -rf train test  
    kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
    unzip dogs-vs-cats-redux-kernels-edition.zip
    rm dogs-vs-cats-redux-kernels-edition.zip
    rm sample_submission.csv
    unzip train.zip
    unzip test.zip
    rm train.zip test.zip
    tar -cf - train test | md5sum > dataset.md5
    echo "Done"
fi
cd ..