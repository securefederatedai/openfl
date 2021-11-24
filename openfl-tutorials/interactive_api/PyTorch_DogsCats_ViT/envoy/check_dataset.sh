#!/bin/bash
set -e 

mkdir -p data

cd data
DIR_SIZE=$(du -s . | cut -f1)
if [ $DIR_SIZE -lt 900000 -a $DIR_SIZE -gt 4 ]
then
echo -n "Your dataset is not full. Directories 'train' and 'test' will be removed and redownloaded. Сontinue? (y/n) "
read answer
case "$answer" in 
    y|Y|yes|Yes)
        rm -rf train test  
        kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
        unzip dogs-vs-cats-redux-kernels-edition.zip
        rm dogs-vs-cats-redux-kernels-edition.zip
        rm sample_submission.csv
        unzip train.zip
        unzip test.zip
        rm train.zip test.zip
        echo "Dataset was downloaded"
        echo "Current dataset directories: "; ls;;
    *) echo "The dataset has been left unchanged";;
esac
elif [ $DIR_SIZE -le 4 ]
then
    kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
    unzip dogs-vs-cats-redux-kernels-edition.zip
    rm dogs-vs-cats-redux-kernels-edition.zip
    rm sample_submission.csv
    unzip train.zip
    unzip test.zip
    rm train.zip test.zip
    echo "Dataset was downloaded"
    echo "Current dataset directories: "; ls
fi
cd ..