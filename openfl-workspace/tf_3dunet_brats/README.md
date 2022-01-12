# TensorFlow 3D U-Net for the BraTS dataset

This is a full example for training the Brain Tumor Segmentation 2020 ([BraTS2020](https://www.med.upenn.edu/cbica/brats2020/data.html)) with OpenFL. 

*Note: This is **not** the 3D U-Net model that was used in the paper and not the sharding used. Nevertheless, it should make a good template for how to train using OpenFL.*

The files `src\dataloader.py` and `src\define_model.py` are where we define the TensorFlow [dataset loader](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) and the 3D U-Net model. In `src\dataloader.py` we demonstrate how to use an out-of-memory data loader that pulls batches of data from files as needed.

## Steps to run

1. Download the [BraTS 2020 dataset](https://www.med.upenn.edu/cbica/brats2020/registration.html). It should be the one labeled **BraTS'20 Training Data: Segmentation Task**. 

2. Extract the `MICCAI_BraTS2020_TrainingData.zip` zip file to any folder. Let's call that folder `${DATA_PATH}`. The file structure of `${DATA_PATH}` should look like this: 

```bash
user@localhost ~$ tree ${DATA_PATH} -L 2
${DATA_PATH}/MICCAI_BraTS2020_TrainingData
├── BraTS20_Training_001
│   ├── BraTS20_Training_001_flair.nii.gz    <── The MRI FLAIR channel (best one for prediction)
│   ├── BraTS20_Training_001_seg.nii.gz      <── The ground truth label
│   ├── BraTS20_Training_001_t1.nii.gz       <── The T1-weighted MRI channel
│   ├── BraTS20_Training_001_t1ce.nii.gz     <── The T1-Contrast Enhanced-weighted MRI channel
│   └── BraTS20_Training_001_t2.nii.gz       <── The T2-weighted MRI channel
├── BraTS20_Training_002
│   ├── BraTS20_Training_002_flair.nii.gz
│   ├── BraTS20_Training_002_seg.nii.gz
│   ├── BraTS20_Training_002_t1.nii.gz
│   ├── BraTS20_Training_002_t1ce.nii.gz
│   └── BraTS20_Training_002_t2.nii.gz
├── ...
├── BraTS20_Training_369
│   ├── BraTS20_Training_369_flair.nii.gz
│   ├── BraTS20_Training_369_seg.nii.gz
│   ├── BraTS20_Training_369_t1.nii.gz
│   ├── BraTS20_Training_369_t1ce.nii.gz
│   └── BraTS20_Training_369_t2.nii.gz
├── name_mapping.csv
└── survival_info.csv
```
If `tree` is not installed, then run `sudo apt-get install tree` to install it (Ubuntu).

3. In order for each collaborator to use separate slice of data, we split main folder into subfolders, one for each collaborator. **NOTE:** In the real world each collaborator will have it's own data and this split already exists. We're splitting here to simulate a federation with different participants.

#### `split_directory.sh`
```bash 
#!/bin/bash
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Split the BraTS data directory into NUM_COLLABORATORS 

SOURCE=${1}  # The directory where the BraTS dataset is located (e.g. ~/data/MICCAI_BraTS2020_TrainingData)
DESTINATION=${2}   # The destination directory for the randomized, split training data folders
NUM_COLLABORATORS=${3:-2}  # The number of collaborator splits for the subdirectories

help() {
    echo
    echo "======================================================================="
    echo "~$ split_directory.sh BRATS_DATA_SOURCE_DIRECTORY DESTINATION_DIRECTORY"
    echo "======================================================================="
    echo
    echo "BRATS_DATA_SOURCE_DIRECTORY: The directory where the BraTS dataset is located (e.g. ~/data/MICCAI_BraTS2020_TrainingData)"
    echo "DESTINATION DIRECTORY: The destination directory for the randomized, split training data folders (e.g. ~/brats_data_split)"
    echo "NUM_COLLABORATORS: The number of collaborator splits for the subdirectories (default: 2)"
    echo "-h, --help            display this help and exit"
    echo
    echo
}

if [ "$#" -lt 2 ] || ! [ -d ${1} ]; then
    help
    exit 1
fi

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

# Remove the destination directory if it exists
if [ -d ${DESTINATION} ] 
then
    echo "Removing existing directory." 
    rm -r ${DESTINATION}
fi

printf "Shard into ${NUM_COLLABORATORS} directories under ${DESTINATION}."
echo ' '
spin='-\|/'

n=0
i=0
# Find the subdirectories under the SOURCE directory and randomly shuffle them (seed is the same)
for f in `find ${SOURCE} -mindepth 1 -maxdepth 2 -type d | shuf --random-source=<(get_seeded_random 816)`; do

  ((n++))

  # The folder to put the folder
  idx=$((n % ${NUM_COLLABORATORS}))

  i=$(( (i+1) %4 ))
  printf "\r${spin:$i:1} ${f}"

  d=${DESTINATION}/split_${idx}/

  # Make the directory (if it doesn't exist) and copy the folder to it.
  mkdir -p ${d}
  cp -r ${f} ${d}

done

echo ' '
echo ' '
```

`~$ bash split_directory.sh ${DATA_PATH} ${NEW_PATH} ${NUMBER OF COLLABORATORS}`

where `${NEW_PATH}` is where you want to copy the original data (and split it randomly into subdirectories). The default is 2 collaborators (so 2 splits).

The new directories for the data are:
```
${NEW_PATH}
├── split_0
│   ├── BraTS20_Training_001
│   ├── BraTS20_Training_002
│   ├── BraTS20_Training_003
│   ├── ...
└── split_1
    ├── BraTS20_Training_009
    ├── BraTS20_Training_014
    ├── BraTS20_Training_015
    ├── ...
```

4. Now update the `plan/data.yaml` file to reflect the new data directories:

```
$ cat plan/data.yaml
# Copyright (C) 2020-2021 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

# all keys under 'collaborators' corresponds to a specific colaborator name the corresponding dictionary has data_name, data_path pairs.
# Note that in the mnist case we do not store the data locally, and the data_path is used to pass an integer that helps the data object
# construct the shard of the mnist dataset to be use for this collaborator.
#
# collaborator_name,data_directory_path

# You'll need to shard as necessary
# Symbolically link the ./data directory to whereever you have BraTS stored.
# e.g. ln -s ~/data/MICCAI_BraTS2020_TrainingData ./data/one

one,${NEW_PATH}/split_0
two,${NEW_PATH}/split_1

```

where you replace `${NEW_PATH}` by the new directory path

5. We are ready to train! Try executing the [Hello Federation](https://openfl.readthedocs.io/en/latest/running_the_federation.baremetal.html#hello-federation-your-first-federated-learning-training) steps. Make sure you have `openfl` installed in your Python virtual environment. All you have to do is to specify collaborator data paths to slice folders. We have combined all 'Hello Federation' steps in a single bash script, so it is easier to test:

```bash
bash tests/github/test_hello_federation.sh tf_3dunet_brats fed_work12345alpha81671 one123dragons beta34unicorns localhost --col1-data-path $NEW_PATH/split_0 --col2-data-path $NEW_PATH/$SUBFOLDER/split_1 --rounds-to-train 5
```
The result of the execution of the command above is 5 completed training rounds. 
