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