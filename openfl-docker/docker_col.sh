#!/bin/bash
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

## PARAMETER SETTING
# openfl params
MODE=${1:-'import'}                # ['import', 'start']
WORKSPACE_DIR=${2:-'workspace'}    # This can be whatever unique$
COL=${3:-'one123dragons'}          # This can be any unique label
FED_PATH=${4:-'/home'}      # FED_WORKSPACE Path


if [ ! -z "$FED_PATH" ]; then
   cd ${FED_PATH}
fi



# Methods 
init() {

    ## Agg will need to validate the outcome of this method ##

    FED_WORKSPACE=$1
    FED_PATH=$2
    COL=$3
    DATA_PATH=$4

    CURRENT_DIR=`pwd`
    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH
    cd $FED_WORKSPACE

    ## CREATING COLLABORATORs
    fx collaborator generate-cert-request -d ${DATA_PATH} -n ${COL} --silent # Remove '--silent' if you run this manually

    # Move back to initial dir
    cd $CURRENT_DIR
}



import_ws() {

    FED_WORKSPACE=$1
    FED_PATH=$2

    ARCHIVE_NAME="${FED_WORKSPACE}.zip"
   
    CURRENT_DIR=`pwd`
    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH

    fx workspace import --archive ${ARCHIVE_NAME} # Import the workspace to this collaborator

    # Move back to initial dir
    cd $CURRENT_DIR
}


import_crt() {

    FED_WORKSPACE=$1
    FED_PATH=$2

    ARCHIVE_NAME="agg_to_col_${COL}_signed_cert.zip"

    CURRENT_DIR=`pwd`
    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH/${FED_WORKSPACE}

    fx collaborator certify --import ${ARCHIVE_NAME}
    #yes | cp -rf ${FED_PATH}/cert/* ${FED_WORKSPACE}/cert/.

    # Move back to initial dir
    cd $CURRENT_DIR
}


start() {

    FED_WORKSPACE=$1
    FED_PATH=$2
    COL=$3

    CURRENT_DIR=`pwd`

    [[ ! -z "$FED_PATH" ]] && cd $FED_PATH
    cd $FED_WORKSPACE
    C=`pwd`

    fx collaborator start -n $COL

    # Move back to initial dir
    cd $CURRENT_DIR
}



if [ "$MODE" == "init" ]; then

    init $WORKSPACE_DIR $FED_PATH $COL 1

elif [ "$MODE" == "import_ws" ]; then

    import_ws $WORKSPACE_DIR $FED_PATH

elif [ "$MODE" == "import_crt" ]; then

    import_crt $WORKSPACE_DIR $FED_PATH


elif [ "$MODE" == "start" ]; then

    start $WORKSPACE_DIR $FED_PATH $COL

else

    echo "Unrecognized Mode. Aborting"

fi
