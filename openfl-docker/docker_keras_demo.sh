#!/bin/bash
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

HOST_USER="$(whoami)"

### VAR definition
DOCKER_IMG=${1:-""}			# name of docker image built through "fx workspace dockerize" cmd 
HOST_WORKSPACE=${2:-''}

if [  -z "$HOST_WORKSPACE" ]; then
   HOST_WORKSPACE=/home/${HOST_USER}
fi

## openfl var
WORKSPACE_DIR=${3:-'workspace'}  # This can be whatever unique directory name you want
COL=${4:-'one123dragons'} 			# This can be any unique label
FED_PATH=${5:-'/home'}    		# Federation workspace PATH within Docker

## Local var
HOST_AGG=${HOST_WORKSPACE}/host_agg_workspace
HOST_COL=${HOST_WORKSPACE}/host_col_workspace

HOST_AGG_DATA=${HOST_AGG}/data
HOST_COL_DATA=${HOST_COL}/data
HOST_AGG_CERT=${HOST_AGG}/cert
HOST_COL_CERT=${HOST_COL}/cert

AGGREGATOR_IMG_NAME="aggregator"
COLLABORATOR_IMG_NAME=${COL}


## Prepare working env mkdir -p $HOST_WORKSPACE/host_agg_workspace
mkdir -p ${HOST_AGG_DATA}
mkdir -p ${HOST_COL_DATA}
mkdir -p ${HOST_AGG_CERT}
mkdir -p ${HOST_COL_CERT}


### Spin-up instances
docker run -d --name ${AGGREGATOR_IMG_NAME} --network=host -v ${HOST_AGG_DATA}:/home/workspace/data ${DOCKER_IMG}
docker run -d --name ${COLLABORATOR_IMG_NAME} --network=host -v ${HOST_COL_DATA}:/home/workspace/data ${DOCKER_IMG}


### AGGREGATOR
## Init workspace
docker exec ${AGGREGATOR_IMG_NAME} /bin/bash -c "bash docker_agg.sh init"

### COLLABORATOR
## Import workspace
docker exec ${COLLABORATOR_IMG_NAME} /bin/bash -c "bash docker_col.sh init"

## Send COLLABORATOR request to AGGREGATOR
docker cp ${COLLABORATOR_IMG_NAME}:/home/workspace/col_${COL}_to_agg_cert_request.zip ${HOST_COL_CERT}
docker cp ${HOST_COL_CERT}/col_${COL}_to_agg_cert_request.zip ${AGGREGATOR_IMG_NAME}:/home/workspace/cert/.


### AGGREGATOR
## Certify collaborator
docker exec ${AGGREGATOR_IMG_NAME} /bin/bash -c "bash docker_agg.sh col"

## Send verified certificate from AGGREGATOR to COLLABORATOR
docker cp ${AGGREGATOR_IMG_NAME}:/home/workspace/agg_to_col_${COL}_signed_cert.zip ${HOST_AGG_CERT}
docker cp ${HOST_AGG_CERT}/agg_to_col_${COL}_signed_cert.zip ${COLLABORATOR_IMG_NAME}:/home/workspace/.


### COLLABORATOR
## Import certificate
docker exec ${COLLABORATOR_IMG_NAME} /bin/bash -c "bash docker_col.sh import_crt"

### AGGREGATOR
## Start the aggregator
docker exec -d ${AGGREGATOR_IMG_NAME} /bin/bash -c "bash docker_agg.sh start"

### COLLABORATOR
## Start the collaborator
docker exec ${COLLABORATOR_IMG_NAME} /bin/bash -c "bash docker_col.sh start"


## Stop and exit the services
docker stop ${AGGREGATOR_IMG_NAME} ${COLLABORATOR_IMG_NAME} && docker rm ${AGGREGATOR_IMG_NAME} ${COLLABORATOR_IMG_NAME}

