#!/bin/bash
set -e

# 1. Create the workspace

TEMPLATE=${1:-'keras_cnn_mnist'}  # ['torch_cnn_mnist', 'keras_cnn_mnist']
FED_WORKSPACE=${2:-'fed_work12345alpha81671'}   # This can be whatever unique directory name you want
COL=${3:-'one123dragons'}  # This can be any unique label (lowercase)
DATA_PATH=${4:-1}
BASE_IMAGE_TAG=${5:-'openfl'}

# If an aggregator container will run on another machine
# a relevant FQDN should be provided
FQDN=${6:-$(hostname --all-fqdns | awk '{print $1}')}

# Build base image
bash ./scripts/build_base_docker_image.sh ${BASE_IMAGE_TAG}

# Create FL workspace
rm -rf ${FED_WORKSPACE}
fx workspace create --prefix ${FED_WORKSPACE} --template ${TEMPLATE}
cd ${FED_WORKSPACE}
FED_DIRECTORY=`pwd`  # Get the absolute directory path for the workspace

# Initialize FL plan
fx plan initialize -a ${FQDN}

# 2. Build the workspace image and save it to a tarball

# This commant builds an image tagged $FED_WORKSPACE
# Then it saves it to a ${FED_WORKSPACE}_image.tar
fx workspace dockerize --base_image ${BASE_IMAGE_TAG}

# We remove the base OpenFL image as well
# as built workspace image to simulate starting 
# on another machine
WORSPACE_IMAGE_NAME=${FED_WORKSPACE}
docker image rm -f ${BASE_IMAGE_TAG} ${WORSPACE_IMAGE_NAME}


# 3. Generate certificates for the aggregator and the collaborator

# Create certificate authority for the workspace
fx workspace certify

# We do certs exchage for all participants in a single workspace
# to speed up this test run.
# Do not do this in real experiments
# in untrusted environments
create_signed_cert_for_collaborator() {
        COL_NAME=$1
        DATA_PATH=$2
        echo "certifying collaborator $COL_NAME with data path $DATA_PATH"
        # Create collaborator certificate request
        fx collaborator generate-cert-request -d ${DATA_PATH} -n ${COL_NAME} --silent
        # Sign collaborator certificate 
        fx collaborator certify --request-pkg col_${COL_NAME}_to_agg_cert_request.zip --silent

        # Pack the collaborators private key and the signed cert
        # as well as it's data.yaml to a tarball
        tar -cf cert_col_${COL_NAME}.tar plan/data.yaml \
                cert/client/*key agg_to_col_${COL_NAME}_signed_cert.zip --remove-files 

        # Remove request archive
        rm -rf col_${COL_NAME}_to_agg_cert_request.zip
}

# Prepare a tarball with the collab's private key, the singed cert,
# and data.yaml for collaborator container
# This step can be repeated for each collaborator
create_signed_cert_for_collaborator ${COL} ${DATA_PATH} 

# Also perform certificate generation for the aggregator.
# Create aggregator certificate
fx aggregator generate-cert-request --fqdn ${FQDN}
# Sign aggregator certificate
fx aggregator certify --fqdn ${FQDN} --silent # Remove '--silent' if you run this manually

# Pack all files that aggregator need to start training
AGGREGATOR_REQUIRED_FILES='cert_agg.tar'
tar -cf ${AGGREGATOR_REQUIRED_FILES} plan/ cert/ save/ --remove-files 


# 4. Load the image
IMAGE_TAR=${FED_WORKSPACE}_image.tar
docker load --input $IMAGE_TAR


# 5. Start federation in containers

# Start the aggregator
docker run --rm \
        --network host \
        -v $(pwd)/${AGGREGATOR_REQUIRED_FILES}:/certs.tar \
        -e "CONTAINER_TYPE=aggregator" \
        ${WORSPACE_IMAGE_NAME} \
        bash /openfl/openfl-docker/start_actor_in_container.sh &

# Start the collaborator
docker run --rm \
        --network host \
        -v $(pwd)/cert_col_${COL_NAME}.tar:/certs.tar \
        -e "CONTAINER_TYPE=collaborator" \
        -e "COL=${COL_NAME}" \
        ${WORSPACE_IMAGE_NAME} \
        bash /openfl/openfl-docker/start_actor_in_container.sh 

# If containers are started but collaborator will fail to 
# conect the aggregator, the pipeline will go to the infinite loop
