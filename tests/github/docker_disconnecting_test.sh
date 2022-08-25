#!/bin/bash
set -e

# PREREQUISITES: you should have openfl installed + this script should be run from the openfl repo base folder

# 1 - Start from building the base image; 2 - start from creating a new workspace and dockerizing it;
# 3 - Certify federation, choose any number of collaborators; 4 - run existing containers

STAGE=${1:-1}  
NUMBER_OF_COLS=${2:-1}  
TEMPLATE=${3:-'keras_cnn_mnist'}  # ['torch_cnn_mnist', 'keras_cnn_mnist']
CUT_ON_LOG=${4:-"Run 0 epoch of 2 round"} 
RECONNECTION_TIMEOUT=${4:-10} 

BASE_IMAGE_TAG='openfl'

# If an aggregator container will run on another machine
# a relevant FQDN should be provided
FQDN='agg'
FED_WORKSPACE='test-federation'   # This can be whatever unique directory name you want
WORSPACE_IMAGE_NAME=${FED_WORKSPACE}
AGGREGATOR_REQUIRED_FILES='cert_agg.tar'

if [ $STAGE -le 1 ]; then
        echo "BUILDING OPENFL BASE IMAGE"
        # 1. Build base image
        bash ./scripts/build_base_docker_image.sh ${BASE_IMAGE_TAG}
fi

if [ $STAGE -le 2 ]; then
        echo "CREATING EXPERIMENT WORKSPACE"
        # 2. Create the workspace
        # Create FL workspace
        rm -rf ${FED_WORKSPACE}
        fx workspace create --prefix ${FED_WORKSPACE} --template ${TEMPLATE}
        cd ${FED_WORKSPACE}
        FED_DIRECTORY=`pwd`  # Get the absolute directory path for the workspace

        # Initialize FL plan
        fx plan initialize -a ${FQDN}

        # Build the workspace image and save it to a tarball

        # This commant builds an image tagged $FED_WORKSPACE
        # Then it saves it to a ${FED_WORKSPACE}_image.tar
        fx workspace dockerize --base_image ${BASE_IMAGE_TAG} --no-save
fi

if [ $STAGE -le 3 ]; then
        echo "CERTIFYING ACTORS"
        cd ${FED_WORKSPACE} || true
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
        for ((i=1; i<=$NUMBER_OF_COLS; i++)); do
                COL_NAME="col$i"
                DATA_PATH=$i
                create_signed_cert_for_collaborator ${COL_NAME} ${DATA_PATH} 
        done

        # Also perform certificate generation for the aggregator.
        # Create aggregator certificate
        fx aggregator generate-cert-request --fqdn ${FQDN}
        # Sign aggregator certificate
        fx aggregator certify --fqdn ${FQDN} --silent # Remove '--silent' if you run this manually

        # Pack all files that aggregator need to start training
        tar -cf ${AGGREGATOR_REQUIRED_FILES} plan/ cert/ save/
fi

process_stream() {
        PATTERN=$1
        RECONNECTION_TIMEOUT=$2

        while read line
        do
                if [[ $line == *$PATTERN* ]]; then
                        docker network disconnect ${FED_WORKSPACE} ${COL_NAME} && printf "\n\n ++++++ DISCONNECTING ++++++ \n\n"
                        echo $line >> triggers.log
                        sleep $RECONNECTION_TIMEOUT && docker network connect ${FED_WORKSPACE} ${COL_NAME} &
                fi
                # echo $line
        done
}

if [ $STAGE -le 4 ]; then
        echo "RUNNING FEDERATION IN DOCKER CONTAINERS"
        # 4. Start federation in containers
        cd ${FED_WORKSPACE} || true

        # Create a docker network
        docker network create ${FED_WORKSPACE} || true


        # Start the aggregator
        docker run --rm \
                --network ${FED_WORKSPACE} \
                -v $(pwd)/${AGGREGATOR_REQUIRED_FILES}:/certs.tar \
                -e "CONTAINER_TYPE=aggregator" \
                --name ${FQDN} \
                ${WORSPACE_IMAGE_NAME} \
                bash /openfl/openfl-docker/start_actor_in_container.sh &
                # --add-host ${FQDN}:127.0.0.1 \ adding this mapping to the aggregator container is not required somehow

        # Start the collaborator
        for ((i=1; i<=$NUMBER_OF_COLS; i++)); do
                COL_NAME='col'$i
                docker run --rm \
                        --network ${FED_WORKSPACE} \
                        -v $(pwd)/cert_col_${COL_NAME}.tar:/certs.tar \
                        -e "CONTAINER_TYPE=collaborator" \
                        -e "no_proxy=${FQDN}" \
                        -e "COL=${COL_NAME}" \
                        --name ${COL_NAME} \
                        ${WORSPACE_IMAGE_NAME} \
                        bash /openfl/openfl-docker/start_actor_in_container.sh | tee /dev/tty | process_stream "$CUT_ON_LOG" $RECONNECTION_TIMEOUT
        done
fi



# sleep 20
# printf "\n\n ++++++ DISCONNECTING ++++++ \n\n"

# docker network disconnect ${FED_WORKSPACE} ${COL_NAME}

# sleep 10
# printf "\n\n ++++++ CONNECTING ++++++ \n\n"

# docker network connect ${FED_WORKSPACE} ${COL_NAME}

wait
docker network rm ${FED_WORKSPACE}