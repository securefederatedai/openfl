# This test is not used due to possible dependency version conflict
# between local version and PyPI version of openfl.

set -euxo pipefail
# Test the pipeline
# =========== Set SGX_RUN variable to 0 or 1 ============

SGX_RUN=${1:-1} # Pass 0 for no-sgx run (gramine-direct)
REBUILD_IMAGES=${2:-0} # Pass 1 to build images with `--no-cache` option
TEMPLATE=${3:-'torch_unet_kvasir_gramine_ready'}  # ['torch_cnn_histology_gramine_ready', 'keras_nlp_gramine_ready']
FED_WORKSPACE=${4:-'fed_gramine'}   # This can be whatever unique directory name you want
COL1=${5:-'one'}  # This can be any unique label (lowercase)
COL2=${6:-'two'} # This can be any unique label (lowercase)

FQDN=localhost
# FQDN=${6:-$(hostname --all-fqdns | awk '{print $1}')}

COL1_DATA_PATH=1
COL2_DATA_PATH=2

# START
# =====
# Make sure you are in a Python virtual environment with the FL package installed.

# Create FL workspace
rm -rf ${FED_WORKSPACE}
fx workspace create --prefix ${FED_WORKSPACE} --template ${TEMPLATE}
cd ${FED_WORKSPACE}
FED_DIRECTORY=`pwd`  # Get the absolute directory path for the workspace

# Initialize FL plan
fx plan initialize -a ${FQDN}

openssl genrsa -3 -out ${FED_DIRECTORY}/key.pem 3072

# Build graminized app image
if [ $REBUILD_IMAGES -gt 0 ]
then
fx workspace graminize -s ${FED_DIRECTORY}/key.pem --no-save --rebuild
else
fx workspace graminize -s ${FED_DIRECTORY}/key.pem --no-save
fi

# CERTIFICATION PART------------------------------
# ================================================
create_collaborator() {

    FED_WORKSPACE=$1
    FED_DIRECTORY=$2
    COL=$3
    COL_DIRECTORY=$4
    DATA_PATH=$5

    ARCHIVE_NAME="${FED_WORKSPACE}.zip"

    # Copy workspace to collaborator directories (these can be on different machines)
    rm -rf ${COL_DIRECTORY}    # Remove any existing directory
    mkdir -p ${COL_DIRECTORY}  # Create a new directory for the collaborator
    cd ${COL_DIRECTORY}
    fx workspace import --archive ${FED_DIRECTORY}/${ARCHIVE_NAME} # Import the workspace to this collaborator

    # Create collaborator certificate request
    cd ${COL_DIRECTORY}/${FED_WORKSPACE}
    fx collaborator create -d ${DATA_PATH} -n ${COL} --silent # Remove '--silent' if you run this manually
    fx collaborator generate-cert-request -n ${COL} --silent # Remove '--silent' if you run this manually

    # Sign collaborator certificate 
    cd ${FED_DIRECTORY}  # Move back to the Aggregator
    fx collaborator certify --request-pkg ${COL_DIRECTORY}/${FED_WORKSPACE}/col_${COL}_to_agg_cert_request.zip --silent # Remove '--silent' if you run this manually

    #Import the signed certificate from the aggregator
    cd ${COL_DIRECTORY}/${FED_WORKSPACE}
    fx collaborator certify --import ${FED_DIRECTORY}/agg_to_col_${COL}_signed_cert.zip

    cp -r ${FED_DIRECTORY}/data ${COL_DIRECTORY}/${FED_WORKSPACE}

}
# Create certificate authority for workspace
fx workspace certify

# Create aggregator certificate
fx aggregator generate-cert-request --fqdn ${FQDN}

# Sign aggregator certificate
fx aggregator certify --fqdn ${FQDN} --silent # Remove '--silent' if you run this manually

# Create collaborator #1
COL1_DIRECTORY=${FED_DIRECTORY}/${COL1}
create_collaborator ${FED_WORKSPACE} ${FED_DIRECTORY} ${COL1} ${COL1_DIRECTORY} ${COL1_DATA_PATH}

# Create collaborator #2
COL2_DIRECTORY=${FED_DIRECTORY}/${COL2}
create_collaborator ${FED_WORKSPACE} ${FED_DIRECTORY} ${COL2} ${COL2_DIRECTORY} ${COL2_DATA_PATH}

# CERTIFICATION PART ENDS-------------------------
# ================================================

# # Run the federation
cd ${FED_DIRECTORY}

RUN_START="docker run --rm --detach "
if [ $SGX_RUN -gt 0 ]
then
RUN_START=${RUN_START}"--device=/dev/sgx_enclave --volume=/var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket"
else
RUN_START=${RUN_START}"--security-opt seccomp=unconfined -e GRAMINE_EXECUTABLE=gramine-direct"
fi

# fx aggregator start & 
$RUN_START \
--network=host --name Aggregator \
--volume=${FED_DIRECTORY}/cert:/workspace/cert \
--volume=${FED_DIRECTORY}/logs:/workspace/logs \
--volume=${FED_DIRECTORY}/plan/cols.yaml:/workspace/plan/cols.yaml \
--mount type=bind,src=${FED_DIRECTORY}/save,dst=/workspace/save,readonly=0 \
${FED_WORKSPACE} aggregator start

sleep 5 

# cd ${COL1_DIRECTORY}/${FED_WORKSPACE}
# fx collaborator start -n ${COL1} & 
$RUN_START \
--network=host --name ${COL1} \
--volume=${COL1_DIRECTORY}/${FED_WORKSPACE}/cert:/workspace/cert \
--volume=${COL1_DIRECTORY}/${FED_WORKSPACE}/plan/data.yaml:/workspace/plan/data.yaml \
--volume=${COL1_DIRECTORY}/${FED_WORKSPACE}/data:/workspace/data \
${FED_WORKSPACE} collaborator start -n ${COL1}

# cd ${COL2_DIRECTORY}/${FED_WORKSPACE}
# fx collaborator start -n ${COL2}
$RUN_START \
--network=host --name ${COL2} \
--volume=${COL2_DIRECTORY}/${FED_WORKSPACE}/cert:/workspace/cert \
--volume=${COL2_DIRECTORY}/${FED_WORKSPACE}/plan/data.yaml:/workspace/plan/data.yaml \
--volume=${COL2_DIRECTORY}/${FED_WORKSPACE}/data:/workspace/data \
${FED_WORKSPACE} collaborator start -n ${COL2}

# tail -f `docker inspect --format='{{.LogPath}}' Aggregator`
docker logs --follow Aggregator &
docker logs --follow ${COL1} &
docker logs --follow ${COL2}

wait

docker stop Aggregator ${COL1} ${COL2}
# rm -rf ${FED_DIRECTORY}
