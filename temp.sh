rm -rf build/ my_federation/ $(find /home/ -type d -name .metaflow)

python -m pip install .

export WORKSPACE_PATH=$(pwd)/my_federation
export WORKSPACE_TEMPLATE=$1

fx experimental activate

fx workspace create --prefix ${WORKSPACE_PATH} --template ${WORKSPACE_TEMPLATE}

cd ${WORKSPACE_PATH}

pip install -r requirements.txt

fx plan initialize -a localhost

fx workspace certify

fx aggregator generate-cert-request --fqdn localhost

fx aggregator certify --fqdn localhost

fx workspace export

# # 1st collaborator
cd ${WORKSPACE_PATH}
mkdir col1
cd col1

# echo "==========================================================================================="
fx workspace import --archive ../my_federation.zip

cd my_federation

fx collaborator create -n col1

fx collaborator generate-cert-request -n col1

cd ../../

# # On Aggregator Node
fx collaborator certify --request-pkg ${WORKSPACE_PATH}/col1/my_federation/col_col1_to_agg_cert_request.zip

cd ${WORKSPACE_PATH}/col1/my_federation

fx collaborator certify --import ${WORKSPACE_PATH}/agg_to_col_col1_signed_cert.zip

cd ../..

# # 2nd collaborator
cd ${WORKSPACE_PATH}
mkdir col2
cd col2

fx workspace import --archive ../my_federation.zip

cd my_federation

fx collaborator create -n col2

fx collaborator generate-cert-request -n col2

cd ../../

# # On Aggregator Node
fx collaborator certify --request-pkg ${WORKSPACE_PATH}/col2/my_federation/col_col2_to_agg_cert_request.zip

cd ${WORKSPACE_PATH}/col2/my_federation

fx collaborator certify --import ${WORKSPACE_PATH}/agg_to_col_col2_signed_cert.zip

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # #    If executing testcase private attributes 
# # #    uncomment col3, and col4
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # # 3rd collaborator
# cd ${WORKSPACE_PATH}
# mkdir col3
# cd col3

# fx workspace import --archive ../my_federation.zip

# cd my_federation

# fx collaborator create -n col3

# fx collaborator generate-cert-request -n col3

# cd ../../

# # # On Aggregator Node
# fx collaborator certify --request-pkg ${WORKSPACE_PATH}/col3/my_federation/col_col3_to_agg_cert_request.zip

# cd ${WORKSPACE_PATH}/col3/my_federation

# fx collaborator certify --import ${WORKSPACE_PATH}/agg_to_col_col3_signed_cert.zip

# cd ../..

# # # 4th collaborator
# cd ${WORKSPACE_PATH}
# mkdir col4
# cd col4

# fx workspace import --archive ../my_federation.zip

# cd my_federation

# fx collaborator create -n col4

# fx collaborator generate-cert-request -n col4

# cd ../../

# # # On Aggregator Node
# fx collaborator certify --request-pkg ${WORKSPACE_PATH}/col4/my_federation/col_col4_to_agg_cert_request.zip

# cd ${WORKSPACE_PATH}/col4/my_federation

# fx collaborator certify --import ${WORKSPACE_PATH}/agg_to_col_col4_signed_cert.zip

cd ${WORKSPACE_PATH}

fx aggregator start
