#!/bin/bash
set -e

git clone https://github.com/igor-davidyuk/openfl.git
cd openfl
git checkout model-proto-ll
python3.7 -m pip install --upgrade pip
python3.7 -m pip install virtualenv
python3.7 -m virtualenv --python=python3.7 env-openfl
source ./env-openfl/bin/activate
pip install -e .
pip install torchvision
cd openfl-tutorials/interactive_api_tutorials_experimental/director_kvasir/

# jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''

get best model "DONE"
envoy registry - remove envoy 
remove experiment workspace on envoy and director in case of fail (demo error) "DONE"
directors API registry
logs streaming "DONE"
remove pickling
shard descriptor `get_item` from validation set
return error to API if collaborator fails
passing envoy node info, choosing devices
assigner
send model
adding tests
workspace should be unpacked by CollaboratorManager component but not by the grpc client
we do not need to deconstruct model proto on director side, just pass it to the aggregator
add support for exporting pacages installed from github
collaborator should try to connect the same fqdn the director is running on
When the aggregator fails director should notify everyone
How long certificates work?

Should get_token be run only on ca node? Why we need ca_url then?
# PKI setup procedure
1. fx ca create <dir> <uri> <pass>
# cd to ca folder
2. fx ca run
3. fx ca get token <name>     * (N_envoy + director + Frontends)

# director create <dir path>
4. fx director certify <token>
# director start <config>

# fx collaborator_manager create <dir path>
5. fx envoy certify <token>    * N
# fx collaborator_manager start <config>

# ??? frontend CLI ???
6. api_layer certify <token>
# choose certs in a notebook or script

# Proposed scenario
# We move all the certs generation routine to a separate CLI component
# as the procedure is the same for all out entities + step ca has nothing 
# to do with openfl entities
1. fx pki start-ca <dir> <uri> <pass>
# checks if there are binaries and creates if not

# This is run on ca node
2. fx pki get_token     * (N_envoy + director + Frontends)
# get_token Procedure is exactly the same for all openfl actors 

3. fx pki get_certs <name> <token>    * (N_envoy + director)
# Envoys and director can get certificates using the unified CLI command

4. from step_ca import get_certs
# Frontends use python API