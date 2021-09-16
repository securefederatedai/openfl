#!/bin/bash
set -m

if [ $CONTAINER_TYPE = 'collaborator' ]
then
    tar -xf /certs.tar
    fx collaborator certify --import agg_to_col_${COL}_signed_cert.zip
    fx --log-level debug collaborator start -n ${COL}

elif [ $CONTAINER_TYPE = 'aggregator' ]
then
    tar -xf /certs.tar
    fx --log-level debug aggregator start
fi
