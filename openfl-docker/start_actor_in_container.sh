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
    fx aggregator certify --fqdn ${FQDN} --silent
    fx --log-level debug aggregator start
fi

fg %1