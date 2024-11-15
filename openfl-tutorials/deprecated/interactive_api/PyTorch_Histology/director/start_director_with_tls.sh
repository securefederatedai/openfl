#!/bin/bash
set -e
FQDN=$1
fx director start -c director_config.yaml -rc cert/root_ca.crt -pk cert/"${FQDN}".key -oc cert/"${FQDN}".crt