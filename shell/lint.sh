#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

# TODO: @karansh1 Apply across all modules
isort --sp "${base_dir}/pyproject.toml" --check openfl/experimental

black --config "${base_dir}/pyproject.toml" --check openfl/experimental

flake8 --config "${base_dir}/setup.cfg" openfl/experimental