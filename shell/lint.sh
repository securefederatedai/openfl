#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

# Run the pre-commit checks
pre-commit run --all-files

ruff check --config "${base_dir}/pyproject.toml" openfl/

ruff format --check --config "${base_dir}/pyproject.toml" openfl/