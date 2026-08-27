#!/bin/bash

set -Eeuo pipefail

rm -rf build

pip uninstall openfl -y

pip install -e .

pip install -r test-requirements.txt

pip install -r openfl-tutorials/experimental/workflow/workflow_interface_requirements.txt

pip install coverage

pip install pytest-cov

fx experimental deactivate 

rm -rf .coverage

python -m pytest -rA --cov=openfl 

python -m pytest -s tests/end_to_end/test_suites/task_runner_tests.py -k test_federation_via_native --model_name torch/mnist --num_rounds 2 --disable_client_auth --secure_agg --cov-report=term-missing --cov-append --cov=openfl  

python -m pytest -s tests/end_to_end/test_suites/task_runner_tests.py -k test_federation_via_native --model_name keras/jax/mnist --num_rounds 2 --disable_tls --cov-report=term-missing --cov-append --cov=openfl 

python -m pytest -s tests/end_to_end/test_suites/memory_logs_tests.py -k test_log_memory_usage_basic --model_name torch/histology --num_rounds 2 --log_memory_usage --secure_agg --cov-report=term-missing --cov-append --cov=openfl  

python -m pytest -s tests/end_to_end/test_suites/tr_resiliency_tests.py --model_name keras/torch/mnist --num_rounds 25 --cov-report=term-missing --cov-append --cov=openfl 

python -m pytest -s tests/end_to_end/test_suites/tr_flower_tests.py -k test_flower_app_pytorch_native --model_name flower-app-pytorch --num_rounds 1 --cov-report=term-missing --cov-append --cov=openfl 


python -m pytest -s tests/end_to_end/test_suites/task_runner_tests.py -m task_runner_dockerized_ws --num_rounds 2 --model_name keras/torch/mnist --cov-report=term-missing --cov-append --cov=openfl 

python -m pytest -s tests/end_to_end/test_suites/tr_with_fedeval_tests.py -m task_runner_basic --model_name xgb_higgs  --num_rounds 2 --cov-report=term-missing --cov-append --cov=openfl 

python -m pytest -s tests/end_to_end/test_suites/wf_local_func_tests.py --num_rounds 2 --cov-report=term-missing --cov-append --cov=openfl 

python -m pytest -s tests/end_to_end/test_suites/wf_local_func_tests.py --workflow_backend ray --num_rounds 2 --cov-report=term-missing --cov-append --cov=openfl

fx experimental activate 

python -m pytest -s tests/end_to_end/test_suites/wf_federated_runtime_tests.py -k test_federated_runtime_301_watermarking --cov-report=term-missing  --cov-append --cov=openfl 

python -m pytest -s tests/end_to_end/test_suites/wf_federated_runtime_tests.py -k test_federated_runtime_secure_aggregation --cov-report=term-missing  --cov-append --cov=openfl 

# python -m pytest -s tests/end_to_end/test_suites/wf_federated_runtime_tests.py -k test_federated_evaluation --cov-report=term-missing  --cov-append --cov=openfl 

fx experimental deactivate 
# Combine and generate the final coverage report
coverage report






