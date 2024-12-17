# End-to-end Pytest Framework

This project aims at integration testing of ```openfl-workspace``` using pytest framework.

## Test Structure

```
tests/end_to_end
├── models                  # Central location for all model-related code for testing purpose
├── test_suites             # Folder containing test files
├── utils                   # Folder containing fixture and helper files
├── conftest.py             # Pytest framework configuration file
├── pytest.ini              # Pytest initialisation file
└── README.md               # Readme file
```

** File `sample_tests.py` provided under `test_suites` acts as a reference on how to add a new test case.

## Pre-requisites

1. Setup virtual environment and install OpenFL using [online documentation](https://openfl.readthedocs.io/en/latest/get_started/installation.html).
2. Ensure that the OpenFL workspace (inside openfl-workspace) is present for the model being tested. If not, create it first.

## Installation

To install the required dependencies on above virtual environment, run:

```sh
pip install -r test-requirements.txt
```

## Usage

### Running Tests

To run a specific test case, use below command:

```sh
python -m pytest -s tests/end_to_end/test_suites/<test_case_filename> -k <test_case_name>
```

** -s will ensure all the logs are printed on screen. Ignore, if not required.

Below parameters are available for modification:

1. --num_collaborators <int>   - to modify the number of collaborators
2. --num_rounds <int>          - to modify the number of rounds to train
3. --model_name <str>          - to use a specific model
4. --disable_tls               - to disable TLS communication (by default it is enabled)
5. --disable_client_auth       - to disable the client authentication (by default it is enabled)

For example, to run Task runner with - torch_cnn_mnist model, 3 collaborators, 5 rounds and non-TLS scenario:

```sh
python -m pytest -s tests/end_to_end/test_suites/task_runner_tests.py --num_rounds 5 --num_collaborators 3 --model_name torch_cnn_mnist --disable_tls
```

### Output Structure

```
results
    ├── <workspace_name>    # Same as model name used for testing.
        ├── aggregator
            ├── workspace   # containing aggregator specific files and folders
        ├── collaborator1
            ├── workspace   # containing collaborator1 specific files and folders
        ├── ....
        ├── collaborator<n>
            ├── workspace   # containing collaborator<n> specific files and folders
    ├── results.xml         # Output file in JUNIT.
    └── deployment.log      # Log file containing step by step test progress.
```
Folders excluded for all the participants - cert and data.

## Contribution

https://github.com/securefederatedai/openfl/blob/develop/CONTRIBUTING.md
