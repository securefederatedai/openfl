# End-to-end Pytest Framework

This project aims at integration testing of ```openfl-workspace``` using pytest framework.

## Test Structure

```
tests/end_to_end
├── models                  # Central location for all model-related code for testing purpose
├── test_suites             # Folder containing test files
├── utils                   # Folder containing helper files
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
python -m pytest tests/end_to_end/test_suites/<test_case_filename> -k <marker> -s
```

** -s will ensure all the logs are printed on screen. Ignore, if not required.

To modify the number of collaborators, rounds to train and/or model name, use below parameters:
1. --num_collaborators
2. --num_rounds
3. --model_name

### Output Structure

```
results
    ├── <workspace_name>    # Based on the workspace name provided during test run.
    ├── results.xml         # Output file in JUNIT.
    └── deployment.log      # Log file containing step by step test progress.
```

## Contribution
Please ensure that you have tested your changes thoroughly before submitting a pull request.

## License
This project is licensed under [Apache License Version 2.0](LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.
