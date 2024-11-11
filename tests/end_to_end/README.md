# Project Title

This project is a machine learning workspace that includes various models and test suites. It is structured to facilitate the development, testing, and deployment of machine learning models.

## Project Structure

end_to_end
├── models                  # Central location for all model-related code for testing purpose
├── test_suites             # Folder containing test files
├── utils                   # Folder containing helper files
├── __init__.py             # To mark test directory as a Python package
├── conftest.py             # Pytest framework configuration file
├── pytest.ini              # Pytest initialisation file
└── README.md               # Readme file

## Pre-requisites

Setup virtual environment and install OpenFL using [online documentation](https://openfl.readthedocs.io/en/latest/get_started/installation.html).

## Installation

To install the required dependencies on above virtual environment, run:

```sh
pip install -r test-requirements.txt
```

## Usage

### Running Tests

To run all the test cases under test_suites, use the following command:

```python -m pytest -s```

To run a specific test case, use below command:

```python -m pytest test_suites/<test_case_filename> -k <marker> -s```

** -s will ensure all the logs are printed on screen. Ignore, if not required.

### Output Structure

end_to_end
├── results
    ├── <workspace_name>    # Based on the workspace name provided during test run.
    ├── results.xml         # Output file in JUNIT.
    ├── deployment.log      # Log file containing step by step test progress.

## Contribution
Please ensure that you have tested your changes thoroughly before submitting a pull request.

## License
This project is licensed under [Apache License Version 2.0](LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.
