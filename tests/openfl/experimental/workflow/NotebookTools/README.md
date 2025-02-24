# Objective

Validate `NotebookTools.export()` and `NotebookTools.export_federated()` APIs that are used to convert the JupyterNotebook into Workflow API experiments


# Test Structure

```
tests/openfl/experimental/workflow/NotebookTools

├── test_export
│   ├── test_artifacts    # Actual output of the testcase is generated when the test is executed.
│   │   └── expected      # Expected output to compare with actual output which is predefined and stored
│   ├── test_101_MNIST    # Notebook used for testing 
│   └── test_export.py    # test script file to run the tests
├── test_export_federated
│   ├── test_artifacts    # Actual output of the testcase is generated when the test is executed.
│   │   └── expected      # Expected output to compare with actual output which is predefined and stored
│   ├── test_MNIST_Watermarking  # Notebook used for testing 
│   └── test_export_federated.py    # test script file to run the tests
├── READ.md               # Readme File
```

## Usage

Ensure that pytest and all dependencies for Workflow Interface are installed in virtual environment


- For running `test_export`

To run a specific test case, use below command:

```sh
pytest -s tests/openfl/experimental/workflow/NotebookTools/testcase_export
```

- For running `test_export_federated`


To run a specific test case, use below command:

```sh
pytest -s tests/openfl/experimental/workflow/NotebookTools/testcase_export_federated
```