# Federated Analytics: Smokers Health Example

This workspace demonstrates how to use OpenFL for privacy-preserving analytics on the Smokers Health dataset. The setup enables distributed computation of health statistics (such as heart rate, cholesterol, and blood pressure) across multiple collaborators, without sharing raw data.

## Instantiating a Workspace from Smokers Health Template
To instantiate a workspace from the `federated_analytics/smokers_health` template, use the `fx workspace create` command. This will set up a new workspace with the required configuration and code.

1. **Install dependencies:**
```bash
pip install virtualenv
mkdir ~/openfl-smokers-health
virtualenv ~/openfl-smokers-health/venv
source ~/openfl-smokers-health/venv/bin/activate
pip install openfl
```

2. **Create the Workspace Folder:**
```bash
cd ~/openfl-smokers-health
fx workspace create --template federated_analytics/smokers_health --prefix fl_workspace
cd ~/openfl-smokers-health/fl_workspace
```

## Directory Structure
The workspace has the following structure:
```
smokers_health
├── requirements.txt
├── .workspace
├── plan
│   ├── plan.yaml
│   ├── cols.yaml
│   ├── data.yaml
│   └── defaults/
├── src
│   ├── __init__.py
│   ├── dataloader.py
│   ├── taskrunner.py
│   └── aggregate_health.py
├── data/
└── save/
```

### Directory Breakdown
- **requirements.txt**: Lists all Python dependencies for the workspace.
- **plan/**: Contains configuration files for the federation:
    - `plan.yaml`: Main plan declaration.
    - `cols.yaml`: List of authorized collaborators.
    - `data.yaml`: Data path for each collaborator.
    - `defaults/`: Default configuration values.
- **src/**: Python modules for federated analytics:
    - `dataloader.py`: Loads and shards the Smokers Health dataset, supports SQL queries.
    - `taskrunner.py`: Groups data and computes mean health metrics by age, sex, and smoking status.
    - `aggregatehealth.py`: Aggregates results from all collaborators.
- **data/**: Place to store the downloaded and unzipped dataset.
- **save/**: Stores aggregated results and analytics outputs.

## Data Preparation
The data loader will automatically download the Smokers Health dataset from Kaggle or a specified source. Make sure you have the required access or download the dataset manually if needed.

## Defining the Data Loader
The data loader supports SQL-like queries and can load data from CSV or other sources as configured. It shards the dataset among collaborators and provides query functionality for analytics tasks.

## Defining the Task Runner
The task runner groups the data by `age`, `sex`, and `current_smoker`, and computes the mean of `heart_rate`, `chol`, and `blood pressure (systolic/diastolic)`. The results are returned as numpy arrays for aggregation.

## Running the Federation
1. **Initialize the plan:**
```bash
fx plan initialize
```
2. **Set up the aggregator and collaborators:**
```bash
fx workspace certify
fx aggregator generate-cert-request
fx aggregator certify --silent

fx collaborator create -n collaborator1 -d 1
fx collaborator generate-cert-request -n collaborator1
fx collaborator certify -n collaborator1 --silent

fx collaborator create -n collaborator2 -d 2
fx collaborator generate-cert-request -n collaborator2
fx collaborator certify -n collaborator2 --silent
```
3. **Start the federation:**
```bash
fx aggregator start &
fx collaborator start -n collaborator1 &
fx collaborator start -n collaborator2 &
```

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.