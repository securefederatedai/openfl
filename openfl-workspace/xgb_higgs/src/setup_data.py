import sys
import os
import shutil
from logging import getLogger
from urllib.request import urlretrieve
from hashlib import sha384
from os import path, makedirs
from tqdm import tqdm
import modin.pandas as pd
import gzip
from sklearn.model_selection import train_test_split
import numpy as np

logger = getLogger(__name__)

"""HIGGS Dataset."""

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
FILENAME = "HIGGS.csv.gz"
CSV_FILENAME = "HIGGS.csv"
CSV_SHA384 = 'b8b82e11a78b81601381420878ad42ba557291f394a88dc5293e4077c8363c87429639b120e299a2a9939c1f943b6a63'
DEFAULT_PATH = path.join(os.getcwd(), 'data')

pbar = tqdm(total=None)

def report_hook(count, block_size, total_size):
    """Update progressbar."""
    if pbar.total is None and total_size:
        pbar.total = total_size
    progress_bytes = count * block_size
    pbar.update(progress_bytes - pbar.n)

def verify_sha384(file_path, expected_hash):
    """Verify the SHA-384 hash of a file."""
    sha384_hash = sha384()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha384_hash.update(byte_block)
    computed_hash = sha384_hash.hexdigest()
    if computed_hash != expected_hash:
        raise ValueError(f"SHA-384 hash mismatch: expected {expected_hash}, got {computed_hash}")
    print(f"SHA-384 hash verified: {computed_hash}")

def setup_data(root: str = DEFAULT_PATH, **kwargs):
    """Initialize."""
    makedirs(root, exist_ok=True)
    filepath = path.join(root, FILENAME)
    csv_filepath = path.join(root, CSV_FILENAME)
    if not path.exists(filepath):
        urlretrieve(URL, filepath, report_hook)  # nosec
        verify_sha384(filepath, CSV_SHA384)
        # Extract the CSV file from the gzip file
        with gzip.open(filepath, 'rb') as f_in:
            with open(csv_filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

def main():
    if len(sys.argv) < 2:
        raise ValueError("Provide the number of collaborators")
    src = 'higgs_data'
    if os.path.exists(src):
        shutil.rmtree(src)
    setup_data(src)
    collaborators = int(sys.argv[1])
    print("Creating splits for {} collaborators".format(collaborators))

    # Load the dataset
    higgs_data = pd.read_csv(path.join(src, CSV_FILENAME), header=None)

    # Split the dataset into features and labels
    X = higgs_data.iloc[:, 1:].values
    y = higgs_data.iloc[:, 0].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Combine X and y for train and test sets
    train_data = pd.DataFrame(data=np.column_stack((y_train, X_train)))
    test_data = pd.DataFrame(data=np.column_stack((y_test, X_test)))

    # Split the training data into parts for each collaborator
    for i in range(collaborators):
        dst = f'data/{i+1}'
        makedirs(dst, exist_ok=True)

        # Split the training data for the current collaborator
        split_train_data = train_data.iloc[i::collaborators]
        split_train_data.to_csv(path.join(dst, 'train.csv'), index=False, header=False)

        # Split the test data for the current collaborator
        split_test_data = test_data.iloc[i::collaborators]
        split_test_data.to_csv(path.join(dst, 'valid.csv'), index=False, header=False)

if __name__ == '__main__':
    main()
