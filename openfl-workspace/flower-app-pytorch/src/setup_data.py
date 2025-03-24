import os
import sys
import hashlib
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from PIL import Image
import numpy as np
from tqdm import tqdm

EXPECTED_DATASET_HASH = 'ee5342b063299eda95304b0e0388e403a3c65ce5a07cabea4c62a3efdb419a5983c7bb204e55d771ae643642ed440182'

def verify_data_hash(dataset, expected_hash):
    """Verify the hash of the entire dataset."""
    calculated_hash = hash_dataset(dataset)
    if calculated_hash != expected_hash:
        raise ValueError(f'Hash mismatch: {calculated_hash} != {expected_hash}')
    print("Dataset hash verification successful.")

def hash_dataset(dataset):
    """Hash the contents of a Dataset."""
    hash_obj = hashlib.sha384()
    for example in dataset:
        img_array = np.array(example['img'])
        label = example['label']
        # Convert image array and label to bytes
        img_bytes = img_array.tobytes()
        label_bytes = bytes([label])
        # Update hash with image and label bytes
        hash_obj.update(img_bytes)
        hash_obj.update(label_bytes)
    return hash_obj.hexdigest()

def load_and_verify_dataset(dataset_name, expected_hash):
    """Load the entire dataset and verify its hash."""
    # Initialize FederatedDataset with a dummy partitioner
    dummy_partitioner = IidPartitioner(num_partitions=1)  # Dummy partitioner
    fds = FederatedDataset(
        dataset=dataset_name,
        partitioners={"train": dummy_partitioner},
    )

    # Load the entire dataset
    full_dataset = fds.load_partition(0)  # Load the entire dataset as a single partition

    # Hash the entire dataset
    verify_data_hash(full_dataset, expected_hash)


def main(num_partitions):
    # Directory to save the partitions
    save_dir = 'data'

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load and verify the entire dataset
    load_and_verify_dataset("uoft-cs/cifar10", EXPECTED_DATASET_HASH)

    # Initialize FederatedDataset with the actual partitioner
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )

    # Download, split, and save the dataset
    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

        # Save partition data
        partition_dir = os.path.join(save_dir, f"{partition_id+1}")
        os.makedirs(partition_dir, exist_ok=True)

        for split, dataset in partition_train_test.items():
            split_dir = os.path.join(partition_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            for idx, example in enumerate(tqdm(dataset, desc=f"Processing partition {partition_id+1}, {split}")):
                img_array = np.array(example['img'])
                label = example['label']
                label_dir = os.path.join(split_dir, str(label))
                os.makedirs(label_dir, exist_ok=True)

                # Save the image
                img = Image.fromarray(img_array)
                img_path = os.path.join(label_dir, f"{idx}.png")
                img.save(img_path)

    print("Dataset downloaded, verified, split, and saved successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python setup_data.py <num_partitions>")
        sys.exit(1)

    num_partitions = int(sys.argv[1])
    main(num_partitions)
