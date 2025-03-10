import os
import sys
import hashlib
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from PIL import Image
import numpy as np
from tqdm import tqdm

EXPECTED_HASHES = {
    '0': 'a9ce765efae9d00f834a8fb0f27c3ff416c0b0a9619e57b0d766ff2788dc8cf5618397fc1be166df4585c1e2fc907893',
    '1': '924f826e13a393970d50c134a24f68fa9b0aa4c59fe9927cbbc0ef4ea64611d56f8e0d85168afc4b4cf6e81cfc5f165e'
}

def verify_data_hash(partition_train_test, expected_hash):
    """Verify the hash of data in memory."""
    calculated_hash = hash_dataset_dict(partition_train_test)
    if calculated_hash != expected_hash:
        raise ValueError(f'Hash mismatch: {calculated_hash} != {expected_hash}')
    print(f"Partition hash verification successful.")

def hash_dataset_dict(dataset_dict):
    """Hash the contents of a DatasetDict."""
    hash_obj = hashlib.sha384()
    for split, dataset in sorted(dataset_dict.items()):
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

def main(num_partitions):
    # Directory to save the partitions
    save_dir = 'data'

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Initialize FederatedDataset
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )

    # Download, split, and save the dataset
    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

        # Hash the DatasetDict
        verify_data_hash(partition_train_test, EXPECTED_HASHES[str(partition_id)])

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