## Overview
This workspace template showcases the use of `VerifiableImageFolder` among a federation of two collaborators with a diverse mix of data sources (`LocalDataSource` and `S3DataSource`). The data source JSON descriptors are located under `./data/collaborator1` and `./data/collaborator2`. In a distributed setup, those would be located on separate machines at each collaborator's premises, to make the entire experiment executable locally.

It is important to note that the integrity verification is done based on a hash of each dataset that is calculated _prior_ to the experiment via the `fx collaborator calchash` command. Doing so provides protection against various data integrity attacks that could occur between the dataset preparation and the actual federated learning process.

## Steps to run the experiment
1. Set up a MinIO server which hosts the S3 data sources from the JSON descriptors
2. Configure the credentials
    ```shell
    export YOUR_ACCESS_KEY=<your_access_key>
    export YOUR_SECRET_KEY=<your_secret_key>
    ```

    For example:
    ```shell
    export MINIO_ROOT_PASSWORD=minioadmin
    export MINIO_ROOT_USER=minioadmin
    ```
3. Create a workspace folder from the `histology_s3` template
    ```shell
    fx workspace create --prefix ~/hist_s3 --template torch/histology_s3
    cd ~/hist_s3/
    pip install -r requirements.txt
    ```
4. Optional: Calculate the dataset hashes for both collaborators.
The hash will be saved at `<data_path>/hash.txt`. Later on, when the Federated Learning process starts, the data loader will check for this file and verify the dataset's integrity against the hash stored inside:
    ```shell
    fx collaborator calchash --data_path plan/collaborator1
    fx collaborator calchash --data_path plan/collaborator2
    ```
5. For the rest of the steps, follow the [quickstart guide](https://openfl.readthedocs.io/en/latest/tutorials/taskrunner.html)
