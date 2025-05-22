# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
import concurrent.futures
import logging
import os
from pathlib import Path
import importlib

import tests.end_to_end.utils.constants as constants
import tests.end_to_end.utils.exceptions as ex
import tests.end_to_end.utils.federation_helper as fh
import tests.end_to_end.utils.s3_helper as s3_helper
import tests.end_to_end.utils.ssh_helper as ssh
from tests.end_to_end.models import aggregator as agg_model, model_owner as mo_model, s3_bucket as s3_model
import tests.end_to_end.utils.docker_helper as dh

log = logging.getLogger(__name__)

# Define a named tuple to store the objects for model owner, aggregator, and
# collaborators
federation_details = collections.namedtuple(
    "federation_details",
    "model_owner, aggregator, collaborators, workspace_path, local_bind_path, "
    "model_name",
)

def common_workspace_creation(request, eval_scope=False):
    """
    Common workspace creation function for task runner and dockerized workspace.

    Args:
        request (object): Pytest request object.
        eval_scope (bool, optional): If True, sets up the evaluation scope for a
        single round. Defaults to False.

    Returns:
        tuple: A tuple containing the workspace path, local bind path, aggregator
        domain name, model owner, plan path, and aggregator workspace path.
    """

    workspace_path, local_bind_path, agg_domain_name = (
        fh.federation_env_setup_and_validate(request, eval_scope)
    )

    agg_workspace_path = constants.AGG_WORKSPACE_PATH.format(workspace_path)

    # For Flower App Pytorch, num of rounds must be 1
    if request.config.model_name.lower() == constants.ModelName.FLOWER_APP_PYTORCH.value:
        if request.config.num_rounds != 1:
            raise ex.FlowerAppException(
                "Flower app with PyTorch only supports 1 round of training."
            )

    # Create model owner object and the workspace for the model
    # Workspace name will be same as the model name
    model_owner = mo_model.ModelOwner(
        request.config.model_name, request.config.log_memory_usage, workspace_path=agg_workspace_path
    )

    # Create workspace for given model name
    fh.create_persistent_store(model_owner.name, local_bind_path)

    model_owner.create_workspace()

    # Modify the plan
    plan_path = constants.AGG_PLAN_PATH.format(local_bind_path)
    param_config = request.config

    initial_model_path = None
    if eval_scope:
        log.info(
            "Setting up evaluation scope, update the plan for 1 round and initial "
            "model to previous experiment best model"
        )
        initial_model_path = request.config.best_model_path

    model_owner.modify_plan(param_config, plan_path=plan_path)

    if hasattr(request.config, 'straggler_policy') and request.config.straggler_policy:
        model_owner.modify_straggler_policy(
            request.config.straggler_policy, plan_path=plan_path
        )

    return workspace_path, local_bind_path, agg_domain_name, model_owner, plan_path, agg_workspace_path, initial_model_path


def create_tr_workspace(request, eval_scope=False):
    """
    Create a task runner workspace.

    Args:
        request (object): Pytest request object.
        eval_scope (bool, optional): If True, sets up the evaluation scope for a
        single round. Defaults to False.

    Returns:
        tuple : A named tuple containing the objects for model owner, aggregator,
        and collaborators.
    """
    if request.config.model_name.lower() == constants.ModelName.TORCH_HISTOLOGY_S3.value:
        colab_bucket_mapping_list = prepare_data_for_s3(request)

    # get details of model owner, collaborators, and aggregator from common
    # workspace creation function
    workspace_path, local_bind_path, agg_domain_name, model_owner, plan_path, agg_workspace_path, initial_model_path = common_workspace_creation(request, eval_scope)

    # Initialize the plan
    model_owner.initialize_plan(
        agg_domain_name=agg_domain_name, extra_args=f"-i {initial_model_path}" if initial_model_path else ""
    )

    # Certify the workspace in case of TLS
    if request.config.use_tls:
        model_owner.certify_workspace()

    # Register the collaborators. It will also update plan/cols.yaml file with the collaborator names.
    model_owner.register_collaborators(plan_path, request.config.num_collaborators)

    # Create the objects for aggregator and collaborators
    # Workspace path for aggregator is uniform in case of docker or task_runner
    # But, for collaborators, it is different
    aggregator = agg_model.Aggregator(
        agg_domain_name=agg_domain_name,
        workspace_path=agg_workspace_path,
        eval_scope=eval_scope,
        container_id=model_owner.container_id,  # None in case of native environment
    )

    # Generate the sign request and certify the aggregator in case of TLS
    if request.config.use_tls:
        aggregator.generate_sign_request()
        model_owner.certify_aggregator(agg_domain_name)

    # Export the workspace
    # By default the workspace will be exported to workspace.zip
    model_owner.export_workspace()

    collaborators = []
    executor = concurrent.futures.ThreadPoolExecutor()

    # In case of torch/histology_s3, we need to pass the data path, flag to calculate hash
    # and bucket mapping to the setup_collaborator function
    if request.config.model_name.lower() == constants.ModelName.TORCH_HISTOLOGY_S3.value:
        futures = [
            executor.submit(
                fh.setup_collaborator,
                index,
                workspace_path=workspace_path,
                local_bind_path=local_bind_path,
                data_path="data",
                calc_hash=True,
                colab_bucket_mapping=next(
                    (item for item in colab_bucket_mapping_list if item["collaborator"] == f"collaborator{index}"),
                    None
                ),
            )
            for index in range(1, request.config.num_collaborators+1)
        ]
    else:
        futures = [
            executor.submit(
                fh.setup_collaborator,
                index,
                workspace_path=workspace_path,
                local_bind_path=local_bind_path,
            )
            for index in range(1, request.config.num_collaborators+1)
        ]
    collaborators = [f.result() for f in futures]

    # Data setup requires total no of collaborators, thus keeping the function call
    # outside of the loop
    if request.config.model_name.lower() in [constants.ModelName.XGB_HIGGS.value, constants.ModelName.FLOWER_APP_PYTORCH.value]:
        fh.setup_collaborator_data(collaborators, request.config.model_name, local_bind_path)

    if request.config.use_tls:
        fh.setup_pki_for_collaborators(collaborators, model_owner, local_bind_path)
        fh.import_pki_for_collaborators(collaborators)

    fh.remove_stale_processes(aggregator, collaborators)

    # Return the federation fixture
    return federation_details(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
        model_name=request.config.model_name,
    )


def create_tr_workspace_gandlf(request, eval_scope=False):
    """
    Create a task runner workspace for Gandlf model.

    Args:
        request (object): Pytest request
    """
    # get details of model owner, collaborators, and aggregator from common
    # workspace creation function
    workspace_path, local_bind_path, agg_domain_name, model_owner, plan_path, agg_workspace_path, initial_model_path = common_workspace_creation(request, eval_scope)

    home_dir = Path().home()
    results_path = os.path.join(home_dir, request.config.results_dir)

    # Raise exception if openfl does not contain gandlf folder.
    if not os.path.isdir(os.path.join(os.getcwd(), "gandlf")):
        raise Exception(
            "Folder 'gandlf' is not present in the current working directory. "
            "Please ensure that all the pre-requisites are met before running the test. "
            "Refer file .github/workflows/gandlf.yaml for the same."
        )

    # Check if valid.csv and train.csv are present in openfl folder
    if not os.path.exists(os.path.join(results_path, "valid.csv")) or not os.path.exists(
        os.path.join(results_path, "train.csv")
    ):
        raise ex.DataSetupException("Required data files for GanDLF are missing in the openfl folder")

    # Check if file config_segmentation.yaml is present in openfl folder
    gandlf_seg_file = os.path.join(results_path, "config_segmentation.yaml")
    if not os.path.exists(gandlf_seg_file):
        raise ex.GaNDLFConfigSegException(f"File {gandlf_seg_file} not available.")

    with open(gandlf_seg_file, 'r') as file:
        content = file.read()

    if not "num_channels" in content:
        raise ex.GaNDLFConfigSegException(f"File {gandlf_seg_file} must contain entry for num_channels.")

    # Create the objects for aggregator
    aggregator = agg_model.Aggregator(
        agg_domain_name=agg_domain_name,
        workspace_path=agg_workspace_path,
        eval_scope=eval_scope,
        container_id=model_owner.container_id,  # None in case of native environment
    )

    # Currently plan initialization internally checks data path in data.yaml
    # So we need to have data and modified data.yaml file in place before initializing the plan
    # Issue - https://github.com/securefederatedai/openfl/issues/73
    fh.download_gandlf_data(aggregator, local_bind_path, request.config.num_collaborators, results_path)

    # Initialize the plan
    extra_args = f"--gandlf_config {gandlf_seg_file}"
    extra_args += f" -i {initial_model_path}" if initial_model_path else ""

    model_owner.initialize_plan(
        agg_domain_name=agg_domain_name, extra_args=extra_args
    )

    # Update cols.yaml file with the collaborator names
    model_owner.register_collaborators(plan_path, request.config.num_collaborators)

    # Certify the workspace, generate the sign request and certify the aggregator
    if request.config.use_tls:
        model_owner.certify_workspace()
        aggregator.generate_sign_request()
        model_owner.certify_aggregator(agg_domain_name)

    # Export the workspace
    # By default the workspace will be exported to workspace.zip
    model_owner.export_workspace()

    collaborators = []
    executor = concurrent.futures.ThreadPoolExecutor()

    futures = [
        executor.submit(
            fh.setup_collaborator,
            index,
            workspace_path=workspace_path,
            local_bind_path=local_bind_path
        )
        for index in range(1, request.config.num_collaborators+1)
    ]
    collaborators = [f.result() for f in futures]

    fh.copy_gandlf_data_to_collaborators(aggregator, collaborators, local_bind_path)

    if request.config.use_tls:
        fh.setup_pki_for_collaborators(collaborators, model_owner, local_bind_path)
        fh.import_pki_for_collaborators(collaborators)

    # Return the federation fixture
    return federation_details(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
        model_name=request.config.model_name,
    )


def create_tr_dws_workspace(request, eval_scope=False):
    """
    Run task runner experiment thru dockerized workspace.

    Args:
        request (object): Pytest request object.
        eval_scope (bool, optional): If True, sets up the evaluation scope for a
        single round. Defaults to False.

    Returns:
        tuple: A named tuple containing the objects for model owner, aggregator,
        and collaborators.
    """
    # get details of model owner, collaborators, and aggregator from common
    # workspace creation function
    workspace_path, local_bind_path, agg_domain_name, model_owner, plan_path, agg_workspace_path, initial_model_path = common_workspace_creation(request, eval_scope)

    # Initialize the plan
    model_owner.initialize_plan(
        agg_domain_name=agg_domain_name, extra_args=f"-i {initial_model_path}" if initial_model_path else ""
    )

    # Create openfl image
    dh.build_docker_image(constants.DEFAULT_OPENFL_IMAGE, constants.DEFAULT_OPENFL_DOCKERFILE)

    # Command 'fx workspace dockerize --save ..' will use the workspace name for
    # image name which is 'workspace' in this case.
    model_owner.dockerize_workspace(constants.DEFAULT_OPENFL_IMAGE)

    # Certify the workspace in case of TLS
    if request.config.use_tls:
        model_owner.certify_workspace()

    # Register the collaborators. It will also update plan/cols.yaml file with the collaborator names.
    model_owner.register_collaborators(plan_path, request.config.num_collaborators)

    # Create the objects for aggregator and collaborators
    # Workspace path for aggregator is uniform in case of docker or task_runner
    # But, for collaborators, it is different
    aggregator = agg_model.Aggregator(
        agg_domain_name=agg_domain_name,
        workspace_path=agg_workspace_path,
        eval_scope=eval_scope,
        container_id=model_owner.container_id,  # None in case of native environment
    )

    collaborators = []
    executor = concurrent.futures.ThreadPoolExecutor()
    futures = [
        executor.submit(
            fh.setup_collaborator,
            index,
            workspace_path=workspace_path,
            local_bind_path=local_bind_path,
        )
        for index in range(1, request.config.num_collaborators + 1)
    ]
    collaborators = [f.result() for f in futures]

    if request.config.use_tls:
        fh.setup_pki_for_collaborators(collaborators, model_owner, local_bind_path)

    # Data setup requires total no of collaborators, thus keeping the function call
    # outside of the loop
    if request.config.model_name.lower() in [constants.ModelName.XGB_HIGGS.value, constants.ModelName.FLOWER_APP_PYTORCH.value]:
        fh.setup_collaborator_data(collaborators, request.config.model_name, local_bind_path)

    # Note: In case of multiple machines setup, scp the created tar for collaborators
    # to the other machine(s)
    fh.create_tarball_for_collaborators(
        collaborators, local_bind_path, use_tls=request.config.use_tls,
        add_data=True if request.config.model_name.lower() in [constants.ModelName.XGB_HIGGS.value, constants.ModelName.FLOWER_APP_PYTORCH.value] else False
    )

    # Generate the sign request and certify the aggregator in case of TLS
    if request.config.use_tls:
        aggregator.generate_sign_request()
        model_owner.certify_aggregator(agg_domain_name)

    local_agg_ws_path = constants.AGG_WORKSPACE_PATH.format(local_bind_path)

    # Note: In case of multiple machines setup, scp this tar to the other machine(s)
    return_code, output, error = ssh.run_command(
        f"tar -cf cert_aggregator.tar plan cert save", work_dir=local_agg_ws_path
    )
    if return_code != 0:
        raise Exception(f"Failed to create tar for aggregator: {error}")

    # Note: In case of multiple machines setup, scp this workspace tar
    # to the other machine(s) so that docker load can load the image.
    model_owner.load_workspace(workspace_tar_name=f"{constants.DFLT_WORKSPACE_NAME}.tar")

    # Return the federation fixture
    return federation_details(
        model_owner=model_owner,
        aggregator=aggregator,
        collaborators=collaborators,
        workspace_path=workspace_path,
        local_bind_path=local_bind_path,
        model_name=request.config.model_name,
    )


def prepare_data_for_s3(request):
    """
    Prepare data for S3. Includes starting minio server, creating bucket, and uploading data.
    Args:
        request (object): Pytest request object.
    Returns:
        dict: A dictionary containing the bucket mapping for each collaborator.
        Example -
        [
            {'collaborator': 'collaborator1', 'local_data_path': '/home/azureuser/openfl/data/1', 'buckets': ['bucket-1']},
            {'collaborator': 'collaborator2', 'local_data_path': '/home/azureuser/openfl/data/2', 'buckets': ['bucket-2-01', 'bucket-2-02']}
        ]
    """
    s3_obj = s3_model.S3Bucket()

    num_collaborators = request.config.num_collaborators

    # Import the dataloader module for torch/histology to download the data
    # As the folder name contains hyphen, we need to use importlib to import the module
    dataloader_module = importlib.import_module("openfl-workspace.torch.histology.src.dataloader")

    # Download the data for torch/histology in current folder as internally it uses the current folder as data path
    try:
        log.info(f"Downloading data for {constants.ModelName.TORCH_HISTOLOGY_S3.value}")
        dataloader_module.HistologyDataset()
        log.info("Download completed")
    except Exception as e:
        raise ex.DataDownloadException(
            f"Failed to download data for {constants.ModelName.TORCH_HISTOLOGY_S3.value}. Error: {e}"
        )

    # Distibute the downloaded data/folders among the collaborators
    hist_data_path = Path.cwd().absolute() / 'data' # We cannot change it, as the data loader is using this path without any input
    try:
        distribute_data_to_collaborators(num_collaborators, hist_data_path)
    except Exception as e:
        raise ex.DataSetupException(
            f"Failed to distribute data to collaborators. Error: {e}"
        )

    # Start minio server, create S3 buckets and upload the data to S3
    try:
        s3_obj.start_minio_server(
            data_dir=os.path.join(Path().home(), request.config.results_dir, constants.MINIO_DATA_FOLDER)
        )
        log.info("Started minio server")
    except Exception as e:
        raise ex.MinioServerStartException(
            f"Failed to start minio server. Error: {e}"
        )

    # Create the buckets for each collaborator
    # The bucket name will be bucket-1, bucket-2, ..., bucket-n
    # where n is the number of collaborators
    colab_bucket_mapping_list = []
    bucket_name = None
    for index in range(1, num_collaborators + 1):
        try:
            folder_path = hist_data_path / str(index)
            if index % 2 == 0:
                bucket_list = []
                for suffix in ["01", "02"]:
                    bucket_name = f"bucket-{index}-{suffix}"
                    s3_obj.create_bucket(bucket_name=bucket_name)
                    log.info(f"Created bucket {bucket_name}")
                    bucket_list.append(bucket_name)
                colab_bucket_mapping_list.append({
                    "collaborator": f"collaborator{index}",
                    "local_data_path": str(folder_path),
                    "buckets": bucket_list
                })
            else:
                bucket_name = f"bucket-{index}"
                s3_obj.create_bucket(bucket_name=bucket_name)
                log.info(f"Created bucket {bucket_name}")
                colab_bucket_mapping_list.append({
                    "collaborator": f"collaborator{index}",
                    "local_data_path": str(folder_path),
                    "buckets": [bucket_name]
                })
        except Exception as e:
            raise ex.S3BucketCreationException(
                f"Failed to create bucket {bucket_name} for collaborator{index}. Error: {e}"
            )

    log.info(f"Bucket mapping: {colab_bucket_mapping_list}")

    # List the buckets to verify
    s3_obj.list_buckets()

    # Copy the data to the S3 buckets by equally distributing the data among the collaborators
    s3_helper.upload_data_to_s3(s3_obj, colab_bucket_mapping_list)

    return colab_bucket_mapping_list


def distribute_data_to_collaborators(num_collaborators, data_path):
    """
    Distribute the data among the collaborators uniformly.
    Example: Assuming num_collaborators is 3
        If data_path has folder Kather_texture_2016_image_tiles_5000 (torch/histology) which further has 8 subfolders,
        then the data will be distributed as:
            collaborator1: 1 / first 3 subfolders
            collaborator2: 2 / next 3 subfolders
            collaborator3: 3 / last 2 subfolders
        If data_path itself has multiple folders say 8, then the data will be distributed as:
            collaborator1: 1 / first 3 folders
            collaborator2: 2 / next 3 folders
            collaborator3: 3 / last 2 folders
    Args:
        num_collaborators (int): Number of collaborators.
        data_path (str): Path to the data directory.
    Raises:
        Exception: If the data distribution fails.
    """
    # If data_path has only one folder, go inside it and use its subfolders
    all_entries = [f for f in data_path.iterdir() if f.is_dir()]
    if len(all_entries) == 1:
        # Use subfolders inside the single folder
        all_folders = [f for f in all_entries[0].iterdir() if f.is_dir()]
    else:
        all_folders = all_entries
    all_folders.sort()  # For deterministic split

    num_folders = len(all_folders)
    folders_per_collab = [num_folders // num_collaborators] * num_collaborators

    # Distribute the remainder (if any) to the first few collaborators
    for i in range(num_folders % num_collaborators):
        folders_per_collab[i] += 1

    start = 0
    for index in range(1, num_collaborators + 1):
        collaborator_data_path = data_path / str(index)
        collaborator_data_path.mkdir(parents=True, exist_ok=True)
        end = start + folders_per_collab[index - 1]
        for folder in all_folders[start:end]:
            # Move or copy the folder to the collaborator's directory
            # Here we move; use shutil.copytree if you want to copy instead
            folder.rename(collaborator_data_path / folder.name)
        start = end
