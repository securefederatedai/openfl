"""
This script automates the setup and execution of a federated learning workflow using 
OpenFL with TLS (Transport Layer Security). It includes functionalities for managing 
certificates, configuring participants, and starting the required services (e.g., 
certifier, director, and envoys) across multiple machines.

Key Features:
1. Fetches the public IP of the current machine.
2. Loads machine mapping data from a JSON file to identify participants and their details.
3. Automates the cleanup of processes and caches on all machines.
4. Installs and starts the certifier service.
5. Generates and retrieves tokens for participants (e.g., director, manager, collaborators).
6. Certifies participants and configures their YAML files for the federated learning workflow.
7. Activates or deactivates experimental features using the `fx` command.
8. Starts the director and envoy services, either locally or on remote machines via SSH.
9. Supports running commands in the background and logging outputs.

Assumptions:
1. `/etc/hosts` is updated for all nodes with director and certifier IPs.
2. The script runs on the machine hosting the certifier and director.
3. Certifier and director services are running on the same machine.
4. The `workflow_ip_mapping.json` file contains correct IPs and PEM file paths.
5. PEM files are present in the home directory of the user.
6. The script assumes the use of the `fx` command-line tool for managing OpenFL components.
7. Since there is bug #1428 manually update notebook and execute.

Dependencies:
- Python libraries: `os`, `subprocess`, `requests`, `tempfile`, `time`, `yaml`, `json`, `paramiko`, `pathlib`.
- External tools: `fx` command-line tool for OpenFL.

Usage:
Run the script as a standalone program to set up and start the federated learning workflow.
"""

import os
import subprocess
import requests
import tempfile
import time
import yaml
from pathlib import Path
import json
import paramiko

collab_venv_path = "/home/azureuser/venv_3.10_1.8"
openfl_dir = "/home/azureuser/openfl_main"
certifier_url = "certifier"
certifier_port = 8080
HOME_DIR = Path().home()
DIRECTOR_HOST = "director"
DIRECTOR_PORT = 50051

workflow_dir = os.path.join(openfl_dir, "openfl-tutorials", "experimental", "workflow", "FederatedRuntime", "301_MNIST_Watermarking")
path_to_certifier = os.path.join(workflow_dir, "certifier")

collaborator_name = ["Bangalore", "Chandler"]

def get_public_ip():
    """
    Get the public IP address of the current machine.

    :return: Public IP address as a string.
    """
    response = requests.get('https://api.ipify.org?format=json')
    return response.json()['ip']


def load_mapping_data(mapping_file_path):
    """
    Load the machine mapping data from the specified JSON file.

    :param mapping_file_path: Path to the workflow_ip_mapping.json file.
    :return: Parsed JSON data as a Python object.
    """
    try:
        with open(mapping_file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Failed to load mapping data: {e}")
        raise e


def get_participant_details(participant_name, mapping_json):
    """
    Get the IP address and PEM file of a specific participant from the mapping data.

    :param participant_name: Name of the participant to search for.
    :param mapping_json: Parsed JSON data containing machine mapping.
    :return: Dictionary with participant details or None if not found.
    """
    for entry in mapping_json:
        if participant_name in entry['participants']:
            return {'ip': entry['ip'], 'pem_file': entry['pem_file'], 'participant_name': participant_name}
    print(f"Participant {participant_name} not found in mapping data.")


def cleanup_workspace():
    """
    Clean up the workspace by killing specific processes and dropping caches on all machines.
    """
    command = "sudo kill -9 $(ps -ef | grep -e 'fx' -e 'certifier' -e 'director' -e 'envoy' | awk '{print $2}')"
    subprocess.run(["echo", "3", "|", "sudo", "tee", "/proc/sys/vm/drop_caches"], check=False)
    try:
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print(f"An error occurred during cleanup: {e}")

    for entry in MACHINE_MAPPING_DATA:
        ip = entry['ip']
        pem_file = os.path.join(HOME_DIR, entry['pem_file'])
        if ip != get_public_ip():
            remote_command_run(ip, pem_file, cmd=command, check=False, workflow_dir=HOME_DIR)
            remote_command_run(ip, pem_file, cmd="echo 3 | sudo tee /proc/sys/vm/drop_caches", check=False, workflow_dir=HOME_DIR)
            print(f"Cache dropped successfully on {ip}")


def install_certifier(certifier_url, certifier_port, path_to_certifier):
    """
    Install the certifier using the fx command.

    :param certifier_url: URL of the certifier.
    :param certifier_port: Port of the certifier.
    :param path_to_certifier: Path to the certifier directory.
    """
    try:
        command = [
            "fx", "pki", "install", "-p", path_to_certifier,
            "--ca-url", f"{certifier_url}:{certifier_port}",
            "--password", "1234"
        ]
        process = subprocess.run(command, input="Y\n", text=True, capture_output=True)
        if process.returncode == 0:
            print("Certifier started successfully.")
        else:
            print(f"Error starting certifier: {process.stderr}")
    except Exception as e:
        print(f"An exception occurred: {e}")
        raise e


def start_certifer(path_to_certifier):
    """
    Start the certifier process in the background using the fx command.

    :param path_to_certifier: Path to the certifier directory.
    """
    directory = os.path.dirname(path_to_certifier)
    certifier_file_name = os.path.basename(path_to_certifier)
    try:
        command = ["fx", "pki", "run", "-p", certifier_file_name]
        bg_file = open(os.path.join(tempfile.mkdtemp(), "tmp.log"), "a", buffering=1)
        print(f"Certifier log file: {bg_file.name}")
        process = subprocess.Popen(command, stdout=bg_file, stderr=subprocess.STDOUT, shell=False, text=True, cwd=directory)
        if process.poll() is None:
            print("Certifier is running in background.")
        else:
            print("Certifier failed to start.")
        time.sleep(5)
    except Exception as e:
        print(f"An exception occurred: {e}")
        raise e


def get_participant_token(certifier_url, certifier_port, path_to_certifier, participant_name):
    """
    Retrieve a participant token using the fx command.

    :param certifier_url: URL of the certifier.
    :param certifier_port: Port of the certifier.
    :param path_to_certifier: Path to the certifier directory.
    :param participant_name: Name of the participant.
    :return: Token as a string.
    """
    directory = os.path.dirname(path_to_certifier)
    certifier_file_name = os.path.basename(path_to_certifier)
    try:
        command = [
            "fx", "pki", "get-token", "-n", participant_name,
            "--ca-path", certifier_file_name,
            "--ca-url", f"{certifier_url}:{certifier_port}"
        ]
        process = subprocess.run(command, text=True, capture_output=True, cwd=directory)
        if process.returncode == 0:
            print(f"{participant_name} token retrieved successfully.")
            token = process.stdout.split("Token: ")[1].split("\n")[0].strip()
            return token
        else:
            raise Exception(f"Error retrieving participant token: {process.stderr}")
    except Exception as e:
        print(f"An exception occurred: {e}")
        raise e
 

def certify_participant(participant_details, workspace_dir):
    """
    Certifies a participant using the `fx pki certify` command. The function determines 
    whether to execute the command locally or on a remote machine based on the participant's 
    IP address.

    Args:
        participant_details (dict): A dictionary containing details of the participant, 
            including their name, IP address, token, and PEM file path.
        workspace_dir (str): The base directory where the participant's workspace is located.

    Raises:
        Exception: If an error occurs during the execution of the certification command, 
            the exception is caught, logged, and re-raised.

    Notes:
        - If the participant's IP matches the public IP of the current machine, the command 
          is executed locally.
        - If the participant's IP does not match, the command is executed on the remote 
          machine using SSH.
        - The `fx pki certify` command is run with the `-n` flag for the participant name 
          and the `--token` flag for the token.
        - For the "manager" participant, the workspace directory is adjusted to include 
          a "workspace" subdirectory.
    """
    participant_name = participant_details['participant_name']
    token = participant_details['token']
    command = ["fx", "pki", "certify", "-n", participant_name, "--token", token]
    
    if participant_name == "manager":
        workspace_dir = os.path.join(workspace_dir, "workspace")
    else:
        workspace_dir = os.path.join(workspace_dir, participant_name)
    try:
        if participant_details['ip'] == get_public_ip():
            # Run the command locally
            subprocess.run(
                command,
                input="Y\n",
                text=True,
                capture_output=True,
                cwd=workspace_dir,
                shell=False,
            )
        else:
            # Run the command on the remote machine
            pem_file = os.path.join(HOME_DIR, participant_details['pem_file'])
            # Convert command to string
            command = " ".join(command)
            command = f"echo Y | {command}"
            print(f"Running command on remote machine: {command}")
            remote_command_run(participant_details['ip'], pem_file, command, check=True, workflow_dir=workspace_dir)
    except Exception as e:
        print(f"An exception occurred: {e}")
        raise e


def create_participant_config_file(participant_detail, workflow_dir):
    """
    Creates and updates a participant configuration YAML file, and optionally transfers it to a remote machine.

    Args:
        participant_detail (dict): A dictionary containing details about the participant. 
            Expected keys:
                - 'participant_name' (str): The name of the participant.
                - 'ip' (str): The IP address of the participant's machine.
                - 'pem_file' (str): The path to the PEM file for SSH authentication.
        workflow_dir (str): The directory path where the workflow files are stored.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If required keys are missing in the participant_detail dictionary or the YAML file.
        subprocess.CalledProcessError: If the SCP command fails to execute.

    Notes:
        - The function updates the 'director_host' and 'director_port' fields in the YAML configuration file.
        - If the participant's IP address does not match the public IP of the current machine, the updated file is transferred to the remote machine using SCP.
        - The PEM file is used for secure authentication during the file transfer.
    """
    participant_name = participant_detail['participant_name']
    workflow_dir = os.path.join(workflow_dir, participant_name)
    config_path = os.path.join(workflow_dir,f"{participant_name}_config.yaml") 
    # load yaml file
    with open(config_path, "r") as file:
        data = yaml.safe_load(file)
    # update the director host and port
    data['settings']['director_host'] = DIRECTOR_HOST
    data['settings']['director_port'] = DIRECTOR_PORT
    # write the updated yaml file
    with open(config_path, "w") as file:
        yaml.dump(data, file)
    if participant_detail['ip'] != get_public_ip():
        # replace existing file in remote location 
        pem_file = os.path.join(HOME_DIR, participant_detail['pem_file'])
        # convert commad to string
        command = f"scp -i {pem_file} {config_path} azureuser@{participant_detail['ip']}:{workflow_dir}"
        print(f"Running command on remote machine: {command}")
        subprocess.run(command, shell=True, check=True)
        
    
def create_director_config_file(participant_name, workflow_dir, machine_mapping_data):
    """
    Creates and updates the director configuration file for a specific participant.
    This function loads an existing `director_config.yaml` file, updates its 
    `listen_host` and `listen_port` settings with predefined constants, and 
    writes the updated configuration back to the file.
    Args:
        participant_name (str): The name of the participant for whom the 
            configuration file is being created.
        workflow_dir (str): The base directory where the workflow files are stored.
        machine_mapping_data (dict): A dictionary containing machine mapping 
            information (currently unused in this function).
    Raises:
        FileNotFoundError: If the `director_config.yaml` file does not exist 
            in the specified path.
        yaml.YAMLError: If there is an error in reading or writing the YAML file.
    """

    # participant_details = get_participant_details(participant_name, machine_mapping_data)
    # load director_config.yaml and update director host and port
    workflow_dir = os.path.join(workflow_dir, participant_name)
    director_config_path = os.path.join(workflow_dir,"director_config.yaml")    
    
    with open(director_config_path, "r") as file:
        data = yaml.safe_load(file)
    # update the director host and port
    data['settings']['listen_host'] = DIRECTOR_HOST
    data['settings']['listen_port'] = DIRECTOR_PORT
    # write the updated yaml file
    with open(director_config_path, "w") as file:
        yaml.dump(data, file)
        

def activate_fx_experimental(machine_mapping_data, activate):
    """
    Activates or deactivates the FX experimental feature on local or remote machines.

    This function determines whether to activate or deactivate the FX experimental 
    feature based on the `activate` parameter. It runs the appropriate command 
    either locally or on remote machines specified in the `machine_mapping_data`.

    Args:
        machine_mapping_data (list): A list of dictionaries containing machine 
            information. Each dictionary should have the following keys:
            - 'ip' (str): The IP address of the machine.
            - 'pem_file' (str): The path to the PEM file for SSH authentication.
        activate (bool): A flag indicating whether to activate (True) or deactivate (False) 
            the FX experimental feature.

    Raises:
        Exception: If an error occurs while running the command on any machine.

    Notes:
        - The function uses the `get_public_ip` function to determine the public IP 
          of the current machine.
        - If the IP address in `machine_mapping_data` matches the public IP, the 
          command is executed locally. Otherwise, it is executed on the remote machine 
          using SSH.
        - The `remote_command_run` function is used to execute commands on remote machines.
    """
    # get the public ip of the machine
    public_ip = get_public_ip()
    if activate:
        command = ["fx", "experimental", "activate"]
    else:
        command = ["fx", "experimental", "deactivate"]
    for entry in machine_mapping_data:
        print(f"Running command on remote machine: {command} and machine: {entry['ip']}")   
        try:
            if entry['ip'] == public_ip:
            # Run the command locally
                subprocess.run(command, check=True)
            else:
            # Run the command on the remote machine
                pem_file = os.path.join(HOME_DIR, entry['pem_file'])
                # convert command to string
                join_command = " ".join(command)
                remote_command_run(entry['ip'], pem_file, join_command, check=True, workflow_dir=HOME_DIR)
        except Exception as e:
            print(f"An error occurred while {command} {entry['ip']}: {e}")


def start_director(workflow_dir, machine_mapping_data):
    """
    Starts the director process in the background using the `fx` command with TLS enabled.

    Args:
        workflow_dir (str): The base directory where the director's files and logs are located.
        machine_mapping_data (dict): Data related to machine mappings (not used in the current implementation).

    Workflow:
        - Constructs the path to the director's working directory.
        - Prepares the `fx` command to start the director with TLS configuration.
        - Logs the director's output to a file named `director.log` in the specified workflow directory.
        - Executes the command in the background.

    Command:
        fx director start --tls -rc cert/root_ca.crt -pk cert/director.key -oc cert/director.crt -c director_config.yaml

    Raises:
        Any exceptions raised by the `run_in_background` function or file operations.

    Note:
        Ensure that the required certificate files (`root_ca.crt`, `director.key`, `director.crt`) 
        and the configuration file (`director_config.yaml`) are present in the `cert` directory 
        relative to the workflow directory.
     """
    workflow_dir = os.path.join(workflow_dir, "director")  
    command = [
        "fx", "director", "start", "--tls",
        "-rc", "cert/root_ca.crt",
        "-pk", "cert/director.key",
        "-oc", "cert/director.crt",
        "-c", "director_config.yaml"
    ]
    bg_file = open(os.path.join(workflow_dir, "director.log"), "w", buffering=1)
    print(f"Director log file: {bg_file.name}")
    run_in_background(command, workflow_dir, bg_file)
    print(f"Director started successfully. {' '.join(command)}")


def start_envoys(envoy_detail, workflow_dir):
    """
    Starts an envoy process either locally or on a remote machine based on the provided details.

    Args:
        envoy_detail (dict): A dictionary containing details about the envoy participant. 
            Expected keys:
                - 'participant_name' (str): The name of the envoy participant.
                - 'ip' (str): The IP address of the envoy participant.
                - 'pem_file' (str): The path to the PEM file for SSH authentication (required for remote execution).
        workflow_dir (str): The base directory for the workflow. The envoy's specific directory will be appended.

    Behavior:
        - If the envoy's IP matches the public IP of the current machine, the envoy process is started locally in the background.
        - If the envoy's IP does not match the public IP, the envoy process is started on the remote machine via SSH.

    Notes:
        - The function uses TLS certificates and keys for secure communication.
        - Logs for the envoy process are written to a file named `<participant_name>.log`.
        - For remote execution, the function activates a virtual environment and runs the command in the specified workflow directory.
        - The remote process is started in the background using a non-blocking SSH command.

    Raises:
        None explicitly, but errors may occur if:
            - Required files (e.g., certificates, PEM file) are missing.
            - SSH connection fails.
            - The command fails to execute on the remote machine.
    """
    workflow_dir = os.path.join(workflow_dir, envoy_detail['participant_name'])
    envoy_name = envoy_detail['participant_name']
    envoy_ip = envoy_detail['ip']
    command = [
        "fx", "envoy", "start", "--tls",
        "-n", envoy_name,
        "-rc", "cert/root_ca.crt",
        "-pk", f"cert/{envoy_name}.key",
        "-oc", f"cert/{envoy_name}.crt",
        "-c", f"{envoy_name}_config.yaml"
    ]
    bg_file = f"{envoy_name}.log"
    print(f"Envoy log file: {bg_file}")
    if envoy_ip == get_public_ip():
        run_in_background(command, workflow_dir, bg_file)
    else:
        # Run the command on the remote machine
        pem_file = os.path.join(HOME_DIR, envoy_detail['pem_file'])
        # convert command to string
        join_command = " ".join(command)
        print(f"Running command on remote machine: {join_command}")
        process = subprocess.Popen([
                "ssh",
                "-i", pem_file,
                "-v",
                "-o", "StrictHostKeyChecking=no",
                f"azureuser@{envoy_detail['ip']}",
                f"source {collab_venv_path}/bin/activate && cd {workflow_dir} && {join_command} > {bg_file} 2>&1 &"],
            stdout=subprocess.DEVNULL,  # Suppress local stdout
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        )
        time.sleep(10)  # Wait for the process to start
        process.kill()  # Terminate the process


def remote_command_run(ip, pem_file, cmd, check=True, workflow_dir=None, bg_file=False):
    """
    Executes a command on a remote machine via SSH.
    Args:
        ip (str): The IP address of the remote machine.
        pem_file (str): The path to the private key file (PEM) for authentication.
        cmd (str): The command to execute on the remote machine.
        check (bool, optional): If True, raises an exception if the command fails. Defaults to True.
        workflow_dir (str, optional): The directory on the remote machine where the command should be executed. Defaults to None.
        bg_file (bool or str, optional): If False, the command runs in the foreground. If a string is provided, the command runs in the background, and its output is redirected to the specified file. Defaults to False.
    Raises:
        Exception: If the command fails and `check` is True, an exception is raised with the error details.
    Notes:
        - The function establishes an SSH connection using the provided private key.
        - If `bg_file` is specified, the command runs in the background, and its output is redirected to the file.
        - The function automatically accepts the host key for the SSH connection.
    Example:
        remote_command_run(
            ip="192.168.1.1",
            pem_file="/path/to/key.pem",
            cmd="ls -la",
            workflow_dir="/home/azureuser",
            bg_file="output.log"
        )
    """
    try:
        # Create an SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy())  # Automatically accept host key

        # Connect to the remote machine with StrictHostKeyChecking disabled
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Automatically accept host key
        private_key = paramiko.RSAKey.from_private_key_file(pem_file)
        ssh.connect(hostname=ip, username="azureuser", pkey=private_key, look_for_keys=False, allow_agent=False)
        activate_venv_cmd = f"source {collab_venv_path}/bin/activate"
        
        # Modify the command to run in the background if bg_flag is True
        if bg_file:
            cmd = f"cd {workflow_dir} && {activate_venv_cmd} && {cmd} > {bg_file} 2>&1 &"
        else:
            cmd = f"cd {workflow_dir} && {activate_venv_cmd} && {cmd}"
        print(f"Running command on remote machine: {cmd}")
        # Execute the command
        stdin, stdout, stderr = ssh.exec_command(cmd)

        # If not running in the background, wait for the command to complete and check the exit status
        if not bg_file:
            exit_status = stdout.channel.recv_exit_status()
            if check and exit_status != 0:
                raise Exception(f"Command failed with exit status {exit_status}: {stderr.read().decode()}")

            # Print the command output
            print(stdout.read().decode())
    except Exception as e:
        print(f"Failed to run command {cmd} on {ip}: {e}")
        raise e
    finally:
        # Close the SSH connection
        ssh.close()


def run_in_background(command, workflow_dir, bg_file):
    """
    Executes a command in the background within a specified directory, redirecting 
    output and error streams to a provided file.
    Args:
        command (list): The command to execute as a list of strings. Each element 
                        represents a part of the command (e.g., ['ls', '-l']).
        workflow_dir (str): The directory in which the command will be executed.
        bg_file (file object): A file object where the command's output and errors 
                               will be redirected.
    Raises:
        Exception: If an error occurs during the execution of the command, the 
                   exception is caught, logged, and re-raised.
    Notes:
        - The function uses `subprocess.Popen` to execute the command in the 
          background.
        - It checks if the process is running by verifying the return value of 
          `process.poll()`.
        - Ensure that `bg_file` is opened in write or append mode before passing 
          it to this function.
    """
    try:
        output_redirect = bg_file
        error_redirect = subprocess.STDOUT
        
        # Run the command in background
        process = subprocess.Popen(
        command, stdout=output_redirect, stderr=error_redirect, shell=False, text=True, cwd=workflow_dir
        ) 
        # verify if the process is running
        if process.poll() is None:
            print("Process is running in background.")
        else:
            print("Process failed to start.")
    except Exception as e:
        print(f"An exception occurred: {e}")
        raise e


if __name__ == "__main__":
    # Get the public IP address of the current machine
    public_ip = get_public_ip()
    print(f"Public IP: {public_ip}")

    MACHINE_MAPPING_DATA = load_mapping_data(os.path.join(os.getcwd(), "workflow_ip_mapping.json"))

    cleanup_workspace()
    activate_fx_experimental(MACHINE_MAPPING_DATA, activate=False)
    # Certifier get started in the host machine.
    install_certifier(certifier_url, certifier_port, path_to_certifier)
    start_certifer(path_to_certifier)

    # create certificates for director
    director_detail = get_participant_details("director", MACHINE_MAPPING_DATA)
    director_token = get_participant_token(certifier_url, certifier_port, path_to_certifier, "director")
    director_detail['token'] = director_token

    manager_detail = get_participant_details("manager", MACHINE_MAPPING_DATA)
    manager_token = get_participant_token(certifier_url, certifier_port, path_to_certifier, "manager")
    manager_detail['token'] = manager_token
 
    collaborator_details = []
    for collaborator in collaborator_name:
        collaborator_detail = {}
        collaborator_detail = get_participant_details(collaborator, MACHINE_MAPPING_DATA)
        collaborator_detail_token = get_participant_token(certifier_url, certifier_port, path_to_certifier, collaborator)
        collaborator_detail['token'] = collaborator_detail_token
        certify_participant(collaborator_detail, workflow_dir)
        create_participant_config_file(collaborator_detail, workflow_dir)
        # create_participant_config_file(collaborator, workflow_dir, MACHINE_MAPPING_DATA)
        collaborator_details.append(collaborator_detail)

    # # create certificates for participants
    
    certify_participant(director_detail, workflow_dir)
 
    certify_participant(manager_detail, workflow_dir)

    # # update director url in Bangalore and Chandler
    # # create_participant_config_file("Bangalore", workflow_dir, MACHINE_MAPPING_DATA)
    # # create_participant_config_file("Chandler", workflow_dir, MACHINE_MAPPING_DATA)
    create_director_config_file("director", workflow_dir, MACHINE_MAPPING_DATA)

    # activate fx experimental in all machines
    activate_fx_experimental(MACHINE_MAPPING_DATA, activate=True)
    # Start the director
    start_director(workflow_dir, MACHINE_MAPPING_DATA)
    # # Start the envoys

    for collaborator_detail in collaborator_details:
        # start the envoys
        start_envoys(collaborator_detail, workflow_dir)
