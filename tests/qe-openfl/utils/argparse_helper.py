import argparse
import os
import sys
from utils.logger import logger as log


def parse_arguments():
    """
    Parse command line arguments to provide the required parameters for running the tests.

    Returns:
    argparse.Namespace: Parsed command line arguments with the following attributes:
        - deploy_repo_path (str): Path to the deploy repository
        - results_dir (str, optional): Directory to store the results
        - repo_dir (str, optional): Path to the repository directory
        - num_collaborators (int, default=2): Number of collaborators
        - test_mode (str, default="ui"): Mode in which the test should run (ui or cli)

    Raises:
    SystemExit: If the required --deploy-repo-path argument is not provided or if any argument parsing error occurs.
    """
    try:
        parser = argparse.ArgumentParser(description='Provide the required arguments to run the tests')
        parser.add_argument('--deploy-repo-path', type=str, required=True, help='Path to the deploy repository')
        parser.add_argument('--results-dir', type=str, required=False, help='Directory to store the results')
        parser.add_argument('--repo-dir', type=str, required=False, help='Path to the repository directory')
        parser.add_argument('--num-collaborators', type=int, default=2, help='Number of collaborators')
        parser.add_argument("--test-mode", type=str, default="ui", help="Mode in which the test should run (ui or cli")
        args = parser.parse_known_args()[0]
        log.info("Arguments parsed successfully.")
        return args
    except Exception as e:
        log.error(f"Failed to parse arguments: {e}")
        sys.exit(1)


def get_default_repo_dir():
    """
    Find the default repository directory by traversing up the file hierarchy until a .git directory is found.

    Returns:
    str: Path to the repository directory.

    Raises:
    FileNotFoundError: If the .git directory is not found in the file hierarchy.
    """
    try:
        current_dir = os.path.abspath(__file__)
        while not os.path.exists(os.path.join(current_dir, '.git')) and os.path.dirname(current_dir) != current_dir:
            current_dir = os.path.dirname(current_dir)
        if not os.path.exists(os.path.join(current_dir, '.git')):
            raise FileNotFoundError("'.git' directory not found in the file hierarchy")
        project_dir = current_dir
        log.info(f"Default repository directory found: {project_dir}")
        return project_dir
    except Exception as e:
        log.error(f"Failed to get default repository directory: {e}")
        raise


# Example usage (you can remove this from the final code if needed)
if __name__ == "__main__":
    try:
        args = parse_arguments()
        repo_dir = get_default_repo_dir()
    except Exception as e:
        log.error(f"Error occurred: {e}")
