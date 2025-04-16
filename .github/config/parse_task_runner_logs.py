# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import json
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_log_files(log_dir):
    """
    Get all .log files in the specified directory and its subdirectories.
    Save the list of log files to a text file.
    Args:
        log_dir (str): Path to the directory containing log files.

    Returns:
        list: List of .log files found in the directory.
    """
    if not os.path.exists(log_dir):
        logger.error(f"Directory '{log_dir}' does not exist.")
        exit(1)

    log_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                log_files.append(os.path.join(root, file))
    return log_files


def run_trufflehog(log_file):
    """
    Run TruffleHog on the specified log file and return the number of unverified secrets found.
    Args:
        log_file (str): Path to the log file to scan.
    Returns:
        int: Number of unverified secrets found in the log file.
    """
    try:
        # Run TruffleHog with JSON output and capture the output
        cmd = ["trufflehog", "filesystem", log_file, "--no-update", "--json"]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=True
        )
        # Extract the last JSON object from the output
        lines = result.stderr.strip().split("\n")
        last_json = json.loads(lines[-1])
        # throw error if las_json not have unverified_secrets
        if "unverified_secrets" not in last_json:
            raise json.JSONDecodeError("unverified_secrets not found in JSON output", "", 0)
        else:
            logger.info(f"Unverified secrets found: {last_json['unverified_secrets']}")
        # Return the unverified_secrets count
        return last_json.get("unverified_secrets", 0)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running TruffleHog on file {log_file}: {e}")
        raise e
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON output for file {log_file}: {e}")
        raise e


def main(log_dir):
    """
    Main function to scan log files for unverified secrets.
    Args:
        log_dir (str): Path to the directory containing log files.
    """
    # Get all .log files
    log_files = get_log_files(log_dir)
    if not log_files:
        logger.info("No .log files found.")
        return

    # Scan each log file with TruffleHog
    for log_file in log_files:
        logger.info(f"Scanning file: {log_file}")
        unverified_secrets = run_trufflehog(log_file)

        if unverified_secrets > 0:
            logger.error(f"File '{log_file}' contains {unverified_secrets} unverified secrets.")
            exit(1)

    logger.info("All files scanned successfully. No unverified secrets found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan log files for unverified secrets.")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to the directory containing log files."
    )
    args = parser.parse_args()
    log_dir = os.path.expanduser(args.log_dir)
    main(log_dir)
