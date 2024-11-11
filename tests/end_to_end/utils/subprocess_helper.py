# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import time
import traceback

from tests.end_to_end.utils.logger import logger as log


def run_command_background(
    cmd, return_error=False, print_stdout=False, work_dir=None, redirect_to_file=None, check_sleep=1
):
    """Execute a command and let it run in background.

    Args:
        cmd (Union[str, list]): Command to execute.
        Can be a shell type string or a list of command and args.
            e.g. ['ps', '-ef'], ['/bin/bash/', script.sh], './script.sh'
        return_error: Whether to return error message. This has no effect.
        print_stdout: If True and the process completes immediately, print the stdout.
            This is obsolete. Will always print debug output and errors.
            Output will be truncated to 10 lines.
        work_dir: Directory from which to run the command. Current directory if None.
        redirect_to_file: The file descriptor to which the STDERR and STDOUT will be written.
        check_sleep: Time in seconds to sleep before polling to make sure
            the background process is still running.

    Returns:
        Popen object of the subprocess. None, if the command completed immediately.
    """
    if isinstance(cmd, list):
        shell = False
    else:
        shell = True

    if redirect_to_file:
        output_redirect = redirect_to_file
        error_redirect = subprocess.STDOUT
    else:
        output_redirect = subprocess.PIPE
        error_redirect = subprocess.PIPE
    process = subprocess.Popen(
        cmd, stdout=output_redirect, stderr=error_redirect, shell=shell, text=True, cwd=work_dir
    )
    time.sleep(check_sleep)
    return_code = process.poll()
    if return_code is None:
        return process
    elif return_code != 0:
        if redirect_to_file:
            log.info(
                "The background process has been writing STDERR and STDOUT to a file passed in as 'redirect_to_file' arg"
            )
        else:
            error = process.stderr.read().rstrip("\n")
            log.warning(f"Error is: {error}")
            log.error(f"Error Traceback: {traceback.print_exc()}")
            raise subprocess.CalledProcessError(returncode=return_code, cmd=cmd)
    else:
        log.warning("Process for Command completed instantly.")
        if redirect_to_file:
            log.info(
                "The background process has been writing STDERR and STDOUT to a file passed in as 'redirect_to_file' arg"
            )
        else:
            output = process.stdout.read().rstrip("\n").split("\n")
            if print_stdout and output is not None:
                log.info(f"Command to run - {cmd}  output - {output}")
        return None


def run_command(
    cmd, return_error=True, print_stdout=False, work_dir=None, timeout=None, check=True
):
    """
    Execute the command using subprocess and log the output to logger.

    Args:
        cmd (str or list): The command to run.
        return_error (bool): Whether to return errors or raise them.
        print_stdout (bool): Whether to print the standard output.
        work_dir (str): The working directory for the command.
        timeout (int): The timeout in seconds for the command to complete.
        check (bool): Whether to check for errors after command execution.

    Returns:
        tuple: (return_code, output, error)
    """
    if isinstance(cmd, list):
        shell = False
    else:
        shell = True

    try:
        result = subprocess.run(
            cmd, capture_output=True, shell=shell, text=True, cwd=work_dir, check=check, timeout=timeout
        )
    except subprocess.CalledProcessError as e:
        log.error(f"Command '{cmd}' failed with return code {e.returncode}")
        log.error(f"Error output: {e.stderr}")
        if not return_error:
            raise
        return e.returncode, [], [e.stderr]
    except Exception as e:
        log.error(f"Failed to execute command '{cmd}': {str(e)}")
        log.error(f"Error Traceback: {traceback.format_exc()}")
        if not return_error:
            raise
        return -1, [], [str(e)]

    output = result.stdout.splitlines()
    error = result.stderr.splitlines()

    if result.returncode == 0:
        log.info(f"Successfully ran command: {cmd}")
        if print_stdout:
            log.info(f"Command output: {result.stdout}")
    else:
        log.error(f"Subprocess command '{cmd}' returned non-zero return_code [{result.returncode}]:")
        log.error(f"stderr: {result.stderr}")
        log.error(f"stdout: {result.stdout}")
        if not return_error:
            raise subprocess.CalledProcessError(returncode=result.returncode, cmd=cmd, stderr=result.stderr)

    return result.returncode, output, error
