import subprocess
import time
import traceback
from utils.logger import logger as log


def run_command_background(
    cmd, return_error=True, print_stdout=True, work_dir=None, redirect_to_file=None, check_sleep=1
):
    """
    Execute a command and let it run in the background.

    Parameters:
    cmd (str or list): The command to run.
    return_error (bool): Whether to return errors or raise them.
    print_stdout (bool): Whether to print the standard output.
    work_dir (str): The working directory for the command.
    redirect_to_file (str): Path to a file to redirect stdout and stderr.
    check_sleep (int): Time to sleep before checking if the process has started.

    Returns:
    process: The subprocess object if the command is running in the background.
    tuple: (status, output) if the command completes instantly.
    """
    if isinstance(cmd, list):
        shell = False
    else:
        shell = True

    if redirect_to_file:
        with open(redirect_to_file, 'w') as file:
            try:
                process = subprocess.Popen(
                    cmd, stdout=file, stderr=subprocess.STDOUT, shell=shell, text=True, cwd=work_dir
                )
            except Exception as e:
                log.error(f"Failed to start command '{cmd}': {str(e)}")
                log.error(f"Error Traceback: {traceback.format_exc()}")
                if return_error:
                    return None
                else:
                    raise
    else:
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell, text=True, cwd=work_dir
            )
        except Exception as e:
            log.error(f"Failed to start command '{cmd}': {str(e)}")
            log.error(f"Error Traceback: {traceback.format_exc()}")
            if return_error:
                return None
            else:
                raise

    log.info(f"Running command in the background: {cmd}")

    time.sleep(check_sleep)
    return_code = process.poll()

    if return_code is None:
        return process
    elif return_code != 0:
        if redirect_to_file:
            log.info("The background process has been writing STDERR and STDOUT to the file provided.")
        else:
            error = process.stderr.read().rstrip("\n")
            if return_error:
                return return_code, error
            else:
                log.info(f"Error is: {error}")
                log.info(f"Error Traceback: {traceback.format_exc()}")
                raise subprocess.CalledProcessError(returncode=return_code, cmd=cmd)
    else:
        log.info("Process for Command completed instantly.")
        if redirect_to_file:
            log.info("The background process has been writing STDERR and STDOUT to the file provided.")
        else:
            output = process.stdout.read().rstrip("\n")
            if print_stdout:
                log.info(f"Command to run - {cmd}  output - {output}")
            return return_code, output

def run_command(
    cmd, return_error=True, print_stdout=False, work_dir=None, timeout=None, check=True
):
    """
    Execute the command using subprocess and log the output to logger.

    Parameters:
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
