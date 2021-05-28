import os
import subprocess
from pathlib import Path

from click import group, option


@group()
def autocomplete():
    pass


@autocomplete.command()
@option('-f', '--force', required=False,
        help='Enable Intel SGX Enclave', is_flag=True, default=False)
def activate(force: bool):
    rcfile = create_fxautocompleterc(force)
    restart_bash(rcfile)


def create_fxautocompleterc(force: bool):
    """
    /bin/bash --rcfile ~/.pycharmrc
    """
    fxautocompleterc_path = Path.home() / '.fxautocompleterc'
    fx_completion_path = Path(__file__).parent.parent.parent.resolve() / 'fx-complete.sh'
    if force or not fxautocompleterc_path.is_file():
        virtual_env = os.getenv('VIRTUAL_ENV')

        with open(fx_completion_path) as f:
            fx_completion = f.readlines()

        with open(fxautocompleterc_path, 'w+') as f:
            f.write('source ~/.bashrc' + os.linesep)
            f.write('export FXAUTO=1' + os.linesep)
            # f.write(f'PATH="{virtual_env}/bin:$PATH"')
            f.write(f'source {virtual_env}/bin/activate' + os.linesep + os.linesep)
            f.writelines(fx_completion)
            f.write(os.linesep)

    return fxautocompleterc_path


def restart_bash(rcfile: str):
    print(f'bash --rcfile {rcfile}')
    subprocess.call(f'bash --rcfile {rcfile}', shell=True)
