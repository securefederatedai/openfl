import flwr.cli.build
from flwr.cli.build import write_to_zip, get_fab_filename
from typing import Annotated, Optional
import typer
from pathlib import Path
from flwr.cli.utils import is_valid_project_name
from flwr.cli.config_utils import load_and_validate
import tempfile
import zipfile
from flwr.common.constant import FAB_ALLOWED_EXTENSIONS
import shutil
import tomli_w
import hashlib
import os
from src.util import is_safe_path

def build(
    app: Annotated[
        Optional[Path],
        typer.Option(help="Path of the Flower App to bundle into a FAB"),
    ] = None,
) -> tuple[str, str]:
    """Build a Flower App into a Flower App Bundle (FAB).

    You can run ``flwr build`` without any arguments to bundle the app located in the
    current directory. Alternatively, you can specify a path using the ``--app``
    option to bundle an app located at the provided path. For example:

    ``flwr build --app ./apps/flower-hello-world``.
    """
    ### PATCH ###
    # # REASONING: original code writes to /tmp/ by default. Writing to flwr_home allows us to consolidate written files
    # # This is useful for running in an SGX enclave with Gramine since we need to strictly control allowed/trusted files
    flwr_home = os.getenv("FLWR_HOME")
    #################################

    if app is None:
        app = Path.cwd()

    app = app.resolve()
    if not app.is_dir():
        typer.secho(
            f"❌ The path {app} is not a valid path to a Flower app.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    if not is_valid_project_name(app.name):
        typer.secho(
            f"❌ The project name {app.name} is invalid, "
            "a valid project name must start with a letter, "
            "and can only contain letters, digits, and hyphens.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    conf, errors, warnings = load_and_validate(app / "pyproject.toml")
    if conf is None:
        typer.secho(
            "Project configuration could not be loaded.\npyproject.toml is invalid:\n"
            + "\n".join([f"- {line}" for line in errors]),
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=1)

    if warnings:
        typer.secho(
            "Project configuration is missing the following "
            "recommended properties:\n" + "\n".join([f"- {line}" for line in warnings]),
            fg=typer.colors.RED,
            bold=True,
        )

    # Load .gitignore rules if present
    ignore_spec = flwr.cli.build._load_gitignore(app)

    list_file_content = ""

    # Remove the 'federations' field from 'tool.flwr' if it exists
    if (
        "tool" in conf
        and "flwr" in conf["tool"]
        and "federations" in conf["tool"]["flwr"]
    ):
        del conf["tool"]["flwr"]["federations"]

    toml_contents = tomli_w.dumps(conf)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_filename = temp_file.name

        with zipfile.ZipFile(temp_filename, "w", zipfile.ZIP_DEFLATED) as fab_file:
            write_to_zip(fab_file, "pyproject.toml", toml_contents)

            # Continue with adding other files
            all_files = [
                f
                for f in app.rglob("*")
                if not ignore_spec.match_file(f)
                and f.name != temp_filename
                and f.suffix in FAB_ALLOWED_EXTENSIONS
                and f.name != "pyproject.toml"  # Exclude the original pyproject.toml
            ]

            all_files.sort()

            for file_path in all_files:
                # Read the file content manually
                with open(file_path, "rb") as f:
                    file_contents = f.read()

                archive_path = file_path.relative_to(app)
                write_to_zip(fab_file, str(archive_path), file_contents)

                # Calculate file info
                sha256_hash = hashlib.sha256(file_contents).hexdigest()
                file_size_bits = os.path.getsize(file_path) * 8  # size in bits
                list_file_content += f"{archive_path},{sha256_hash},{file_size_bits}\n"

            # Add CONTENT and CONTENT.jwt to the zip file
            write_to_zip(fab_file, ".info/CONTENT", list_file_content)

    # Get hash of FAB file
    content = Path(temp_filename).read_bytes()
    fab_hash = hashlib.sha256(content).hexdigest()

    # Set the name of the zip file
    fab_filename = get_fab_filename(conf, fab_hash)

    ### PATCH ###
    # # REASONING: original code writes to /tmp/ by default. Writing to flwr_home allows us to consolidate written files
    if not os.path.isdir(flwr_home):
        raise ValueError("Invalid directory")

    if not is_safe_path(fab_filename):
        raise ValueError("Invalid filename")

    final_path = os.path.join(flwr_home, fab_filename)
    shutil.move(temp_filename, final_path)
    #################################

    typer.secho(
        f"🎊 Successfully built {fab_filename}", fg=typer.colors.GREEN, bold=True
    )

    ### PATCH ###
    # return final_path
    return final_path, fab_hash
    ################

flwr.cli.build.build = build
