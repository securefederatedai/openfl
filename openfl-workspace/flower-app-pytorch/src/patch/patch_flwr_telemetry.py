import flwr.common.telemetry
from pathlib import Path
import os
import uuid

def _get_source_id() -> str:
    """Get existing or new source ID."""
    source_id = "unavailable"
    # Check if .flwr in home exists

    ### PATCH ###
    # REASONING: consolidate written file locations
    if os.getenv("FLWR_HOME"):
        flwr_dir = Path(os.getenv("FLWR_HOME"))
    #############
    else:
        try:
            home = flwr.common.telemetry._get_home()
        except RuntimeError:
            # If the home directory can’t be resolved, RuntimeError is raised.
            return source_id

        flwr_dir = home.joinpath(".flwr")

    # Create .flwr directory if it does not exist yet.
    try:
        flwr_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return source_id

    source_file = flwr_dir.joinpath("source")

    # If no source_file exists create one and write it
    if not source_file.exists():
        try:
            source_file.touch(exist_ok=True)
            source_file.write_text(str(uuid.uuid4()), encoding="utf-8")
        except PermissionError:
            return source_id

    source_id = source_file.read_text(encoding="utf-8").strip()

    try:
        uuid.UUID(source_id)
    except ValueError:
        source_id = "invalid"

    return source_id

flwr.common.telemetry._get_source_id = _get_source_id
