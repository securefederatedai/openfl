import re

def is_safe_path(path):
    """
    Validate the path to ensure it contains only allowed characters.

    Args:
        path (str): The path to validate.

    Returns:
        bool: True if the path is safe, False otherwise.
    """
    return re.match(r'^[\w\-/\.]+$', path) is not None
