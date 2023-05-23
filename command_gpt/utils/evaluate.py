from pathlib import Path
import os

from config import WORKSPACE_DIR

WORKSPACE_PATH = Path(WORKSPACE_DIR)


def get_filesystem_representation(path=WORKSPACE_PATH, verbose=False):
    """
    Return map of all files in a directory with the value being the file contents if verbose=True or a summary if verbose=False
    """
    file_system = {}

    for root, dirs, files in os.walk(path):
        current_dir = file_system
        relative_path = os.path.relpath(root, path)
        subdirs = relative_path.split(os.path.sep)

        for subdir in subdirs:
            if subdir == '.':
                continue
            if subdir not in current_dir:
                current_dir[subdir] = {}
            current_dir = current_dir[subdir]

        for file in files:
            file_path = os.path.join(root, file)
            if verbose:
                with open(file_path, 'r', errors='ignore') as file_obj:
                    current_dir[file] = file_obj.read()
            else:
                current_dir[file] = get_file_stats(file_path)

    return file_system


def get_file_stats(file_path):
    file_size = os.path.getsize(file_path)
    with open(file_path, 'r', errors='ignore') as file_obj:
        content = file_obj.read()
        file_length = len(content)

    return {
        "size": f"{file_size}b",
        "length": f"{file_length} chars",
    }
