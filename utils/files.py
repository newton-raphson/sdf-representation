import os

def create_directory(path):
    """Create a directory if it does not exist.
    Args:
        path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path
