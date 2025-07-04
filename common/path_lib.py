"""This module contains common functions for operating with directories."""
import os


def create_path_if_not_exists(path):
    """Creates a provided path if it doesn't exist.
    
    Args:
        path: Path to check existing for and create otherwise.
    """
    if not os.path.exists(path):
        os.makedirs(path)
