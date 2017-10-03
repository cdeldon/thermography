import os
from .camera import Camera
from .modules import Modules

SETTINGS_DIR = os.path.dirname(os.path.abspath(__file__))
THERMOGRAPHY_ROOT_DIR = os.path.dirname(SETTINGS_DIR)
DATA_DIR = ""
TEST_DIR = os.path.join(THERMOGRAPHY_ROOT_DIR, "test")


def get_settings_dir() -> str:
    """
    Returns the absolute path to the settings directory, i.e. the directory containing this file.
    :return: Absolute path to the settings directory.
    """
    return SETTINGS_DIR


def get_thermography_root_dir() -> str:
    """
    Returns the absolute path to the root directory of the thermography project,
    i.e. the parent directory of the settings directory.
    :return: Absolute path to the project root directory.
    """
    return THERMOGRAPHY_ROOT_DIR


def get_data_dir() -> str:
    """
    Returns the absolute path to the data directory.
    :return: Absolute path to the data directory.
    """
    if DATA_DIR is "":
        raise EnvironmentError("Data directory has not been specified."
                               "\nSpecify the directory using 'thermography.set_data_dir(data_dir)' "
                               "before using this function.")
    return DATA_DIR


def set_data_dir(data_dir: str):
    """
    Sets the data directory.
    :param data_dir: Absolute path to the data directory.
    """
    if not os.path.exists(data_dir):
        raise FileExistsError("Data directory {} does not exist.".format(data_dir))
    global DATA_DIR
    DATA_DIR = data_dir


def get_test_dir() -> str:
    """
    Returns the absolute path to the test directory.
    :return: Absolute path to the rest directory.
    """
    return TEST_DIR


__all__ = ["Camera",
           "get_data_dir",
           "get_settings_dir",
           "get_thermography_root_dir",
           "get_test_dir",
           "set_data_dir"]
