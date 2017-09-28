import os

SETTINGS_DIR = os.path.dirname(os.path.abspath(__file__))
THERMOGRAPHY_ROOT_DIR = os.path.dirname(SETTINGS_DIR)
DATA_DIR = None


def get_settings_dir():
    return SETTINGS_DIR


def get_thermography_root_dir():
    return THERMOGRAPHY_ROOT_DIR


def get_data_dir():
    if DATA_DIR is None:
        raise EnvironmentError("Data directory has not been specified."
                               "\nSpecify the directory using 'thermography.set_data_dir(data_dir)' "
                               "before using this function.")
    return DATA_DIR


def set_data_dir(data_dir):
    if not os.path.exists(data_dir):
        raise FileExistsError("Data directory {} does not exist.".format(data_dir))
    global DATA_DIR
    DATA_DIR = data_dir
