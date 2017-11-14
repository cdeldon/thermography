"""This package contains the settings and associated function used in :mod:`thermography`."""
from .paths import *
from .camera import Camera


__all__ = ["Camera",
           "get_data_dir",
           "get_log_dir",
           "get_settings_dir",
           "get_thermography_root_dir",
           "get_test_dir",
           "set_data_dir",
           "get_resources_dir"]
