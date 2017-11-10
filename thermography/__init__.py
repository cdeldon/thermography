"""The thermography package implements different computer vision and image progessing algorithms for automatic detection of solar panel modules."""

# Import everything from the subpackages.
from . import detection
from . import io
from . import settings
from . import utils

from .module_map import ModuleMap
from .thermo_app import ThermoApp as App

# Seed all random number generators.
import random
import numpy as np

random.seed(0)
np.random.seed(0)
