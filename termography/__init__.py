import random
import numpy as np

random.seed(0)
np.random.seed(0)

from .settings import get_termography_root_dir, get_settings_dir, get_data_dir, set_data_dir
from .utils import scale_image, random_color
