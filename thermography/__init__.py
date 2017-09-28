import random
import numpy as np

random.seed(0)
np.random.seed(0)

from .settings import get_thermography_root_dir, get_settings_dir, get_data_dir, set_data_dir
from .utils import angle, line_estimate, segment_intersection, segment_min_distance, random_color, scale_image
