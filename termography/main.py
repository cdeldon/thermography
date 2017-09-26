import termography as tg
from termography.io import ImageLoader

import os

if __name__ == '__main__':
    TERMOGRAPHY_ROOT_DIR = tg.get_termography_root_dir()
    DATA_ROOT_DIR = os.path.join(os.path.dirname(TERMOGRAPHY_ROOT_DIR), "data")
    IN_FILE_NAME = os.path.join(DATA_ROOT_DIR, "checkerboard.jpg")

    image_loader = ImageLoader(image_path=IN_FILE_NAME)
    image_loader.show_raw()

