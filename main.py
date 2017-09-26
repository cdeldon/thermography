import termography as tg
from termography.io import ImageLoader

import os

if __name__ == '__main__':
    TERMOGRAPHY_ROOT_DIR = tg.get_termography_root_dir()
    tg.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/foto FLIR")
    IN_FILE_NAME = os.path.join(tg.get_data_dir(), "Hotspots.jpg")

    image_loader = ImageLoader(image_path=IN_FILE_NAME)
    image_loader.show_raw()

