import os

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from simple_logger import Logger

import thermography as tg


class ThermoDatasetCreationThread(QThread):
    """Class encapsulating the :class:`~thermography.thermo_app.ThermoApp` application used for the
    :class:`~gui.dialogs.create_dataset_dialog.CreateDatasetGUI` class.
    """

    last_frame_signal = QtCore.pyqtSignal(np.ndarray)
    """Signal emitted in every :func:`ThermoApp.step <thermography.thermo_app.ThermoApp.step>` containing the data
    representing the last processed frame."""

    module_list_signal = QtCore.pyqtSignal(list)
    """Signal emitted in every :func:`ThermoApp.step <thermography.thermo_app.ThermoApp.step>` containing the data
    representing a python list of detected modules."""

    def __init__(self):
        """Initializes the ThermoThread for dataset creation.
        """
        super(self.__class__, self).__init__()
        Logger.info("Created dataset creation ThermoThread")
        self.camera_param_file_name = None

        self.__load_default_paths()

        self.app = tg.App(input_video_path=None, camera_param_file=self.camera_param_file_name)

        self.processing_frame = None
        self.processing_frame_id = None

    def __load_default_paths(self):
        # Load camera parameters.
        settings_dir = tg.settings.get_settings_dir()

        self.camera_param_file_name = os.path.join(settings_dir, "camera_parameters.json")
        Logger.debug("Using default camera param file: {}".format(self.camera_param_file_name))
        tg.settings.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/")

    def run(self):
        """Function executed when the current thread is created."""
        if self.processing_frame_id is None:
            Logger.error("Processing frame id is None!")
            return
        if self.processing_frame is None:
            Logger.error("Processing frame is None")
            return

        Logger.debug("Processing frame id {}".format(self.processing_frame_id))
        self.app.step(self.processing_frame_id, self.processing_frame)

        self.last_frame_signal.emit(self.app.last_scaled_frame_rgb)
        self.module_list_signal.emit(self.app.create_module_list())

        self.app.reset()
