from PyQt5 import QtCore
from PyQt5.QtCore import QThread

import thermography as tg
import cv2
import os
import numpy as np


class ThermoDatasetCreationThread(QThread):
    iteration_signal = QtCore.pyqtSignal(int)
    finish_signal = QtCore.pyqtSignal()
    last_frame_signal = QtCore.pyqtSignal(np.ndarray)
    edge_frame_signal = QtCore.pyqtSignal(np.ndarray)
    segment_frame_signal = QtCore.pyqtSignal(np.ndarray)
    rectangle_frame_signal = QtCore.pyqtSignal(np.ndarray)
    module_map_frame_signal = QtCore.pyqtSignal(np.ndarray)
    module_list_signal = QtCore.pyqtSignal(list)

    def __init__(self):
        """
        Initializes the Thermo Thread for dataset creation.
        """
        super(self.__class__, self).__init__()

        self.camera_param_file_name = None

        self.load_default_paths()

        self.app = tg.App(input_video_path=None, camera_param_file=self.camera_param_file_name)

        self.processing_frame = None
        self.processing_frame_id = None

    def load_default_paths(self):
        # Load camera parameters.
        settings_dir = tg.settings.get_settings_dir()

        self.camera_param_file_name = os.path.join(settings_dir, "camera_parameters.json")
        tg.settings.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/")

    def run(self):
        if self.processing_frame_id is None:
            print("Processing frame id is None!")
            return
        if self.processing_frame is None:
            print("Processing frame is None")
            return

        self.app.step(self.processing_frame_id, self.processing_frame)

        self.last_frame_signal.emit(self.app.last_scaled_frame_rgb)
        self.edge_frame_signal.emit(self.app.last_edges_frame)
        self.segment_frame_signal.emit(self.app.create_segment_image())
        self.rectangle_frame_signal.emit(self.app.create_rectangle_image())
        self.module_map_frame_signal.emit(self.app.create_module_map_image())
        self.iteration_signal.emit(self.processing_frame_id)
        self.module_list_signal.emit(self.app.create_module_list())

        self.app.reset()

        self.finish_signal.emit()
