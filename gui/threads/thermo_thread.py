import os

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from simple_logger import Logger

import thermography as tg


class ThermoGuiThread(QThread):
    iteration_signal = QtCore.pyqtSignal(int)
    finish_signal = QtCore.pyqtSignal(bool)
    last_frame_signal = QtCore.pyqtSignal(np.ndarray)
    edge_frame_signal = QtCore.pyqtSignal(np.ndarray)
    segment_frame_signal = QtCore.pyqtSignal(np.ndarray)
    rectangle_frame_signal = QtCore.pyqtSignal(np.ndarray)
    module_map_frame_signal = QtCore.pyqtSignal(np.ndarray)
    classes_frame_signal = QtCore.pyqtSignal(np.ndarray)
    module_list_signal = QtCore.pyqtSignal(list)

    def __init__(self):
        """
        Initializes the Thermo Thread.
        """
        super(ThermoGuiThread, self).__init__()
        Logger.info("Created thermoGUI thread")

        self.camera_param_file_name = None
        self.input_file_name = None

        self.pause_time = 50
        self.is_paused = False

        self.webcam_port = None
        self.cap = None
        self.should_use_webcam = False

        self.load_default_paths()

        self.app = tg.App(input_video_path=self.input_file_name, camera_param_file=self.camera_param_file_name)

    def use_webcam(self, webcam_port: int):
        Logger.debug("Thermo thread uses webcam port {}".format(webcam_port))
        self.webcam_port = webcam_port
        self.cap = cv2.VideoCapture(self.webcam_port)
        self.should_use_webcam = True

    def load_default_paths(self):
        # Load camera parameters.
        settings_dir = tg.settings.get_settings_dir()

        self.camera_param_file_name = os.path.join(settings_dir, "camera_parameters.json")
        tg.settings.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/")
        self.input_file_name = os.path.join(tg.settings.get_data_dir(), "Ispez Termografica Ghidoni 1.mov")
        Logger.debug("Using default camera param file: {}\n"
                     "Default input file name: {}".format(self.camera_param_file_name, self.input_file_name))

    def load_video(self, start_frame: int, end_frame: int):
        self.app = tg.App(input_video_path=self.input_file_name, camera_param_file=self.camera_param_file_name)
        self.app.load_video(start_frame=start_frame, end_frame=end_frame)

    def run(self):
        if self.should_use_webcam:
            frame_id = 0
            while True:
                while self.is_paused:
                    self.msleep(self.pause_time)

                ret, frame = self.cap.read()
                if ret:
                    Logger.debug("Using webcam frame {}".format(frame_id))
                    self.app.step(frame_id, frame)

                    self.last_frame_signal.emit(self.app.last_scaled_frame_rgb)
                    self.edge_frame_signal.emit(self.app.last_edges_frame)
                    self.segment_frame_signal.emit(self.app.create_segment_image())
                    self.rectangle_frame_signal.emit(self.app.create_rectangle_image())
                    self.module_map_frame_signal.emit(self.app.create_module_map_image())
                    frame_id += 1

                    self.app.reset()
        else:
            for frame_id, frame in enumerate(self.app.frames):
                while self.is_paused:
                    self.msleep(self.pause_time)

                Logger.debug("Using video frame {}".format(frame_id))
                # Perfom one step in the input video (i.e. analyze one frame)
                self.app.step(frame_id, frame)
                # Perform inference (classification on the detected modules)
                self.app.classify_detected_modules()

                self.last_frame_signal.emit(self.app.last_scaled_frame_rgb)
                self.edge_frame_signal.emit(self.app.last_edges_frame)
                self.segment_frame_signal.emit(self.app.create_segment_image())
                self.rectangle_frame_signal.emit(self.app.create_rectangle_image())
                self.module_map_frame_signal.emit(self.app.create_module_map_image())
                self.classes_frame_signal.emit(self.app.create_classes_image())
                self.iteration_signal.emit(frame_id)
                self.module_list_signal.emit(self.app.create_module_list())

                self.app.reset()

        self.finish_signal.emit(True)
