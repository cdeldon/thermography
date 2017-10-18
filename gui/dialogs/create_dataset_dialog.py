import os
import cv2
import numpy as np

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QImage

import thermography as tg
from gui.threads import ThermoDatasetCreationThread
from gui.design import Ui_CreateDataset_main_window
from gui.dialogs import AboutDialog, SaveImageDialog


class VideoLoaderThread(QtCore.QThread):
    finish_signal = QtCore.pyqtSignal(list)

    def __init__(self, video_path: str, from_index: int, to_index: int, parent=None):
        super(self.__class__, self).__init__(parent=parent)
        self.video_path = video_path
        self.from_index = from_index
        self.to_index = to_index

    def run(self):
        video_loader = tg.io.VideoLoader(self.video_path, self.from_index, self.to_index)
        self.finish_signal.emit(video_loader.frames)


class CreateDatasetGUI(QtWidgets.QMainWindow, Ui_CreateDataset_main_window):
    """
    Dataset creation GUI.
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.set_logo_icon()

        self.last_folder_opened = None
        self.frames = []
        self.last_frame_image = None
        self.current_frame_id = 0
        self.current_module_id_in_frame = 0
        self.current_frame_modules = []
        self.discarded_modules = {}
        self.accepted_modules = {}

        self.thermo_thread = None

        self.connect_widgets()

    def set_logo_icon(self):
        gui_path = os.path.join(os.path.join(tg.settings.get_thermography_root_dir(), os.pardir), "gui")
        logo_path = os.path.join(gui_path, "img/logo.png")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(logo_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

    def connect_widgets(self):

        # File buttons
        self.file_about.triggered.connect(self.open_about_window)
        self.file_exit.triggered.connect(self.deleteLater)

        # Main buttons.
        self.load_video_button.clicked.connect(self.load_video_from_file)

        self.play_video_button.clicked.connect(self.start_playing_frames)

        self.image_scaling_slider.valueChanged.connect(self.update_image_scaling)

        # Working and Broken module buttons.
        self.module_working_button.clicked.connect(self.current_module_is_working)
        self.module_broken_button.clicked.connect(self.current_module_is_broken)

        # Segment clustering.
        self.gmm_value.clicked.connect(self.update_clustering_params)
        self.knn_value.clicked.connect(self.update_clustering_params)
        self.num_clusters_value.valueChanged.connect(self.update_clustering_params)
        self.num_init_value.valueChanged.connect(self.update_clustering_params)
        self.use_angle_value.stateChanged.connect(self.update_clustering_params)
        self.use_centers_value.stateChanged.connect(self.update_clustering_params)
        self.swipe_clusters_value.stateChanged.connect(self.update_clustering_params)

        # Segment cleaning
        self.max_angle_variation_mean_value.valueChanged.connect(self.update_cluster_cleaning_params)
        self.max_merging_angle_value.valueChanged.connect(self.update_cluster_cleaning_params)
        self.max_merging_distance_value.valueChanged.connect(self.update_cluster_cleaning_params)

        # Rectangle detection.
        self.expected_ratio_value.valueChanged.connect(self.update_rectangle_detection_params)
        self.ratio_max_deviation_value.valueChanged.connect(self.update_rectangle_detection_params)
        self.min_area_value.valueChanged.connect(self.update_rectangle_detection_params)

    def connect_thermo_thread(self):
        self.thermo_thread.module_map_frame_signal.connect(lambda x: self.store_last_frame_image(x))
        self.thermo_thread.module_list_signal.connect(lambda x: self.display_all_modules(x))

    def store_last_frame_image(self, img: np.ndarray):
        self.last_frame_image = img

    def open_about_window(self):
        about = AboutDialog(parent=self)
        about.show()

    def load_video_from_file(self):
        open_directory = ""
        if self.last_folder_opened is not None:
            open_directory = self.last_folder_opened
        video_file_name, _ = QtWidgets.QFileDialog.getOpenFileName(caption="Select a video",
                                                                   filter="Videos (*.mov *.mp4 *.avi)",
                                                                   directory=open_directory)
        if video_file_name == "":
            return
        self.last_folder_opened = os.path.dirname(video_file_name)
        self.setWindowTitle("Thermography: {}".format(video_file_name))

        start_frame = self.video_from_index.value()
        end_frame = self.video_to_index.value()
        if end_frame == -1:
            end_frame = None

        video_loader_thread = VideoLoaderThread(video_path=video_file_name, from_index=start_frame, to_index=end_frame,
                                                parent=self)
        video_loader_thread.start()
        video_loader_thread.finish_signal.connect(self.video_loader_finished)

    def video_loader_finished(self, frame_list: list):
        self.frames = frame_list.copy()
        self.global_progress_bar.setMinimum(0)
        self.global_progress_bar.setMaximum(len(self.frames) - 1)

        self.play_video_button.setEnabled(True)
        self.module_working_button.setEnabled(True)
        self.module_broken_button.setEnabled(True)

    def save_module_dataset(self):
        save_dialog = SaveImageDialog(working_modules=self.accepted_modules, broken_modules=self.discarded_modules, parent=self)
        save_dialog.show()

    def start_playing_frames(self):
        self.thermo_thread = ThermoDatasetCreationThread()
        self.connect_thermo_thread()
        self.image_scaling_slider.setEnabled(False)
        self.update_image_scaling()

        self.play_video_button.setEnabled(False)
        self.stop_video_button.setEnabled(True)

        self.current_frame_id = 0
        self.current_module_id_in_frame = 0

        self.thermo_thread.processing_frame_id = self.current_frame_id
        self.thermo_thread.processing_frame = self.frames[self.current_frame_id]

        self.thermo_thread.start()

    def current_module_is_working(self):
        self.register_module(self.accepted_modules)
        self.display_next_module()

    def current_module_is_broken(self):
        self.register_module(self.discarded_modules)
        self.display_next_module()

    def register_module(self, m: dict):
        current_module = self.current_frame_modules[self.current_module_id_in_frame]
        image = current_module["image"]
        coords = current_module["coordinates"]
        moduel_id = current_module["id"]
        if moduel_id not in m:
            m[moduel_id] = []
        m[moduel_id].append({"image": image, "coordinates": coords, "frame_id": self.current_frame_id})

    def update_global_progress_bar(self, frame_index: int):
        self.global_progress_bar.setValue(frame_index)

    def update_image_scaling(self):
        image_scaling = self.image_scaling_slider.value() * 0.1
        if self.thermo_thread is not None:
            self.thermo_thread.app.image_scaling = image_scaling
        self.image_scaling_label.setText("Input image scaling: {:0.2f}".format(image_scaling))

    def update_histeresis_params(self):
        min_value = self.min_histeresis_value.value()
        max_value = self.max_histeresis_value.value()
        if max_value <= min_value:
            max_value = min_value + 1
        self.max_histeresis_value.setValue(max_value)
        self.thermo_thread.app.edge_detection_parameters.hysteresis_max_thresh = max_value
        self.thermo_thread.app.edge_detection_parameters.hysteresis_min_thresh = min_value

    def update_dilation_steps(self):
        self.thermo_thread.app.edge_detection_parameters.dilation_steps = self.dilation_value.value()

    def update_image_distortion(self):
        self.thermo_thread.app.should_undistort_image = self.undistort_image_box.isChecked()

    def update_image_angle(self):
        self.thermo_thread.app.image_rotating_angle = self.angle_value.value() * np.pi / 180
        if self.angle_value.value() == 360:
            self.angle_value.setValue(0)

    def update_blur_value(self):
        self.thermo_thread.app.gaussian_blur = self.blur_value.value()

    def update_edge_params(self):
        self.thermo_thread.app.segment_detection_parameters.d_rho = self.delta_rho_value.value()
        self.thermo_thread.app.segment_detection_parameters.d_theta = np.pi / 180 * self.delta_theta_value.value()
        self.thermo_thread.app.segment_detection_parameters.min_num_votes = self.min_votes_value.value()
        self.thermo_thread.app.segment_detection_parameters.min_line_length = self.min_length_value.value()
        self.thermo_thread.app.segment_detection_parameters.max_line_gap = self.max_gap_value.value()
        self.thermo_thread.app.segment_detection_parameters.extension_pixels = self.extend_segments_value.value()

    def update_clustering_params(self):
        self.thermo_thread.app.segment_clustering_parameters.num_init = self.num_init_value.value()
        self.thermo_thread.app.segment_clustering_parameters.swipe_clusters = self.swipe_clusters_value.isChecked()
        self.thermo_thread.app.segment_clustering_parameters.num_clusters = self.num_clusters_value.value()
        self.thermo_thread.app.segment_clustering_parameters.use_centers = self.use_centers_value.isChecked()
        self.thermo_thread.app.segment_clustering_parameters.use_angles = self.use_angle_value.isChecked()
        if self.knn_value.isChecked():
            self.thermo_thread.app.segment_clustering_parameters.cluster_type = "knn"
            self.swipe_clusters_value.setEnabled(False)
            self.num_init_value.setEnabled(True)
        elif self.gmm_value.isChecked():
            self.thermo_thread.app.segment_clustering_parameters.cluster_type = "gmm"
            self.swipe_clusters_value.setEnabled(True)
            self.num_init_value.setEnabled(False)

    def update_cluster_cleaning_params(self):
        self.thermo_thread.app.cluster_cleaning_parameters.max_angle_variation_mean = np.pi / 180 * self.max_angle_variation_mean_value.value()
        self.thermo_thread.app.cluster_cleaning_parameters.max_merging_angle = np.pi / 180 * self.max_merging_angle_value.value()
        self.thermo_thread.app.cluster_cleaning_parameters.max_endpoint_distance = np.pi / 180 * self.max_merging_distance_value.value()

    def update_rectangle_detection_params(self):
        self.thermo_thread.app.rectangle_detection_parameters.aspect_ratio = self.expected_ratio_value.value()
        self.thermo_thread.app.rectangle_detection_parameters.aspect_ratio_relative_deviation = self.ratio_max_deviation_value.value()
        self.thermo_thread.app.rectangle_detection_parameters.min_area = self.min_area_value.value()

    def display_all_modules(self, module_list: list):
        self.current_frame_modules = module_list.copy()
        self.current_module_id_in_frame = -1
        if len(self.current_frame_modules) == 0:
            self.rectangle_image_view.setText("No Module detected")
            self.rectangle_image_view.setAlignment(QtCore.Qt.AlignCenter)
            self.frame_finished()
            return
        self.display_next_module()

    def display_next_module(self):
        self.current_module_id_in_frame += 1
        if len(self.current_frame_modules) == self.current_module_id_in_frame:
            self.frame_finished()
            return

        d = self.current_frame_modules[self.current_module_id_in_frame]
        coordinates = d["coordinates"]
        mask = np.zeros_like(self.last_frame_image)
        tmp_image = self.last_frame_image.copy()
        cv2.polylines(tmp_image, np.int32([coordinates]), True, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.fillConvexPoly(mask, np.int32([coordinates]), (0, 0, 255), cv2.LINE_4)
        cv2.addWeighted(tmp_image, 0.8, mask, 0.5, 0, tmp_image)
        image = QImage(tmp_image.data, tmp_image.shape[1], tmp_image.shape[0], tmp_image.strides[0],
                       QImage.Format_RGB888)
        image = image.scaled(self.rectangle_image_view.size(), QtCore.Qt.KeepAspectRatio,
                             QtCore.Qt.SmoothTransformation)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.rectangle_image_view.setPixmap(pixmap)

        module_image = d["image"]
        self.resize_video_view(module_image.shape, self.current_module_view)
        image = QImage(module_image.data, module_image.shape[1], module_image.shape[0], module_image.strides[0],
                       QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.current_module_view.setPixmap(pixmap)
        self.current_module_view.repaint()

    @staticmethod
    def resize_video_view(size, view):
        view.setFixedSize(size[1], size[0])

    def frame_finished(self):
        print("Frame finished")
        self.current_frame_id += 1
        self.current_module_id_in_frame = 0

        self.global_progress_bar.setValue(self.current_frame_id)

        if self.current_frame_id == len(self.frames):
            _ = QtWidgets.QMessageBox.information(self, "Finished", "Analyzed all frames", QtWidgets.QMessageBox.Ok)
            self.save_module_dataset()
            return

        self.thermo_thread.processing_frame = self.frames[self.current_frame_id]
        self.thermo_thread.processing_frame_id = self.current_frame_id

        print("Starting new frame {}".format(self.current_frame_id))
        self.thermo_thread.start()
