import os
import cv2
import numpy as np

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QImage

import thermography as tg
from gui.design import Ui_CreateDataset_main_window
from gui.dialogs.about_dialog import AboutDialog
from gui.dialogs.webcam_dialog import WebCamWindow
from gui.dialogs import ThermoGuiThread


class CreateDatasetGUI(QtWidgets.QMainWindow, Ui_CreateDataset_main_window):
    """
    Dataset creation GUI.
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.set_logo_icon()

        self.thermo_thread = ThermoGuiThread()
        self.is_stoppable = True

        self.last_folder_opened = None
        self.output_folder = None

        self.connect_widgets()
        self.connect_thermo_thread()

        self.capture = None
        self.webcam_port = None

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
        self.output_path_button.clicked.connect(self.select_output_path)

        self.play_video_button.clicked.connect(self.play_all_frames)
        self.stop_video_button.clicked.connect(self.stop_all_frames)

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
        self.thermo_thread.rectangle_frame_signal.connect(lambda x: self.display_rectangle_image(x))
        self.thermo_thread.module_list_signal.connect(lambda x: self.display_all_modules(x))

        self.thermo_thread.finish_signal.connect(self.video_finished)

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

        self.thermo_thread.input_file_name = video_file_name
        self.is_stoppable = True
        self.setWindowTitle("Thermography: {}".format(video_file_name))

        start_frame = self.video_from_index.value()
        end_frame = self.video_to_index.value()
        if end_frame == -1:
            end_frame = None
        self.thermo_thread.load_video(start_frame=start_frame, end_frame=end_frame)

        self.global_progress_bar.setMinimum(0)
        self.global_progress_bar.setMaximum(len(self.thermo_thread.app.frames) - 1)

        self.thermo_thread.iteration_signal.connect(self.update_global_progress_bar)

    def select_output_path(self):
        output_directory = QtWidgets.QFileDialog.getExistingDirectory(caption="Select output directory")
        if output_directory == "":
            return

        self.output_folder = output_directory

        if len(os.listdir(self.output_folder)) > 0:
            warning_box = QtWidgets.QMessageBox(parent=self)
            warning_box.setText("Directory {} not empty! Select an empty directory!".format(self.output_folder))
            warning_box.show()
        else:
            self.selected_output_path_label.setText('Selected output path: "{}"'.format(self.output_folder))

    def play_all_frames(self):
        self.thermo_thread.is_paused = False
        self.image_scaling_slider.setEnabled(False)
        self.update_image_scaling()

        self.image_scaling_label.setText("Input image scaling: {:0.2f}".format(self.thermo_thread.app.image_scaling))
        self.play_video_button.setEnabled(False)
        if self.is_stoppable:
            self.stop_video_button.setEnabled(True)

        self.thermo_thread.start()

    def stop_all_frames(self):
        self.thermo_thread.terminate()
        self.video_finished(True)

    def pause_all_frames(self):
        self.thermo_thread.is_paused = True
        self.play_video_button.setEnabled(True)
        if self.is_stoppable:
            self.stop_video_button.setEnabled(True)

    def current_module_is_working(self):
        pass

    def current_module_is_broken(self):
        pass

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

    def display_rectangle_image(self, frame: np.ndarray):
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        image = image.scaled(self.rectangle_image_view.size(), QtCore.Qt.KeepAspectRatio,
                             QtCore.Qt.SmoothTransformation)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.rectangle_image_view.setPixmap(pixmap)

    def display_all_modules(self, module_list: list):
        self.thermo_thread.is_paused = True
        for d in module_list:
            coordinates = d["coordinates"]
            module_image = d["image"]
            self.resize_video_view(module_image.shape, self.current_module_view)
            image = QImage(module_image.data, module_image.shape[1], module_image.shape[0], module_image.strides[0],
                           QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(image)
            self.current_module_view.setPixmap(pixmap)
            self.current_module_view.repaint()
            input("Hello")

        self.thermo_thread.is_paused = False

    @staticmethod
    def resize_video_view(size, view):
        view.setFixedSize(size[1], size[0])

    def video_finished(self, finished: bool):
        self.play_video_button.setEnabled(finished)
        self.stop_video_button.setEnabled(not finished)
        self.image_scaling_slider.setEnabled(finished)

    def set_webcam_port(self, port):
        self.webcam_port = port
        self.thermo_thread.use_webcam(self.webcam_port)
        self.is_stoppable = False
        self.setWindowTitle("Thermography: Webcam")
        self.play_all_frames()

    def load_webcam(self):
        self.capture = WebCamWindow(parent=self)
        self.capture.webcam_port_signal.connect(lambda port: self.set_webcam_port(port))
        self.capture.show()
        self.capture.start()
        self.undistort_image_box.setChecked(True)
        self.undistort_image_box.setChecked(False)

    def reset_app(self):
        self.thermo_thread.terminate()
        self.thermo_thread = ThermoGuiThread()
        self.image_scaling_slider.setValue(10)
        self.video_finished(True)
        self.global_progress_bar.setValue(0)
        self.video_view.setText("Input Image")
        self.canny_edges_view.setText("Edges Image")
        self.segment_image_view.setText("Segment Image")
        self.rectangle_image_view.setText("Rectangle Image")
        self.module_image_view.setText("Module Map")
        self.module_image_view.setAlignment(QtCore.Qt.AlignCenter)
        self.capture = None
        self.webcam_port = None

        self.setWindowTitle("Thermography")

        self.connect_thermo_thread()
