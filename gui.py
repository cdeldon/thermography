from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QImage, QPixmap
from PyQt4.QtCore import QThread
import sys, os, cv2
import numpy as np
from gui import design
import thermography as tg


class ThermoGuiThread(QThread):
    iteration_signal = QtCore.pyqtSignal(int)
    finish_signal = QtCore.pyqtSignal(bool)
    last_frame_signal = QtCore.pyqtSignal(np.ndarray)
    processed_frame_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(ThermoGuiThread, self).__init__()

        self.camera_param_file_name = None
        self.module_param_file_name = None
        self.input_file_name = None

        self.load_default_paths()

        self.app = tg.App(input_video_path=self.input_file_name, camera_param_file=self.camera_param_file_name,
                          module_param_file=self.module_param_file_name)

    def load_default_paths(self):
        # Load camera parameters.
        settings_dir = tg.settings.get_settings_dir()

        self.camera_param_file_name = os.path.join(settings_dir, "camera_parameters.json")
        self.module_param_file_name = os.path.join(settings_dir, "module_parameters.json")
        tg.settings.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/")
        self.input_file_name = os.path.join(tg.settings.get_data_dir(), "Ispez Termografica Ghidoni 1.mov")

    def load_video(self, start_frame: int, end_frame: int):
        self.app = tg.App(input_video_path=self.input_file_name, camera_param_file=self.camera_param_file_name,
                          module_param_file=self.module_param_file_name)
        self.app.load_video(start_frame=start_frame, end_frame=end_frame)

    def run(self):
        for frame_id, frame in enumerate(self.app.frames):
            self.app.step(frame_id, frame)
            self.last_frame_signal.emit(self.app.last_scaled_frame_rgb)
            self.processed_frame_signal.emit(self.app.last_edges_frame)
            self.iteration_signal.emit(frame_id)

        self.finish_signal.emit(True)


class ThermoGUI(QtGui.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.thermo_thread = ThermoGuiThread()

        self.connect_widgets()
        self.connect_thermo_thread()

    def connect_widgets(self):
        self.load_video_button.clicked.connect(self.load_video_from_file)

        self.reset_button.clicked.connect(self.reset_app)
        self.play_video_button.clicked.connect(self.play_all_frames)
        self.stop_video_button.clicked.connect(self.stop_all_frames)

        self.image_scaling_slider.valueChanged.connect(self.update_image_scaling)

        self.undistort_image_box.clicked.connect(self.update_image_distortion)
        self.angle_value.valueChanged.connect(self.update_image_angle)
        self.blur_value.valueChanged.connect(self.update_blur_value)
        self.max_histeresis_value.valueChanged.connect(self.update_histeresis_params)
        self.min_histeresis_value.valueChanged.connect(self.update_histeresis_params)
        self.dilation_value.valueChanged.connect(self.update_dilation_steps)

    def connect_thermo_thread(self):
        self.thermo_thread.last_frame_signal.connect(self.display_image)
        self.thermo_thread.processed_frame_signal.connect(self.display_canny_edges)

        self.thermo_thread.finish_signal.connect(self.video_finished)

    def load_video_from_file(self):
        video_file_name = QtGui.QFileDialog.getOpenFileName(caption="Select a video",
                                                            filter="Videos (*.mov *.mp4 *.avi)")
        if video_file_name == "":
            return

        self.thermo_thread.input_file_name = video_file_name

        start_frame = self.video_from_index.value()
        end_frame = self.video_to_index.value()
        if end_frame == -1:
            end_frame = None
        self.thermo_thread.load_video(start_frame=start_frame, end_frame=end_frame)

        self.global_progress_bar.setMinimum(0)
        self.global_progress_bar.setMaximum(len(self.thermo_thread.app.frames) - 1)
        self.thermo_thread.iteration_signal.connect(self.update_global_progress_bar)

    def play_all_frames(self):
        self.update_image_scaling()
        self.image_scaling_label.setText("Input image scaling: {:0.2f}".format(self.thermo_thread.app.image_scaling))
        self.play_video_button.setEnabled(False)
        self.stop_video_button.setEnabled(True)

        self.thermo_thread.start()

    def stop_all_frames(self):
        self.thermo_thread.terminate()
        self.video_finished(True)

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

    def display_image(self, frame: np.ndarray):
        self.resize_video_view(frame.shape, self.video_view)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_view.setPixmap(pixmap)

    def display_canny_edges(self, frame: np.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self.resize_video_view(frame.shape, self.canny_edges_view)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.canny_edges_view.setPixmap(pixmap)

    @staticmethod
    def resize_video_view(size, view):
        view.setFixedSize(size[1], size[0])

    def video_finished(self, finished: bool):
        self.play_video_button.setEnabled(finished)
        self.stop_video_button.setEnabled(not finished)

    def reset_app(self):
        self.thermo_thread.terminate()
        self.thermo_thread = ThermoGuiThread()
        self.image_scaling_slider.setValue(10)
        self.video_finished(True)
        self.global_progress_bar.setValue(0)
        self.video_view.setText("Video")
        self.canny_edges_view.setText("Processed Video")

        self.connect_thermo_thread()


def main():
    app = QtGui.QApplication(sys.argv)
    form = ThermoGUI()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
