from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QImage, QPixmap
from PyQt4.QtCore import QThread, SIGNAL
import sys, cv2, os
import numpy as np
from gui import design
import thermography as tg
from thermography.io import VideoLoader
from thermography.utils.images import scale_image


class ThermoGuiThread(QThread):
    iteration_signal = QtCore.pyqtSignal(int)
    finish_signal = QtCore.pyqtSignal(bool)
    last_frame_signal = QtCore.pyqtSignal(np.ndarray)

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
            self.iteration_signal.emit(frame_id)

        self.finish_signal.emit(True)


class ThermoGUI(QtGui.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.thermo_thread = None

        self.load_video_button.clicked.connect(self.load_video_from_file)

        self.play_video_button.clicked.connect(self.play_all_frames)
        self.stop_video_button.clicked.connect(self.stop_all_frames)
        self.image_scaling_slider.valueChanged.connect(self.update_image_scaling)

    def load_video_from_file(self):
        video_file_name = QtGui.QFileDialog.getOpenFileName(caption="Select a video",
                                                            filter="Videos (*.mov *.mp4 *.avi)")
        if self.thermo_thread is not None:
            self.thermo_thread.terminate()
            del self.thermo_thread

        self.thermo_thread = ThermoGuiThread()
        self.thermo_thread.input_file_name = video_file_name

        start_frame = self.video_from_index.value()
        end_frame = self.video_to_index.value()
        if end_frame == -1:
            end_frame = None
        self.thermo_thread.load_video(start_frame=start_frame, end_frame=end_frame)

    def play_all_frames(self):
        self.thermo_thread.app.image_scaling = self.image_scaling_slider.value() * 0.01
        self.image_scaling_label.setText("Scaling: {}".format(self.thermo_thread.app.image_scaling))
        self.play_video_button.setEnabled(False)
        self.stop_video_button.setEnabled(True)

        self.thermo_thread.last_frame_signal.connect(self.display_image)

        self.global_progress_bar.setMinimum(0)
        self.global_progress_bar.setMaximum(len(self.thermo_thread.app.frames)-1)
        self.thermo_thread.iteration_signal.connect(self.update_global_progress_bar)

        self.thermo_thread.finish_signal.connect(self.video_finished)
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
        self.image_scaling_label.setText("Scaling: {:0.2f}".format(image_scaling))

    def display_image(self, frame: np.ndarray):
        self.resize_video_view(frame.shape)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_view.setPixmap(pixmap)

    def resize_video_view(self, size):
        print(size)
        self.video_view.setFixedSize(size[1], size[0])

    def video_finished(self, finished: bool):
        self.play_video_button.setEnabled(finished)
        self.stop_video_button.setEnabled(not finished)


def main():
    app = QtGui.QApplication(sys.argv)
    form = ThermoGUI()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
