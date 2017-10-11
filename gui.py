from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QImage, QPixmap
from PyQt4.QtCore import QThread, SIGNAL
import sys, cv2
import numpy as np
from gui import design
from thermography.io import VideoLoader
from thermography.utils.images import scale_image


class VideoLoaderThread(QThread):
    numpy_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, video_file_name, start_frame, end_frame):
        super(VideoLoaderThread, self).__init__()
        self.video_file_name = video_file_name
        self.start_frame = start_frame
        self.end_frame = end_frame

    def run(self):
        video_loader = VideoLoader(video_path=self.video_file_name, start_frame=self.start_frame,
                                   end_frame=self.end_frame)
        for frame_index, frame in enumerate(video_loader.frames):
            self.numpy_signal.emit(frame)


class ThermographyThread(QThread):
    size_signal = QtCore.pyqtSignal(tuple)
    finish_signal = QtCore.pyqtSignal(bool)
    iteration_signal = QtCore.pyqtSignal(int)

    def __init__(self, frames, image_size: tuple = (640, 512)):
        super(ThermographyThread, self).__init__()
        self.frames = frames
        self.image_size = image_size
        self.paused = False

    def run(self):
        for frame_id, frame in enumerate(self.frames):
            scaling_factor = np.mean([float(self.image_size[1 - i]) / frame.shape[i] for i in range(2)])
            frame_scale = scale_image(frame, scaling_factor)
            cv2.putText(frame_scale, str(frame_id), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
            image = QImage(frame_scale.data, frame_scale.shape[1], frame_scale.shape[0], frame_scale.strides[0],
                           QImage.Format_RGB888)
            self.emit(SIGNAL('newImage(QImage)'), image)
            self.size_signal.emit((frame_scale.shape[1], frame_scale.shape[0]))
            self.iteration_signal.emit(frame_id)
            self.msleep(200)

        self.finish_signal.emit(True)


class ThermoGUI(QtGui.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.frames = []

        self.video_loader_thread = None
        self.load_video_button.clicked.connect(self.load_video_from_file)

        self.video_player_thread = None
        self.play_video_button.clicked.connect(self.play_all_frames)
        self.stop_video_button.clicked.connect(self.stop_all_frames)

    def load_video_from_file(self):
        self.frames = []
        video_file_name = QtGui.QFileDialog.getOpenFileName(caption="Select a video",
                                                            filter="Videos (*.mov *.mp4 *.avi)")
        start_frame = self.video_from_index.value()
        end_frame = self.video_to_index.value()
        if end_frame == -1:
            end_frame = None
        self.video_loader_thread = VideoLoaderThread(video_file_name, start_frame, end_frame)
        self.video_loader_thread.numpy_signal.connect(self.add_frame)
        self.video_loader_thread.start()

    def play_all_frames(self):
        self.play_video_button.setEnabled(False)
        self.stop_video_button.setEnabled(True)
        self.thermography_thread = ThermographyThread(self.frames)
        self.video_view.connect(self.thermography_thread, SIGNAL('newImage(QImage)'), self.display_image)

        self.global_progress_bar.setMinimum(0)
        self.global_progress_bar.setMaximum(len(self.frames))
        self.thermography_thread.iteration_signal.connect(self.update_global_progress_bar)

        self.thermography_thread.size_signal.connect(self.resize_video_view)
        self.thermography_thread.finish_signal.connect(self.video_finished)
        self.thermography_thread.start()

    def stop_all_frames(self):
        self.thermography_thread.terminate()
        self.video_finished(True)

    def add_frame(self, frame):
        self.frames.append(frame)

    def update_global_progress_bar(self, frame_index: int):
        self.global_progress_bar.setValue(frame_index)

    def display_image(self, image):
        pixmap = QPixmap.fromImage(image)
        self.video_view.setPixmap(pixmap)

    def resize_video_view(self, size):
        self.video_view.setFixedSize(size[0], size[1])

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
