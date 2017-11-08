import os

import cv2
from PyQt5 import QtGui, QtCore, QtWidgets
from simple_logger import Logger

import thermography as tg
from gui.design import Ui_WebCam


class WebcamDialog(QtWidgets.QMainWindow, Ui_WebCam):
    webcam_port_signal = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent=parent)
        Logger.info("Opened Webcam dialog")
        self.setupUi(self)
        self.set_logo_icon()

        self.webcam_value = 0
        self.cap = cv2.VideoCapture(self.webcam_value)

        self.next_button.clicked.connect(self.increase_webcam_value)
        self.previous_button.clicked.connect(self.decrease_webcam_value)
        self.ok_button.clicked.connect(self.current_webcam_value_found)

    def set_logo_icon(self):
        gui_path = os.path.join(os.path.join(tg.settings.get_thermography_root_dir(), os.pardir), "gui")
        logo_path = os.path.join(gui_path, "img/logo-webcam.png")
        Logger.debug("Setting logo <{}>".format(logo_path))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(logo_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

    def increase_webcam_value(self):
        Logger.debug("Increasing webcam port value to {}".format(self.webcam_value + 1))
        self.webcam_value += 1
        self.previous_button.setEnabled(True)
        self.set_webcam()

    def decrease_webcam_value(self):
        Logger.debug("Decreasing webcam port value to {}".format(self.webcam_value - 1))
        self.webcam_value -= 1
        if self.webcam_value == 0:
            self.previous_button.setEnabled(False)
        self.set_webcam()

    def current_webcam_value_found(self):
        self.webcam_port_signal.emit(self.webcam_value)
        self.close()

    def set_webcam(self):
        self.stop()
        self.cap.release()
        self.cap = cv2.VideoCapture(self.webcam_value)
        self.start()
        self.ok_button.setText("Use port {}!".format(self.webcam_value))

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(1000. / 30)

    def next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.webcam_view.setPixmap(pix)
        else:
            font = QtGui.QFont()
            font.setPointSize(15)
            self.webcam_view.setFont(font)
            self.webcam_view.setAlignment(QtCore.Qt.AlignCenter)
            self.webcam_view.setText("No webcam found")

    def stop(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QtWidgets, self).deleteLater()
