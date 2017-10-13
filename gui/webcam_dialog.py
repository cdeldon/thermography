import cv2
from PyQt4 import QtGui, QtCore

from .design import webcam_dialog_design


class WebCamWindow(QtGui.QMainWindow, webcam_dialog_design.Ui_WebCam):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent=parent)
        self.setupUi(self)

        self.webcam_value = 0
        self.cap = cv2.VideoCapture(self.webcam_value)

        self.next_button.clicked.connect(self.increase_webcam_value)
        self.previous_button.clicked.connect(self.decrease_webcam_value)
        self.ok_button.clicked.connect(self.current_webcam_value_found)

    def increase_webcam_value(self):
        self.webcam_value += 1
        self.previous_button.setEnabled(True)
        self.set_webcam()

    def decrease_webcam_value(self):
        self.webcam_value -= 1
        if self.webcam_value == 0:
            self.previous_button.setEnabled(False)
        self.set_webcam()

    def current_webcam_value_found(self):
        self.deleteLater()
        self.close()
        return self.webcam_value

    def set_webcam(self):
        self.stop()
        self.cap.release()
        self.cap = cv2.VideoCapture(self.webcam_value)
        self.start()

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
        super(QtGui.QWidget, self).deleteLater()
