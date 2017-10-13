# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'webcam_view.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_WebCam(object):
    def setupUi(self, WebCam):
        WebCam.setObjectName(_fromUtf8("WebCam"))
        WebCam.resize(310, 265)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(WebCam.sizePolicy().hasHeightForWidth())
        WebCam.setSizePolicy(sizePolicy)
        WebCam.setMinimumSize(QtCore.QSize(200, 200))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("img/logo-webcam.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        WebCam.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(WebCam)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.webcam_view = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.webcam_view.sizePolicy().hasHeightForWidth())
        self.webcam_view.setSizePolicy(sizePolicy)
        self.webcam_view.setMinimumSize(QtCore.QSize(150, 150))
        self.webcam_view.setAutoFillBackground(True)
        self.webcam_view.setTextFormat(QtCore.Qt.RichText)
        self.webcam_view.setScaledContents(True)
        self.webcam_view.setObjectName(_fromUtf8("webcam_view"))
        self.verticalLayout.addWidget(self.webcam_view)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.previous_button = QtGui.QPushButton(self.centralwidget)
        self.previous_button.setEnabled(False)
        self.previous_button.setObjectName(_fromUtf8("previous_button"))
        self.horizontalLayout.addWidget(self.previous_button)
        self.ok_button = QtGui.QPushButton(self.centralwidget)
        self.ok_button.setObjectName(_fromUtf8("ok_button"))
        self.horizontalLayout.addWidget(self.ok_button)
        self.next_button = QtGui.QPushButton(self.centralwidget)
        self.next_button.setObjectName(_fromUtf8("next_button"))
        self.horizontalLayout.addWidget(self.next_button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        WebCam.setCentralWidget(self.centralwidget)

        self.retranslateUi(WebCam)
        QtCore.QMetaObject.connectSlotsByName(WebCam)

    def retranslateUi(self, WebCam):
        WebCam.setWindowTitle(_translate("WebCam", "ThermoGUI - Webcam", None))
        self.webcam_view.setText(_translate("WebCam", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">WebCam</span></p></body></html>", None))
        self.previous_button.setText(_translate("WebCam", "Previous", None))
        self.ok_button.setText(_translate("WebCam", "OK!", None))
        self.next_button.setText(_translate("WebCam", "Next", None))

