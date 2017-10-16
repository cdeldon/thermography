# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'webcam_view.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_WebCam(object):
    def setupUi(self, WebCam):
        WebCam.setObjectName("WebCam")
        WebCam.resize(310, 265)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(WebCam.sizePolicy().hasHeightForWidth())
        WebCam.setSizePolicy(sizePolicy)
        WebCam.setMinimumSize(QtCore.QSize(200, 200))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("img/logo-webcam.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        WebCam.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(WebCam)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.webcam_view = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.webcam_view.sizePolicy().hasHeightForWidth())
        self.webcam_view.setSizePolicy(sizePolicy)
        self.webcam_view.setMinimumSize(QtCore.QSize(150, 150))
        self.webcam_view.setAutoFillBackground(True)
        self.webcam_view.setTextFormat(QtCore.Qt.RichText)
        self.webcam_view.setScaledContents(True)
        self.webcam_view.setObjectName("webcam_view")
        self.verticalLayout.addWidget(self.webcam_view)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.previous_button = QtWidgets.QPushButton(self.centralwidget)
        self.previous_button.setEnabled(False)
        self.previous_button.setObjectName("previous_button")
        self.horizontalLayout.addWidget(self.previous_button)
        self.ok_button = QtWidgets.QPushButton(self.centralwidget)
        self.ok_button.setObjectName("ok_button")
        self.horizontalLayout.addWidget(self.ok_button)
        self.next_button = QtWidgets.QPushButton(self.centralwidget)
        self.next_button.setObjectName("next_button")
        self.horizontalLayout.addWidget(self.next_button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        WebCam.setCentralWidget(self.centralwidget)

        self.retranslateUi(WebCam)
        QtCore.QMetaObject.connectSlotsByName(WebCam)

    def retranslateUi(self, WebCam):
        _translate = QtCore.QCoreApplication.translate
        WebCam.setWindowTitle(_translate("WebCam", "ThermoGUI - Webcam"))
        self.webcam_view.setText(_translate("WebCam", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">WebCam</span></p></body></html>"))
        self.previous_button.setText(_translate("WebCam", "Previous"))
        self.ok_button.setText(_translate("WebCam", "Use port 0!"))
        self.next_button.setText(_translate("WebCam", "Next"))

