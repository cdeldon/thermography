# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'thermography_gui.ui'
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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(466, 286)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.video_view = QtGui.QLabel(self.centralwidget)
        self.video_view.setMinimumSize(QtCore.QSize(250, 250))
        font = QtGui.QFont()
        font.setPointSize(26)
        self.video_view.setFont(font)
        self.video_view.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.video_view.setMouseTracking(False)
        self.video_view.setAutoFillBackground(True)
        self.video_view.setFrameShape(QtGui.QFrame.Box)
        self.video_view.setFrameShadow(QtGui.QFrame.Plain)
        self.video_view.setTextFormat(QtCore.Qt.RichText)
        self.video_view.setScaledContents(False)
        self.video_view.setAlignment(QtCore.Qt.AlignCenter)
        self.video_view.setWordWrap(False)
        self.video_view.setObjectName(_fromUtf8("video_view"))
        self.verticalLayout_3.addWidget(self.video_view)
        self.global_progress_bar = QtGui.QProgressBar(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.global_progress_bar.sizePolicy().hasHeightForWidth())
        self.global_progress_bar.setSizePolicy(sizePolicy)
        self.global_progress_bar.setMinimumSize(QtCore.QSize(0, 10))
        self.global_progress_bar.setMaximumSize(QtCore.QSize(16777215, 10))
        self.global_progress_bar.setProperty("value", 0)
        self.global_progress_bar.setTextVisible(True)
        self.global_progress_bar.setInvertedAppearance(False)
        self.global_progress_bar.setTextDirection(QtGui.QProgressBar.TopToBottom)
        self.global_progress_bar.setObjectName(_fromUtf8("global_progress_bar"))
        self.verticalLayout_3.addWidget(self.global_progress_bar)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.video_tools_frame = QtGui.QFrame(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_tools_frame.sizePolicy().hasHeightForWidth())
        self.video_tools_frame.setSizePolicy(sizePolicy)
        self.video_tools_frame.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.video_tools_frame.setFrameShape(QtGui.QFrame.Box)
        self.video_tools_frame.setFrameShadow(QtGui.QFrame.Raised)
        self.video_tools_frame.setObjectName(_fromUtf8("video_tools_frame"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.video_tools_frame)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setContentsMargins(-1, 2, -1, 2)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.video_from_index = QtGui.QSpinBox(self.video_tools_frame)
        self.video_from_index.setMaximum(100000)
        self.video_from_index.setObjectName(_fromUtf8("video_from_index"))
        self.gridLayout.addWidget(self.video_from_index, 1, 0, 1, 1)
        self.video_to_index = QtGui.QSpinBox(self.video_tools_frame)
        self.video_to_index.setMaximum(100000)
        self.video_to_index.setProperty("value", -1)
        self.video_to_index.setObjectName(_fromUtf8("video_to_index"))
        self.gridLayout.addWidget(self.video_to_index, 1, 1, 1, 1)
        self.from_video_index_label = QtGui.QLabel(self.video_tools_frame)
        self.from_video_index_label.setObjectName(_fromUtf8("from_video_index_label"))
        self.gridLayout.addWidget(self.from_video_index_label, 0, 0, 1, 1)
        self.to_video_index_label = QtGui.QLabel(self.video_tools_frame)
        self.to_video_index_label.setObjectName(_fromUtf8("to_video_index_label"))
        self.gridLayout.addWidget(self.to_video_index_label, 0, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.load_video_button = QtGui.QPushButton(self.video_tools_frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.load_video_button.sizePolicy().hasHeightForWidth())
        self.load_video_button.setSizePolicy(sizePolicy)
        self.load_video_button.setCheckable(False)
        self.load_video_button.setFlat(False)
        self.load_video_button.setObjectName(_fromUtf8("load_video_button"))
        self.verticalLayout_2.addWidget(self.load_video_button)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.play_video_button = QtGui.QPushButton(self.video_tools_frame)
        self.play_video_button.setCheckable(True)
        self.play_video_button.setObjectName(_fromUtf8("play_video_button"))
        self.horizontalLayout_3.addWidget(self.play_video_button)
        self.stop_video_button = QtGui.QPushButton(self.video_tools_frame)
        self.stop_video_button.setEnabled(False)
        self.stop_video_button.setCheckable(False)
        self.stop_video_button.setChecked(False)
        self.stop_video_button.setObjectName(_fromUtf8("stop_video_button"))
        self.horizontalLayout_3.addWidget(self.stop_video_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.load_video_button.raise_()
        self.horizontalLayout_2.addWidget(self.video_tools_frame, QtCore.Qt.AlignTop)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Thermography", None))
        self.video_view.setText(_translate("MainWindow", "Video", None))
        self.global_progress_bar.setFormat(_translate("MainWindow", "%p%", None))
        self.from_video_index_label.setText(_translate("MainWindow", "From:", None))
        self.to_video_index_label.setText(_translate("MainWindow", "To:", None))
        self.load_video_button.setText(_translate("MainWindow", "Choose Video", None))
        self.play_video_button.setText(_translate("MainWindow", "Play", None))
        self.stop_video_button.setText(_translate("MainWindow", "Stop", None))

