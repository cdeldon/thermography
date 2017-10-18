# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'image_saving_gui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Save_images_dialog(object):
    def setupUi(self, Save_images_dialog):
        Save_images_dialog.setObjectName("Save_images_dialog")
        Save_images_dialog.resize(380, 190)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Save_images_dialog.sizePolicy().hasHeightForWidth())
        Save_images_dialog.setSizePolicy(sizePolicy)
        Save_images_dialog.setMinimumSize(QtCore.QSize(380, 190))
        Save_images_dialog.setMaximumSize(QtCore.QSize(380, 190))
        self.horizontalLayout = QtWidgets.QHBoxLayout(Save_images_dialog)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.choose_directory_button = QtWidgets.QPushButton(Save_images_dialog)
        self.choose_directory_button.setObjectName("choose_directory_button")
        self.horizontalLayout_2.addWidget(self.choose_directory_button)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.save_directory_label = QtWidgets.QLabel(Save_images_dialog)
        self.save_directory_label.setObjectName("save_directory_label")
        self.verticalLayout.addWidget(self.save_directory_label)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.save_button = QtWidgets.QPushButton(Save_images_dialog)
        self.save_button.setEnabled(False)
        self.save_button.setObjectName("save_button")
        self.horizontalLayout_3.addWidget(self.save_button)
        spacerItem2 = QtWidgets.QSpacerItem(35, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.progress_bar_all_frames = QtWidgets.QProgressBar(Save_images_dialog)
        self.progress_bar_all_frames.setEnabled(False)
        self.progress_bar_all_frames.setMaximumSize(QtCore.QSize(16777215, 15))
        self.progress_bar_all_frames.setProperty("value", 0)
        self.progress_bar_all_frames.setTextVisible(True)
        self.progress_bar_all_frames.setObjectName("progress_bar_all_frames")
        self.verticalLayout.addWidget(self.progress_bar_all_frames)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.progress_bar_intra_frame = QtWidgets.QProgressBar(Save_images_dialog)
        self.progress_bar_intra_frame.setEnabled(False)
        self.progress_bar_intra_frame.setMaximumSize(QtCore.QSize(16777215, 8))
        self.progress_bar_intra_frame.setProperty("value", 0)
        self.progress_bar_intra_frame.setTextVisible(False)
        self.progress_bar_intra_frame.setObjectName("progress_bar_intra_frame")
        self.horizontalLayout_4.addWidget(self.progress_bar_intra_frame)
        spacerItem3 = QtWidgets.QSpacerItem(35, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(Save_images_dialog)
        QtCore.QMetaObject.connectSlotsByName(Save_images_dialog)

    def retranslateUi(self, Save_images_dialog):
        _translate = QtCore.QCoreApplication.translate
        Save_images_dialog.setWindowTitle(_translate("Save_images_dialog", "ThermoGUI - Save Images"))
        self.choose_directory_button.setText(_translate("Save_images_dialog", "Choose Output Directory"))
        self.save_directory_label.setText(_translate("Save_images_dialog", "Saving to directory: \" \""))
        self.save_button.setText(_translate("Save_images_dialog", "Save!"))

