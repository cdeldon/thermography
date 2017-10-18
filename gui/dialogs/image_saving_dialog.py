from PyQt5 import QtWidgets, QtGui
import os
import cv2
import thermography as tg

from gui.design import Ui_Save_images_dialog


class SaveImageDialog(QtWidgets.QMainWindow, Ui_Save_images_dialog):
    def __init__(self, working_modules: dict, broken_modules: dict, misdetected_modules: dict, parent=None):
        super(self.__class__, self).__init__(parent=parent)
        self.setupUi(self)
        self.set_logo_icon()

        self.working_modules = working_modules
        self.broken_modules = broken_modules
        self.misdetected_modules = misdetected_modules

        self.output_directory = " "

        self.choose_directory_button.clicked.connect(self.open_directory_dialog)
        self.save_button.clicked.connect(self.save_module_dataset)
        self.progress_bar_all_frames.setMinimum(0)
        self.progress_bar_all_frames.setMaximum(
            len(self.working_modules.keys()) + len(self.broken_modules.keys()) + len(
                self.misdetected_modules.keys()) - 1)

    def set_logo_icon(self):
        gui_path = os.path.join(os.path.join(tg.settings.get_thermography_root_dir(), os.pardir), "gui")
        logo_path = os.path.join(gui_path, "img/logo.png")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(logo_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

    def open_directory_dialog(self):
        output_directory = QtWidgets.QFileDialog.getExistingDirectory(caption="Select dataset output directory")
        if output_directory == "":
            return

        self.output_directory = output_directory

        if len(os.listdir(self.output_directory)) > 0:
            QtWidgets.QMessageBox.warning(self, "Non empty directory",
                                          "Directory {} not empty! Select an empty directory!".format(
                                              self.output_directory), QtWidgets.QMessageBox.Ok,
                                          QtWidgets.QMessageBox.Ok)
            self.open_directory_dialog()
        else:
            self.save_directory_label.setText('Saving to directory: "{}"'.format(self.output_directory))
            self.save_button.setEnabled(True)

    def save_module_dataset(self):
        self.progress_bar_all_frames.setEnabled(True)
        self.progress_bar_intra_frame.setEnabled(True)
        button_reply = QtWidgets.QMessageBox.question(self, 'Save dataset',
                                                      "Want to save dataset to {}?".format(self.output_directory),
                                                      QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                      QtWidgets.QMessageBox.No)
        if button_reply == QtWidgets.QMessageBox.No:
            self.output_directory = None
            self.save_module_dataset()
        else:
            working_modules_output_dir = os.path.join(self.output_directory, "working")
            broken_modules_output_dir = os.path.join(self.output_directory, "broken")
            misdetected_modules_output_dir = os.path.join(self.output_directory, "misdetected")

            overall_iter = 0

            def save_modules_into_directory(module_dict: dict, directory: str):
                global overall_iter

                os.mkdir(os.path.abspath(directory))
                for module_number, (module_id, registered_modules) in enumerate(module_dict.items()):
                    print("Saving all views of module {} ({}/{})".format(module_id, module_number,
                                                                         len(module_dict.keys()) - 1))
                    self.progress_bar_all_frames.setValue(self.progress_bar_all_frames.value() + 1)
                    self.progress_bar_intra_frame.setValue(0)
                    self.progress_bar_intra_frame.setMaximum(len(registered_modules))
                    for m_index, m in enumerate(registered_modules):
                        name = "id_{0:05d}_frame_{1:05d}.jpg".format(module_id, m["frame_id"])
                        path = os.path.join(directory, name)
                        img = cv2.cvtColor(m["image"], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(path, img)
                        self.progress_bar_intra_frame.setValue(m_index + 1)

            save_modules_into_directory(self.working_modules, working_modules_output_dir)
            save_modules_into_directory(self.broken_modules, broken_modules_output_dir)
            save_modules_into_directory(self.misdetected_modules, misdetected_modules_output_dir)

        _ = QtWidgets.QMessageBox.information(self, "Saved!", "Saved all modules to {}".format(self.output_directory),
                                              QtWidgets.QMessageBox.Ok)
        self.close()
