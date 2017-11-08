from PyQt5 import QtWidgets
import sys

from thermography.io import setup_logger, LogLevel
from gui import CreateDatasetGUI

if __name__ == '__main__':
    setup_logger(console_log_level=LogLevel.INFO, file_log_level=LogLevel.DEBUG)

    app = QtWidgets.QApplication(sys.argv)
    form = CreateDatasetGUI()
    form.show()
    app.exec_()
