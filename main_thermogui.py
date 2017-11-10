import sys

from PyQt5 import QtWidgets

from gui import ThermoGUI
from thermography.io import setup_logger, LogLevel

if __name__ == '__main__':
    setup_logger(console_log_level=LogLevel.INFO, file_log_level=LogLevel.DEBUG)

    app = QtWidgets.QApplication(sys.argv)
    form = ThermoGUI()
    form.show()
    app.exec_()
