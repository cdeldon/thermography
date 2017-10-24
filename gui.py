from PyQt5 import QtWidgets
import sys
from gui import ThermoGUI, CreateDatasetGUI

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = CreateDatasetGUI()
    form.show()
    app.exec_()
