from PyQt5 import QtWidgets
import sys
from gui import ThermoGUI

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = ThermoGUI()
    form.show()
    app.exec_()
