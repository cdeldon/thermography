from PyQt4 import QtGui
import sys
from gui import ThermoGUI

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    form = ThermoGUI()
    form.show()
    app.exec_()
