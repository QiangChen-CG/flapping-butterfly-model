# Makes parent directory the working directory
os.chdir('..')

from PyQt4 import QtGui, QtCore
import testui2

app = QApplicationWidget()
w = Ui_MainWindow()

w.MainWindow()