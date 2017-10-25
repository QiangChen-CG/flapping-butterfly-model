# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'testui.ui'
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
        MainWindow.resize(800, 544)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 351, 491))
        self.tabWidget.setStyleSheet(_fromUtf8("font: 8pt \"MS Gothic\""))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.main_tab = QtGui.QWidget()
        self.main_tab.setObjectName(_fromUtf8("main_tab"))
        self.tabWidget.addTab(self.main_tab, _fromUtf8(""))
        self.calibration_tab = QtGui.QWidget()
        self.calibration_tab.setObjectName(_fromUtf8("calibration_tab"))
        self.calib_hsv = QtGui.QPushButton(self.calibration_tab)
        self.calib_hsv.setGeometry(QtCore.QRect(10, 30, 75, 23))
        self.calib_hsv.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.calib_hsv.setObjectName(_fromUtf8("calib_hsv"))
        self.tabWidget.addTab(self.calibration_tab, _fromUtf8(""))
        self.frame_2 = QtGui.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(580, 20, 191, 141))
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(370, 20, 191, 141))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        MainWindow.setMenuBar(self.menubar)
        self.actionExit2 = QtGui.QAction(MainWindow)
        self.actionExit2.setObjectName(_fromUtf8("actionExit2"))
        self.menuFile.addAction(self.actionExit2)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.main_tab), _translate("MainWindow", "Main", None))
        self.calib_hsv.setText(_translate("MainWindow", "HSV", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.calibration_tab), _translate("MainWindow", "Calibration", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.actionExit2.setText(_translate("MainWindow", "Exit2", None))

