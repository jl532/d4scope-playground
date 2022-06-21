# -*- coding: utf-8 -*-
"""

PYQT UI for generating pattern masks and such...

Created on Tue Jun 21 14:23:23 2022

@author: jason
"""

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QWidget

import sys

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.centralwidget = QWidget()
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.plotting_widget = QWidget()
        self.verticalLayout.addWidget(self.plotting_widget)
        
        self.button = QPushButton("Push for Window")
        self.verticalLayout.addWidget(self.button)
        self.setCentralWidget(self.centralwidget)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()