# -*- coding: utf-8 -*-
"""

PYQT UI for generating pattern masks and such...

Created on Tue Jun 21 14:23:23 2022

@author: jason
"""
defaultArraySetup = {"rows": 7,
                    "cols": 7,
                    "radii":  15,
                    "row_pitch": 34.5,
                    "col_pitch": 34.5,
                    "top_left_coords": [302,394],
                    "fiducials":[[232,393],[232,427],[232,599],[232,565]],
                    "spot_index":[4,1,4,1,3,1,2,
                    	      3,3,4,3,2,2,1,
                    	      4,2,2,1,4,3,4,
                    	      3,4,3,2,4,1,2,
                    	      3,1,4,1,2,3,3,
                    	      4,1,3,2,1,1,2,
                    	      2,4,1,4,1,3,2]}

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QWidget

import sys

import pyqtgraph as pg
import numpy as np
import cv2

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.centralwidget = QWidget()
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        
        self.im_widget = pg.ImageView(self)
        self.im_widget.ui.roiBtn.hide()
        self.im_widget.ui.menuBtn.hide()
        self.plotting_widget = QWidget()
        self.plotting_widget.setLayout(QVBoxLayout())
        self.plotting_widget.layout().addWidget(self.im_widget)
        self.im_widget.show()
        self.verticalLayout.addWidget(self.plotting_widget)
        
        self.button = QPushButton("Push for Window")
        self.verticalLayout.addWidget(self.button)
        self.setCentralWidget(self.centralwidget)
        
        self.genericImage = np.zeros((800,600,3))
        self.genericImage = cv2.circle(self.genericImage, 
                            (400, 300),
                            50,
                            (0,255,0),
                            -1
                            )
        self.genericImage = cv2.circle(self.genericImage, 
                            (0, 300),
                            50,
                            (0,255,255),
                            -1
                            )
        self.genericImage = cv2.circle(self.genericImage, 
                            (400, 0),
                            50,
                            (255,255,0),
                            -1
                            )
        
        self.im_widget.setImage(self.genericImage)
            


def main():
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    frame = MainWindow()
    frame.show()
    app.exec_()
    app.quit()

if __name__ == '__main__':
    main()