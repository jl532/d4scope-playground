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

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QHBoxLayout, QGridLayout, QVBoxLayout, QWidget, QLineEdit, QFormLayout
from PyQt5.QtGui import QIntValidator,QDoubleValidator,QFont
from PyQt5.QtCore import Qt

import sys, os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.rows = QLineEdit()
        self.rows.setValidator(QIntValidator())
        self.rows.setMaxLength(2)
        self.rows.setAlignment(Qt.AlignRight)
        self.rows.setFont(QFont("Arial",15))
        self.rows.editingFinished.connect(self.enterPress)

        self.cols = QLineEdit()
        self.cols.setValidator(QIntValidator())
        self.cols.setMaxLength(2)
        self.cols.setAlignment(Qt.AlignRight)
        self.cols.setFont(QFont("Arial",15))
        self.cols.editingFinished.connect(self.enterPress)

        self.radii = QLineEdit()
        self.radii.setValidator(QIntValidator())
        self.radii.setMaxLength(2)
        self.radii.setAlignment(Qt.AlignRight)
        self.radii.setFont(QFont("Arial",15))
        self.radii.editingFinished.connect(self.enterPress)

        self.row_pitch = QLineEdit()
        self.row_pitch.setMaxLength(4)
        self.row_pitch.setAlignment(Qt.AlignRight)
        self.row_pitch.setFont(QFont("Arial",15))
        self.row_pitch.editingFinished.connect(self.enterPress)

        self.col_pitch = QLineEdit()
        self.col_pitch.setMaxLength(4)
        self.col_pitch.setAlignment(Qt.AlignRight)
        self.col_pitch.setFont(QFont("Arial",15))
        self.col_pitch.editingFinished.connect(self.enterPress)

        self.topLeft_rowCoord = QLineEdit()
        self.topLeft_rowCoord.setValidator(QIntValidator())
        self.topLeft_rowCoord.setMaxLength(4)
        self.topLeft_rowCoord.setAlignment(Qt.AlignRight)
        self.topLeft_rowCoord.setFont(QFont("Arial",15))
        self.topLeft_rowCoord.editingFinished.connect(self.enterPress)
        self.topLeft_colCoord = QLineEdit()
        self.topLeft_colCoord.setValidator(QIntValidator())
        self.topLeft_colCoord.setMaxLength(4)
        self.topLeft_colCoord.setAlignment(Qt.AlignRight)
        self.topLeft_colCoord.setFont(QFont("Arial",15))
        self.topLeft_colCoord.editingFinished.connect(self.enterPress)

        self.bg_rows = QLineEdit()
        self.bg_rows.setValidator(QIntValidator())
        self.bg_rows.setMaxLength(4)
        self.bg_rows.setAlignment(Qt.AlignRight)
        self.bg_rows.setFont(QFont("Arial",15))
        self.bg_rows.editingFinished.connect(self.enterPress)
        self.bg_cols = QLineEdit()
        self.bg_cols.setValidator(QIntValidator())
        self.bg_cols.setMaxLength(4)
        self.bg_cols.setAlignment(Qt.AlignRight)
        self.bg_cols.setFont(QFont("Arial",15))
        self.bg_cols.editingFinished.connect(self.enterPress)
        
        self.contrast = QLineEdit()
        self.contrast.setValidator(QIntValidator())
        self.contrast.setMaxLength(3)
        self.contrast.setAlignment(Qt.AlignRight)
        self.contrast.setFont(QFont("Arial",15))
        self.contrast.editingFinished.connect(self.enterPress)

        e4 = QLineEdit()
        e4.textChanged[str].connect(self.textchanged)

        flo = QGridLayout()
        flo.addWidget(QLabel("Circle radius"), 0, 0)
        flo.addWidget(self.radii, 0, 1)
        
        flo.addWidget(QLabel("Rows"), 1, 0)
        flo.addWidget(self.rows, 1, 1)
        flo.addWidget(QLabel("Row Pitch"), 1, 2)
        flo.addWidget(self.row_pitch, 1 , 3)

        flo.addWidget(QLabel("Columns"), 2, 0)
        flo.addWidget(self.cols, 2, 1)
        flo.addWidget(QLabel("Column Pitch"), 2, 2)
        flo.addWidget(self.col_pitch, 2, 3)

        flo.addWidget(QLabel("array Top Left Row Coord"), 3, 0)
        flo.addWidget(self.topLeft_rowCoord, 3, 1)
        flo.addWidget(QLabel("array Top Left Col Coord"), 3, 2)
        flo.addWidget(self.topLeft_colCoord, 3, 3)

        flo.addWidget(QLabel("BG total rows"), 4, 0)
        flo.addWidget(self.bg_rows, 4, 1)
        flo.addWidget(QLabel("BG total cols"), 4, 2)
        flo.addWidget(self.bg_cols, 4, 3)
        flo.addWidget(QLabel("contrast"), 5, 0)
        flo.addWidget(self.contrast, 5, 1)

        self.setLayout(flo)
        self.setWindowTitle("Array Setup Menu")
        
        
        #initialize the main figure panel
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 2)
        self.fig.show()
        self.setArrayDefaults()
        

    def importImage(self):
        """
        imports the image from the os.listdir from the listed directory.
        needs to be cleaned up with a file dialog, but shouldn't be too bad

        Returns
        -------
        img_payload : dict with original 16 bit image, and converted 8bit image
        img_payload["original"]
        img_payload["8 bit"]

        """
        try:
            return self.img_payload
            
        except AttributeError:
            print("re-reading image...")
            self.img_payload = {}
            
            cwd = os.getcwd()
            inputdir = (cwd + "\\images- contrast- test\\")
            
            #recursively go through directories and pull images to shove into cut images dir
            dirList = os.listdir(inputdir)
            img1 = dirList[1]
            image = cv2.imread(inputdir + img1, -1)
            self.img_payload["original"] = image
            
            image8b = cv2.normalize(image.copy(),
                                np.zeros(image.shape),
                                0, 255,
                                norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
            self.img_payload["8 bit"] = image8b
            
            return self.img_payload
    
    def setArrayDefaults(self):
        self.rows.setText(str(defaultArraySetup["rows"]))
        self.cols.setText(str(defaultArraySetup["cols"]))
        self.radii.setText(str(defaultArraySetup["radii"]))
        self.row_pitch.setText(str(defaultArraySetup["row_pitch"]))
        self.col_pitch.setText(str(defaultArraySetup["col_pitch"]))
        self.topLeft_rowCoord.setText(str(defaultArraySetup["top_left_coords"][0]))
        self.topLeft_colCoord.setText(str(defaultArraySetup["top_left_coords"][1]))
        self.bg_rows.setText(str(defaultArraySetup["rows"]))
        self.bg_cols.setText(str(defaultArraySetup["cols"]))
        self.contrast.setText(str(1))
        
        
    def contrast_enhance(self, multiplier):
        image8b = self.img_payload["8 bit"]
        print(image8b.shape)
        contrastEnhanced_image = np.zeros(image8b.shape)
        contrastEnhanced_image = np.dot(int(multiplier), image8b)
        contrastEnhanced_image = np.clip(contrastEnhanced_image, 0, 255)
        contrastEnhanced_image = np.uint8(contrastEnhanced_image)
        return contrastEnhanced_image
        
    def plotUpdate(self):
        self.fig = plt.gcf()
        
        img_payload = self.importImage()
        
        self.ax1[0].imshow(img_payload["8 bit"],
                      cmap="gray")
        zoom_factory(self.ax1[0])
        
        if self.contrast.text():
            contrastFactor = self.contrast.text()
            contrast_img = self.contrast_enhance(contrastFactor)
            self.ax1[1].imshow(contrast_img,
                          cmap="gray")
        zoom_factory(self.ax1[1])
        
        
        verImg = cv2.cvtColor(contrast_img, cv2.COLOR_GRAY2BGR)
        
        rows = int(self.rows.text())
        cols = int(self.cols.text())
        rowPos = int(self.topLeft_rowCoord.text())
        colPos = int(self.topLeft_colCoord.text())
        radii = int(self.radii.text())
        rowPitch = float(self.row_pitch.text())
        colPitch = float(self.col_pitch.text())
        
        for eachRow in range(rows):
            colPos = int(self.topLeft_colCoord.text())
            for eachCol in range(cols):
                cv2.circle(verImg,
                           (int(colPos), int(rowPos)),
                           radii, 
                           (255,0,0), 
                           thickness = 2)
                colPos = colPos + colPitch
            rowPos = rowPos + rowPitch
                        
        self.ax2[0].imshow(verImg)
        zoom_factory(self.ax2[0])
        

        self.bg_rows.text()
        self.bg_cols.text()
        
        panhandler(self.fig, button=2)
        self.fig.canvas.draw()
        
    def textchanged(self,text):
        print("updated plot...")
        

    def enterPress(self):
        self.plotUpdate()
        print("Enter pressed")


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