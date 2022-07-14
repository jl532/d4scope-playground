# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:32:44 2022

@author: jason
"""

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QHBoxLayout, QGridLayout, QVBoxLayout, QWidget, QLineEdit, QFormLayout
from PyQt5.QtGui import QIntValidator,QDoubleValidator,QFont
from PyQt5.QtCore import Qt

import sys, os, json

import numpy as np
import cv2
import matplotlib.pyplot as plt



### main here


# first arg is file path, second arg is file import flag
# -1 is input as is
image = cv2.imread("C:/Users/jason/CODE/Image labeller/42A_SLSp_LT.tiff", -1) # input image into python! 
# image is 16 bit.
print(image.shape)

# cv2.imshow("test display 16 bit", image) #name for window, image array.
# keypress = cv2.waitKey(-1)
# cv2.destroyAllWindows()

# normalize to 8bit 

image8b = cv2.normalize(image.copy(),
                    np.zeros(image.shape),
                    0, 255,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U)

multiplier = 1
contrastEnhanced_image = np.zeros(image8b.shape)
contrastEnhanced_image = np.dot(int(multiplier), image8b)
contrastEnhanced_image = np.clip(contrastEnhanced_image, 0, 255)
contrastEnhanced_image = np.uint8(contrastEnhanced_image)

#800 rows
# 1000 columns
#image8b = image8b[200:650,350:750]

# cv2.imshow("test display 8 bit", image8b) #name for window, image array.
# keypress = cv2.waitKey(-1)
# cv2.destroyAllWindows()

# median blur
image8b_blur = cv2.medianBlur(contrastEnhanced_image, 5)

image8b_verification = cv2.cvtColor(image8b_blur, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(image8b_blur,cv2.HOUGH_GRADIENT,
                           1,
                           20,
                           param1=20,
                           param2=15,
                           minRadius=13,
                           maxRadius=18)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(image8b_verification,
               (i[0],i[1]),
               i[2],
               (0,255,0),
               2)
    # draw the center of the circle
    cv2.circle(image8b_verification,
               (i[0],i[1]),
               2,
               (0,0,255),
               3)

# cv2.imshow("test display 8 bit COLOR", image8b_verification) #name for window, image array.
# keypress = cv2.waitKey(-1)
# cv2.destroyAllWindows()

pattern8b = np.zeros((60,300))

# image, position, radii, color, fill or thickness
pattern8b = cv2.circle(pattern8b,
                        (60,30),
                        15,
                        255,
                        8)

pattern8b = cv2.circle(pattern8b,
                        (60+35,30),
                        15,
                        255,
                        8)

# pattern8b = cv2.circle(pattern8b,
#                         (60+35+105,30),
#                         15,
#                         255,
#                         -1)

# pattern8b = cv2.circle(pattern8b,
#                         (60+35+105+35,30),
#                         15,
#                         255,
#                         -1)


pattern8b = cv2.normalize(pattern8b.copy(),
                        np.zeros(shape=pattern8b.shape),
                        0, 255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)


res = cv2.matchTemplate(image8b, pattern8b, cv2.TM_CCORR_NORMED)
#cvWindow("fidpattern",res)
_, _, _, max_loc = cv2.minMaxLoc(res)


cv2.imshow("display pattern", pattern8b) #name for window, image array.

cv2.imshow("display res", res) #name for window, image array.

cv2.imshow("image8b",image8b)
keypress = cv2.waitKey(-1)
cv2.destroyAllWindows()


