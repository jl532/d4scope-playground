# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:00:27 2022

@author: jason
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2


cwd = os.getcwd()
dirList = os.listdir()
outputDir = (cwd + "\\cut images\\")

#recursively go through directories and pull images to shove into cut images dir
dirList = os.listdir(cwd + "\\images- all\\")
imageDirFullList = []

for each in dirList:
    imageDirFullList.append(cwd + "\\images- all\\" + each)
print(cwd)
print(dirList)
print(imageDirFullList)

for eachImageDir in imageDirFullList:
    image = cv2.imread(eachImageDir, -1)

    imgRowPixels, imgColPixels = image.shape
    print(image.shape)
    rows = 8
    cols = 3

    eachBlockRowPixels = imgRowPixels // rows 
    eachBlockColPixels = imgColPixels // cols

    print(eachBlockRowPixels)
    print(eachBlockColPixels)

    imgList = []
    for eachRow in range(0,imgRowPixels-3, eachBlockRowPixels):
        for eachCol in range(0,imgColPixels-3, eachBlockColPixels):
        #    print(str(eachRow) + " " + str(eachRow+eachBlockRowPixels))
            outImg = image[eachRow:eachRow+eachBlockRowPixels, eachCol:eachCol+eachBlockColPixels]
            imgList.append(outImg)
            outputFileName = outputDir + eachImageDir[:-4].split("\\")[-1] + "_R" + str(eachRow//eachBlockRowPixels) + "_C" + str(eachCol//eachBlockColPixels) + ".tif"
            cv2.imwrite(outputFileName, outImg)
            print("output successful: " + outputFileName)

