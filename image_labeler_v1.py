import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler

cwd = os.getcwd()
print(cwd)
image = cv2.imread(cwd + '/cassio ovalbumin images' + '/2_drops_1_635.tif', -1)

imgRowPixels, imgColPixels = image.shape

rows = 8
cols = 3

eachBlockRowPixels = imgRowPixels // rows 
eachBlockColPixels = imgColPixels // cols

print(eachBlockRowPixels)
print(eachBlockColPixels)

imgList = []
for eachRow in range(0,imgRowPixels, eachBlockRowPixels):
    for eachCol in range(0,imgColPixels, eachBlockColPixels):
        imgList.append(image[eachRow:eachRow+eachBlockRowPixels, eachCol:eachCol+eachBlockColPixels])


fig, ax = plt.subplots()
positions = {}
def closed_report(event, klicker):
    print('Closed Figure! locations here: ')
    positions = klicker.get_positions()
    print(positions)
    
fig.canvas.mpl_connect('close_event', lambda event: closed_report(event, klicker))

ax.imshow(imgList[1], cmap="gray")
zoom_factory(ax)
ph = panhandler(fig, button=2)

klicker = clicker(
    ax,
    ["capt", "fiduc", "detect", "smear"],
    markers=["o", "x", "*", "+"]
)

# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
plt.show()