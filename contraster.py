# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:18:55 2022

@author: jason
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler

cwd = os.getcwd()
inputdir = (cwd + "\\images- contrast- test\\")

#recursively go through directories and pull images to shove into cut images dir
dirList = os.listdir(inputdir)
img1 = dirList[1]
image = cv2.imread(inputdir+img1, -1)

image8b = cv2.normalize(image.copy(),
                            np.zeros(image.shape),
                            0, 255,
                            norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U)


new_image = np.zeros(image8b.shape, image.dtype)
print(image8b.shape)


fig, (ax1, ax2) = plt.subplots(2, 2)

ax1[0].imshow(image8b, cmap="gray")
zoom_factory(ax1[0])
ph = panhandler(fig, button=2)
#plt.show()


alpha = 30 # Simple contrast control
beta = 0    # Simple brightness control

new_image = np.dot(alpha, image8b)
new_image = new_image + beta
new_image = np.clip(new_image, 0, 255)
# for y in range(image8b.shape[0]):
#     for x in range(image8b.shape[1]):
#         new_image[y,x] = np.clip(alpha*image8b[y,x] + beta, 0, 255)

ax2[0].imshow(new_image, cmap="gray")
zoom_factory(ax2[0])
ph = panhandler(fig, button=2)

print(new_image.dtype)
new_image = new_image.astype(np.uint8)
print(new_image.dtype)

edges = cv2.Canny(image=new_image, threshold1=100, threshold2 = 200)
ax1[1].imshow(edges, cmap="gray")
zoom_factory(ax1[1])
ph = panhandler(fig, button=2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3
flat_img = new_image.reshape((-1, 1))
flat_img = np.float32(flat_img)
print(flat_img.shape)
_, labels, (centers) = cv2.kmeans(flat_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

seg_img = centers[labels.flatten()]

seg_img = seg_img.reshape(new_image.shape)

######## otsu test

blur = cv2.medianBlur(image8b, 5)
blur = np.dot(130,blur)
blur = np.clip(blur, 0, 255)
blur = np.uint8(blur)
verImg = cv2.cvtColor(blur,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,40,
                            param1=14,param2=8,minRadius=14,maxRadius=17)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(verImg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(verImg,(i[0],i[1]),2,(0,0,255),3)
    
#ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_TOZERO)

ax2[1].imshow(verImg, cmap="gray"),
ax1[1].imshow(blur, cmap="gray")

kernel = np.ones((5,5), np.uint8)
blur = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
blur = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)



blurCan = cv2.Canny(image=blur, threshold1=50, threshold2 = 200)
#blurCan = cv2.morphologyEx(blurCan, cv2.MORPH_OPEN, kernel)

verImg2 = cv2.cvtColor(blurCan.copy(),cv2.COLOR_GRAY2BGR)
contours, hierarchy = cv2.findContours(blurCan, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#print(contours[0])
verImg2 = cv2.drawContours(verImg2, contours[0], -1, (0,255,0), 2)
#print(contours)
#contours = contours[0] if len(contours) == 2 else contours[1]
#big_contour = max(contours, key=cv2.contourArea)

# draw white filled contour on black background
#result = np.zeros_like(blurCan)
#cv2.drawContours(result, contours, 0, (255,255,255), cv2.FILLED)


#####################################################################
#blob detector
####################################
im = image8b.copy()

## inputs - radii, width/height of background, maybe emphasized ring?
kern = {"radii": 17,
        "widhei": 70}
kernel = np.zeros((kern["widhei"], kern["widhei"]), dtype=np.uint8)
kernel = cv2.circle(kernel, 
                    (kern["widhei"]//2, kern["widhei"]//2,),
                    kern["radii"],
                    255,
                    -1
                    )

kernel = cv2.circle(kernel, 
                    (kern["widhei"]//2, kern["widhei"]//2,),
                    kern["radii"]-4,
                    100,
                    -1
                    )

res = cv2.matchTemplate(im, kernel, cv2.TM_CCORR_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(res)

ax1[1].imshow(res, cmap="gray")

#ax4.imshow(seg_img, cmap="gray"),
zoom_factory(ax2[1])

plt.show()

