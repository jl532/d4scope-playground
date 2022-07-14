# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:36:24 2022

@author: jay
"""

# IO packages
import os
import json
import csv
import csv
import sys

# math and image processing packages
import cv2
import numpy as np
from scipy import ndimage

arrayCoords = []
def mouseLocationClick(event, x, y, flags, param):
    """
        displays the active clicked location, and the distance between two clicked locations, specifically in pixels.
        Does not correct for downsampled images.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print("click identified at: " + str([y,x]))
        arrayCoords.append([x,y])
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(arrayCoords) > 1:
            locOne = arrayCoords.pop()
            locTwo = arrayCoords.pop()
            distance = np.linalg.norm(np.array(locOne)-np.array(locTwo))
            print("distance: " + str(distance))
        else:
            print("click 2 places first")

def cvWindow(name, image, keypressBool = False, delay = -1):
    print("---Displaying: "
          +  str(name)
          + "  ---")
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(name, mouseLocationClick)
    #cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, 
     #                         cv2.WINDOW_FULLSCREEN)
    cv2.imshow(name, image)
    pressedKey = cv2.waitKey(delay)
    cv2.destroyAllWindows()
    if keypressBool:
        return pressedKey
        sys.exit()

# open file image
# import as numpy array

cwd = os.getcwd()
inputdir = (cwd + "\\images-cov\\")

#recursively go through directories and pull images to shove into cut images dir
dirList = os.listdir(inputdir)

# filePath = easygui.fileopenbox()
# print(filePath)
print(inputdir)
print(dirList)
filePath = dirList[0]
print(inputdir + filePath)
original_image = cv2.imread(inputdir + filePath,-1)

#original_image = cv2.rotate(original_image.copy(), cv2.ROTATE_180)

array_setup = {}
stringConfig = ""
with open("array_setup-flipped.json", 'r') as jsonConfigs:
    stringConfig = jsonConfigs.read()
    array_setup = json.loads(stringConfig)

fids3 = [[565, 401],[565, 435],[565, 572],[565, 607]]
fids3 = [0,0], [0,34.5],[0,172.5],[0,207]


fidPatternRows = int(np.amax([x[0] for x in fids3]) - np.amin([x[0] for x in fids3]) + 2 *  round(array_setup["row_pitch"]))
fidPatternCols = int(np.amax([x[1] for x in fids3]) - np.amin([x[1] for x in fids3]) + 2 *  round(array_setup["col_pitch"]))

fidPatternImg = np.zeros((fidPatternRows,fidPatternCols))
fidPrimeRowShift = fids3[0][0] + array_setup["row_pitch"]
fidPrimeColShift = fids3[0][1] + array_setup["col_pitch"]
for eachFiducial in fids3:
    # shift fiducials to center them between the pitch padding
    cv2.circle(fidPatternImg, 
                (round(eachFiducial[1] + fidPrimeColShift), 
                 round(eachFiducial[0] + fidPrimeRowShift)),
                array_setup["radii"], 
                255, 
                thickness = -1)

# cv2.imshow("test fiducial pattern", fidPatternImg)
# cv2.waitKey(-1)
# cv2.destroyAllWindows()
imageCols, imageRows = original_image.shape[::-1]
image8b = cv2.normalize(original_image.copy(),
                         np.zeros(shape=(imageRows, imageCols)),
                         0, 255,
                         norm_type=cv2.NORM_MINMAX,
                         dtype=cv2.CV_8U)
verImg = cv2.cvtColor(image8b.copy(), cv2.COLOR_GRAY2RGB)

fidPatternImg = cv2.normalize(fidPatternImg.copy(),
                        np.zeros(shape=(fidPatternRows, fidPatternCols)),
                        0, 255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)

#cvWindow("fidpattern",fidPatternImg)
res = cv2.matchTemplate(image8b, fidPatternImg, cv2.TM_CCORR_NORMED)
#cvWindow("fidpattern",res)
_, _, _, max_loc = cv2.minMaxLoc(res)

# cv2.rectangle takes in positions as (column, row)....
cv2.rectangle(verImg,
              max_loc,
              (max_loc[0] + fidPatternCols,
               max_loc[1] + fidPatternRows),
              (0, 105, 255),
              3)

topLeftMatch = max_loc

#cvWindow("rectangle drawn", verImg)
### implement relative positioning between fiducials and main array are
relPosRows = array_setup["top_left_coords"][0] - fids3[0][0]
relPosCols = array_setup["top_left_coords"][1] - fids3[0][1]


#mark up the image with circles to verify ROI alignment
radii = array_setup["radii"]
circle_centerRow = max_loc[1] + relPosRows + array_setup["row_pitch"]
circlePositions = []
for eachRow in range(int(array_setup["rows"])):
    circle_centerCol = max_loc[0] + relPosCols + array_setup["col_pitch"]
    for eachCol in range(int(array_setup["cols"])):
        circlePositions.append([eachRow,eachCol])
        cv2.circle(verImg, 
                    (round(circle_centerCol), round(circle_centerRow)),
                    radii, 
                    (0,0,255), 
                    thickness = 1)
        true_row = circle_centerCol + array_setup["col_pitch"]
        circle_centerCol = circle_centerCol + array_setup["col_pitch"]
    circle_centerRow = circle_centerRow + array_setup["row_pitch"]


## pattern match here
image8b = cv2.normalize(original_image.copy(),
                         np.zeros(shape=(imageRows, imageCols)),
                         0, 255,
                         norm_type=cv2.NORM_MINMAX,
                         dtype=cv2.CV_8U)

resFinal = cv2.matchTemplate(image8b, fidPatternImg, cv2.TM_CCORR_NORMED)
_, _, _, max_locResFinal = cv2.minMaxLoc(resFinal)
print(max_locResFinal)
cv2.rectangle(verImg,(max_locResFinal[0], max_locResFinal[1]),
                              (max_locResFinal[0]+fidPatternCols, max_locResFinal[1]+fidPatternRows),
                              (65000,65000,0),2)
print(fids3)
for each in fids3:
    print(each)
    print(max_locResFinal[1] + each[0] + array_setup["row_pitch"])
    print(max_locResFinal[0] + each[1] + array_setup["col_pitch"])
    cv2.circle(verImg,
               (round(max_locResFinal[0] + each[1] + array_setup["row_pitch"]),
                round(max_locResFinal[1] + each[0] + array_setup["col_pitch"])),
               radii+1,
               (65000,65000,0),1)
               
    
circle_centerRow = max_locResFinal[1] + relPosRows + array_setup["row_pitch"]
print(circle_centerRow)
#mark up the image with circles to verify ROI alignment
radii = int(array_setup["radii"])
circlePositions = []
for eachRow in range(int(array_setup["rows"])):
    circle_centerCol = max_locResFinal[0] + relPosCols + array_setup["col_pitch"]
    for eachCol in range(int(array_setup["cols"])):
        circlePositions.append([eachRow,eachCol])
        cv2.circle(verImg, 
                    (round(circle_centerCol), round(circle_centerRow)),
                    radii+1, 
                    (0,65000,65000), 
                    thickness = 1)
        circle_centerCol = circle_centerCol + array_setup["col_pitch"]
    circle_centerRow = circle_centerRow + array_setup["row_pitch"]




cv2.imshow("non-rotated", verImg)
cv2.imshow("rotated",verImg)
cv2.waitKey(-1)
cv2.destroyAllWindows()

# seems fiducial to top left is 605-332 = 273 row pixels, and 0 column (aligned)



# keep original image, but track MARKED image as well
#marked_image = original_image.copy()
checkLoop = True
while checkLoop:
    array_setup = json.loads(stringConfig)
    #array_setup["top_left_coords"] = array_setup["top_left_coords"].strip('][').split(',')
    marked_image = cv2.cvtColor((original_image.copy()/256).astype('uint8'),
                                cv2.COLOR_GRAY2BGR)
    print(marked_image.shape)
    foreground_image_mask = np.zeros(original_image.shape)
    foreground_negative_OBO = np.zeros(original_image.shape)
    background_image_masks = []
    
    # mark up the image with circles to verify ROI alignment
    circle_centerRow = int(array_setup["top_left_coords"][0])
    radii = int(array_setup["radii"])
    circlePositions = []
    for eachRow in range(int(array_setup["rows"])):
        circle_centerCol = int(array_setup["top_left_coords"][1])
        for eachCol in range(int(array_setup["cols"])):
            bg_image_mask = np.zeros(original_image.shape)
            circlePositions.append([eachRow,eachCol])
            cv2.circle(marked_image, 
                        (circle_centerCol, circle_centerRow),
                        radii, 
                        (0,0,255), 
                        thickness = 2)
            cv2.circle(foreground_image_mask, 
                        (circle_centerCol, circle_centerRow),
                        radii, 
                        255, 
                        thickness = -1)
            cv2.circle(foreground_negative_OBO, 
                        (circle_centerCol, circle_centerRow),
                        radii+2, 
                        255, 
                        thickness = -1)
            cv2.circle(bg_image_mask,
                        (circle_centerCol, circle_centerRow),
                        round(radii*2)+2, 
                        255, 
                        thickness = -1)
            background_image_masks.append(bg_image_mask)
            circle_centerCol = circle_centerCol + int(array_setup["col_pitch"])
        circle_centerRow = circle_centerRow + int(array_setup["row_pitch"])
    for each in array_setup["fiducials"]:
        print(each)
        cv2.circle(marked_image,
                    (each[1],each[0]),
                    radii,
                    (0,255,0),
                    thickness = 2)
    
    
    cv2.imshow("marked image", marked_image)
    print("press r to rerun with adjusted settings, q to proceed")
    print(stringConfig)
    cancelEarly = cv2.waitKey(0)
    cv2.destroyAllWindows()
#    if cancelEarly == ord("r"):
#        sys.exit()
    if cancelEarly == ord("q"):
        checkLoop = False

finalBGmasks = []
for eachBGMask in background_image_masks:
    finalBGmasks.append(eachBGMask-foreground_negative_OBO)
    
# 

crop_padding = 4
cropFGareaRows = [array_setup["top_left_coords"][0] - (crop_padding * array_setup["radii"]), 
                  array_setup["top_left_coords"][0] + ((array_setup["rows"]-1)*array_setup["row_pitch"]) + (crop_padding * array_setup["radii"])]

cropFGareaCols = [array_setup["top_left_coords"][1]  - (crop_padding * array_setup["radii"]), 
                  array_setup["top_left_coords"][1]   + ((array_setup["cols"]-1)*array_setup["col_pitch"]) + (crop_padding * array_setup["radii"])]

croppedFGmask = foreground_image_mask[cropFGareaRows[0]:cropFGareaRows[1],
                                      cropFGareaCols[0]:cropFGareaCols[1]]

#pattern match the cropped area to the OG image
imageCols, imageRows = original_image.shape[::-1]
stdCols, stdRows = croppedFGmask.shape[::-1]
# print("pattern std shape: " + str(pattern.shape[::-1]))
# grab dimensions of input image and convert to 8bit for manipulation
image8b = cv2.normalize(original_image.copy(),
                        np.zeros(shape=(imageRows, imageCols)),
                        0, 255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)

croppedFGmask = cv2.normalize(croppedFGmask.copy(),
                        np.zeros(shape=(imageRows, imageCols)),
                        0, 255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)

verImg = cv2.cvtColor(image8b.copy(), cv2.COLOR_GRAY2RGB)

res = cv2.matchTemplate(image8b, croppedFGmask, cv2.TM_CCORR_NORMED)

_, _, _, max_loc = cv2.minMaxLoc(res)
gausCols, gausRows = res.shape[::-1]

x, y = np.meshgrid(range(gausCols), range(gausRows))
centerRow = int((imageRows - stdRows)/2)
centerCol = int((imageCols - stdCols)/2) - 20
# print("center row and col" + " " + str(centerRow) + " " + str(centerCol))
# draws circle where the gaussian is centered.
cv2.circle(verImg, (centerCol, centerRow), 3, (0, 0, 255), 3)
sigma = 80  # inverse slope-- smaller = sharper peak, larger = dull peak
gausCenterWeight = np.exp(-((x-centerCol)**2 + (y-centerRow)**2) /
                          (2.0 * sigma**2))


_, _, _, testCenter = cv2.minMaxLoc(gausCenterWeight)
# print("gaussian center: " + str(testCenter))
weightedRes = res * gausCenterWeight

cv2.imshow("cropped foreground image", res )
cv2.imshow("croppedound image", weightedRes )
cv2.waitKey(0)
cv2.destroyAllWindows()

_, _, _, max_loc = cv2.minMaxLoc(weightedRes)
# print(max_loc)  # max loc is reported as written as column,row...
bottomRightPt = (max_loc[0] + stdCols,
                  max_loc[1] + stdRows)
# cv2.rectangle takes in positions as (column, row)....
cv2.rectangle(verImg,
              max_loc,
              bottomRightPt,
              (0, 105, 255),
              5)
cv2.imshow("rectangle drawn", verImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
topLeftMatch = max_loc  # col, row
topLeftMatch = max_loc  # col, row

# checkLoop = True
# while checkLoop:
#     array_setup = json.loads(stringConfig)
#     array_setup["top_left_coords"] = [topLeftMatch[1] + (crop_padding * array_setup["radii"]),topLeftMatch[0] + (crop_padding * array_setup["radii"])]
#     marked_image = cv2.cvtColor((original_image.copy()/256).astype('uint8'),
#                                 cv2.COLOR_GRAY2BGR)
#     print(marked_image.shape)
#     foreground_image_mask = np.zeros(original_image.shape)
#     background_image_masks = []
    
#     # mark up the image with circles to verify ROI alignment
#     circle_centerRow = int(array_setup["top_left_coords"][0])
#     radii = int(array_setup["radii"])
#     circlePositions = []
#     rowPos = 0
#     colPos = 0
#     for eachRow in range(int(array_setup["rows"])):
#         rowPos = rowPos + 1
#         circle_centerCol = int(array_setup["top_left_coords"][1])
#         colPos = 0
#         for eachCol in range(int(array_setup["cols"])):
#             colPos = colPos + 1
#             bg_image_mask = np.zeros(original_image.shape)
#             circlePositions.append([rowPos,colPos])
#             cv2.circle(marked_image, 
#                        (circle_centerCol, circle_centerRow),
#                        radii, 
#                        (0,0,255), 
#                        thickness = 2)
#             cv2.circle(foreground_image_mask, 
#                        (circle_centerCol, circle_centerRow),
#                        radii, 
#                        255, 
#                        thickness = -1)
#             cv2.circle(bg_image_mask,
#                        (circle_centerCol, circle_centerRow),
#                        round(radii*2)+2, 
#                        255, 
#                        thickness = -1)
#             background_image_masks.append(bg_image_mask)
#             circle_centerCol = circle_centerCol + int(array_setup["col_pitch"])
#         circle_centerRow = circle_centerRow + int(array_setup["row_pitch"])
    
#     cv2.imshow("marked image", marked_image)
#     print("press r to rerun with adjusted settings, q to proceed")
#     print(stringConfig)
#     stringConfig = easygui.textbox(msg="edit the variables here to adjust array setup", 
#                               title="edit ROI configuration",
#                               text=stringConfig)
#     print(stringConfig)
#     cancelEarly = cv2.waitKey(0)
#     cv2.destroyAllWindows()
# #    if cancelEarly == ord("r"):
# #        sys.exit()
#     if cancelEarly == ord("q"):
#         checkLoop = False
        
# # run scipy ndimage whatever to quantify the spots and backgrounds 


# label_im, nb_labels = ndimage.label(foreground_image_mask)
# spot_vals = ndimage.measurements.mean(original_image, label_im,
#                                           range(1, nb_labels+1))
# mean_vals = ndimage.measurements.mean(original_image, label_im)

# print("avg spot intensity: " + str(mean_vals))

# bgCalculated = []
# for eachBgMask in background_image_masks:
#     # maskedSubImg = np.multiply(eachBgMask,subImage)
#     # cvWindow("mult result", maskedSubImg, False, 0)
#     label_bgEa, _ = ndimage.label(eachBgMask)
#     mean_bgEa = ndimage.measurements.mean(original_image, label_bgEa)
#     bgCalculated.append(mean_bgEa)
        
# print("avg spot intensity: " + str(bgCalculated))
# # report values in 2 sets of row/col format (foreground, background) as indicated from above

# csvRowOut = []
# csvRowOut.append(filePath)

# sequence = array_setup["spot_index"]
# print(len(spot_vals))
# print(len(sequence))
# zipped = zip(sequence,spot_vals,bgCalculated,circlePositions)
# zipped = list(zipped)
# res = sorted(zipped, key = lambda x: x[0])

# print(res)
# with open('d4DataOutput.csv', 'a', newline='') as csvfile:
#     csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for each in res:
#         csvRowOut.append("R" + str(each[3][0]) + "C" + str(each[3][1]) + "G" + str(each[0]))
#         csvRowOut.append(each[1])
#         csvRowOut.append(each[2])
#         csvRowOut.append(each[1]-each[2])
#     csvWriter.writerow(csvRowOut)
#     print("csv written")
# # save data as json and csv for data export and analysis
# # use two csvs - foregrounds and backgrounds, and report with row and column numbers
# # let user do whatever formatting they want later, we can do that in pos





# # debug bits
# #cv2.imshow("original image", original_image)
# cv2.imshow("marked image", marked_image)
# cv2.imshow("foreground masked image", foreground_image_mask)
# cv2.imshow("cropped foreground image", croppedFGmask)
# cv2.imshow("background masked image1", finalBGmasks[3])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("ok, pattern should be made")