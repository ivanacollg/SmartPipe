# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:28:33 2019

@author: Robert
"""

import cv2
import numpy as np
import os


file = 'SEQ_0478'
fileType = ".wmv"
hasLeak = False
pathImg = "../images/train/"
pathVid = "../videos/"
width = 320
height = 240
# set video file path of input video with name and extension
vid = cv2.VideoCapture(pathVid+file+fileType)
fgbg = cv2.createBackgroundSubtractorMOG2()

if hasLeak:
    pathImg=pathImg+"fuga/"
else:
    pathImg=pathImg+"nofuga/"    

#for frame identity
index = 0
while(True):
    # Extract images
    ret, frame = vid.read()
    fgmask = fgbg.apply(frame)    
    # end of frames
    if not ret:
        break    
    name = pathImg+file+'_' + str(index) + '.jpg'                 
    print(name)    
    print ('Creating...' + name)
    fgmask = cv2.bitwise_not(fgmask)
    dim = (width, height)
    resized = cv2.resize(fgmask, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(name, resized)
    # next frame
    index += 1
