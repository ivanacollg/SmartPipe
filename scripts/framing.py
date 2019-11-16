# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:28:33 2019

@author: Robert
"""

import cv2
import numpy as np
from random import random
from vidstab import VidStab
import math

import os


file = 'SEQ_0508'
fileStab = file+'_stab'
fileType = ".wmv"
hasLeak = True
pathImgTest = "../images/validation/"
pathImgTrain = "../images/training/"
pathVid = "../videos/grayscale/"
width = 320
height = 240


# set video file path of input video with name and extension
vid = cv2.VideoCapture(pathVid+fileStab+fileType)
#Declare MOG Background Substractor
fgbg = cv2.createBackgroundSubtractorMOG2()
if hasLeak:
    pathImgTest=pathImgTest+"fuga/"
    pathImgTrain=pathImgTrain+"fuga/"
else:   
    pathImgTest=pathImgTest+"nofuga/"
    pathImgTrain=pathImgTrain+"nofuga/"

length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )
testN = math.ceil(length*.2)
trainN = length-testN
print(testN, trainN)
index = 0

while(True):
    ret, frame = vid.read()
    fgmask = fgbg.apply(frame)
    if not ret:
        break
    if(random() > 0.5 and testN>0):
        name = pathImgTest+file+'_' + str(index) + '.jpg'
        testN-=1
    else:
        name = pathImgTrain+file+'_' + str(index) + '.jpg'
    
    print(name)
    print ('Creating...' + name)        
    fgmask = cv2.bitwise_not(fgmask)    
    dim = (width, height)
    resized = cv2.resize(fgmask, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(name, resized)    
    index += 1

vid.release()
