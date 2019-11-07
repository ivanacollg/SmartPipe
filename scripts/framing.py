# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:28:33 2019

@author: Robert
"""

import cv2
import numpy as np
import os


file = 'SEQ_0490'
fileType = ".wmv"
hasLeak = True
width = 320
height = 240
# set video file path of input video with name and extension
vid = cv2.VideoCapture(file+'.wmv')
fgbg = cv2.createBackgroundSubtractorMOG2()

if not os.path.exists(file):
    os.makedirs(file)

f = open('./'+file+'/'+file+".txt", "w")   # 'r' for reading and 'w' for writing

#for frame identity
index = 0
while(True):
    # Extract images
    ret, frame = vid.read()
    fgmask = fgbg.apply(frame)
    # end of frames
    if not ret: 
        f.close()   
        break
    # Saves images
    name = './'+file+'/'+file+'_' + str(index) + '.jpg'
    print ('Creating...' + name)
    fgmask = cv2.bitwise_not(fgmask)
    dim = (width, height)
    # resize image
    resized = cv2.resize(fgmask, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(name, resized)
    if hasLeak:
        f.write("1 " + file+'_' + str(index) + '.jpg\n')    # Write inside file 
    else:
        f.write("0 " + file+'_' + str(index) + '.jpg\n')    # Write inside file 
    # next frame
    index += 1