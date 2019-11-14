# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:28:33 2019

@author: Robert
"""

import cv2
import numpy as np
import os
from pathlib import Path


file = 'SEQ_0486'
fileType = ".wmv"
hasLeak = False
width = 320
height = 240
# set video file path of input video with name and extension
vid = cv2.VideoCapture('../videos/'+file+fileType)
fgbg = cv2.createBackgroundSubtractorMOG2()

if hasLeak:
    if not os.path.exists("../dataset/"+file+"_fuga"):
        os.makedirs("../dataset/"+file+"_fuga")
    #f = open('./dataset/'+file+'_fuga/'+file+".txt", "w")   # 'r' for reading and 'w' for writing
else:
    if not os.path.exists("../dataset/"+file+"_nofuga"):
        os.makedirs("../dataset/"+file+"_nofuga")
    #f = open('./dataset/'+file+'_nofuga/'+file+".txt", "w")   # 'r' for reading and 'w' for writing

#for frame identity
index = 0
while(True):
    # Extract images
    ret, frame = vid.read()
    fgmask = fgbg.apply(frame)    
    # end of frames
    if not ret: 
        #f.close()   
        break    
    if hasLeak:
        name = '../dataset/'+file+'_fuga/'+file+'_' + str(index) + '.jpg' 
        #f.write("1 " + file+'_' + str(index) + '.jpg\n')    # Write inside file 
    else:
        name = '../dataset/'+file+'_nofuga/'+file+'_' + str(index) + '.jpg' 
        #f.write("0 " + file+'_' + str(index) + '.jpg\n')    # Write inside file 
    print(name)    
    print ('Creating...' + name)
    fgmask = cv2.bitwise_not(fgmask)
    dim = (width, height)
    resized = cv2.resize(fgmask, dim, interpolation = cv2.INTER_AREA)    
    cv2.imwrite(name, resized)
    # next frame
    index += 1
