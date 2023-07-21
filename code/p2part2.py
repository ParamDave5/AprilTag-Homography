#!/usr/bin/env python3

import cv2
import matplotlib.pylab as plt
import numpy as np
from scipy import fft
import operator
import math
from functools import reduce
import imutils
from moviepy.editor import *
from utils import *

#no  long time


def cubesup(im_org,size = 64):
    K =np.array( [[1346.100595	,0,	        932.1633975],   
    [ 0,	  1355.933136, 654.8986796],
    [0,	      0	,    1 ] ])
    finalimg = im_org.copy()
    blur = cv2.GaussianBlur(finalimg,(51,51),0)
    _, binary = cv2.threshold(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY) 
    
    blur2 = cv2.GaussianBlur(binary,(71,71),0)
    corners = tag_corners(blur2)
    
    reference_corners = np.array([[0,0],[size,0],[size,size],[0,size]])

    H = findhomography(np.float32(corners), np.float32(reference_corners))   # custom HomoGraphy

    P , _ , _ = projectionMatrix(np.linalg.inv(H), K)
    XY = getcubecoor(P,128)
    finalimg = drawcube(finalimg, XY)
         
    return finalimg
prev_points = []
vid = cv2.VideoCapture('1tagvideo.mp4')
t_img = cv2.imread("testudo.png")


outputs = []
counter = 0

while(1):
    ret, frame = vid.read()
    if ret:
        try:
            finalimg = cubesup(frame,size = 128)
            outputs.append(finalimg)
            cv2.imshow("finalimg",finalimg)
            cv2.waitKey(1)  
        
            counter +=1
            if counter%100 == 0:
                print(" Running frame by frame:", counter)
        except:
            continue                
    else:
        print("File run complete")
        break        
vid.release()

clip = ImageSequenceClip(outputs , fps = 20)
clip.write_videofile("Cube.mp4")
