import cv2
import numpy as np
import imutils
from moviepy.editor import *
from utils import *


vid = cv2.VideoCapture('1tagvideo.mp4')
t_img = cv2.imread("testudo.png")
t_img = cv2.rotate(t_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
final_out = []
count = 0
fourcc = cv2.VideoWriter_fourcc(*'DIVX')#*'XVID'
out = cv2.VideoWriter('output.mp4',fourcc, 15, (1960,1080))
while vid.isOpened():
    print("count ",count)
    ret,frame = vid.read()
    if ret==False:
            print("Video Read Completely")
            break
    
    try:
        if count%2 ==0:
            print("count in p2y", count)
            blur = cv2.GaussianBlur(frame,(51,51),0)
            gray= cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
            _,threshold= cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
            blur = cv2.GaussianBlur(threshold,(71,71),0)
            points = final_corners(blur)
            p1,p2,p3,bl,maxWidth,maxHeight,d= arcorners(points)
            H_2 = find_homography(np.array([p1,p2,p3,bl],np.float32), d)
            warped = warp(H_2,frame, maxWidth, maxHeight)
            img = cv2.resize(t_img,(warped.shape[0],warped.shape[1]))
            rows, cols, channel = img.shape
            p1 = np.array([[0,0],[0,cols],[rows,cols],[rows,0]])
            p2 = np.array([points[0],points[1],points[2],points[3]])
            h_new = find_homography(p2,p1)
            h_new_inv = np.linalg.inv(h_new)

            for i in range(0,warped.shape[1]):
                for j in range(0,warped.shape[0]):
                    x_testudo = np.array(np.matmul(h_new_inv,[i,j,1]))[0][0]
                    y_testudo = np.array(np.matmul(h_new_inv,[i,j,1]))[0][1]
                    z_testudo = np.array(np.matmul(h_new_inv,[i,j,1]))[0][2]
                    frame[int(y_testudo/z_testudo)][int(x_testudo/z_testudo)] = img[i][j]
            final_out.append(frame)
            print("appended frame")
    except:
        final_out.append(frame)
    count = count+1

for i in final_out:
    cv2.imshow("final",i)
    cv2.waitKey(50)
    out.write(i)

vid.release()
out.release()
cv2.desp2oyAllWindows()
for i in final_out:
    i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
clip = ImageSequenceClip(final_out , fps = 20)

clip.write_videofile("Testudo.mp4")