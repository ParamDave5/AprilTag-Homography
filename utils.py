from re import I
from cv2 import transform, warpPerspective
import numpy as np
import cv2
import imutils
import operator
from functools import reduce
import matplotlib.pylab as plt
from moviepy.editor import *
import math
import random

#converts a path to gray scale
def grayimage(imagepath):
    image = cv2.imread('imagepath')
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    return gray


#return thresholded image from grayscale image
def threshold(grayimage):
    ret,thresh = cv2.threshold(grayimage,220,255,cv2.THRESH_BINARY)
    return thresh

#returns corners and image with corners
def corner(grayimage , colorimage):
    image = colorimage.copy
    dest = cv2.cornerHarris(grayimage, 2, 5, 0.07)
    dest = cv2.dilate(dest, None)
    image[dest > 0.01 * dest.max()]=[0, 0, 255]
    return dest , image



#p1 = source , p2 = destination
def findhomography(points1 , points2):
    
    if (len(points1) != len(points2)):
        print("Numbers are not same")

    A = []   
    for i in range(len(points1)):
        x , y = points1[i][0] , points1[i][1]
        u , v = points2[i][0] , points2[i][1]

        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

    A = np.asarray(A)
    U , s , V = np.linalg.svd(A)
    h = V[-1,:]/V[-1,-1]
    H = h.reshape(3,3)

    return H


def inverse_homography(H):
    return np.linalg.inv(H)

def find_homography(input_pts, output_pts):
    x1 = input_pts[0][0]
    x2 = input_pts[1][0]
    x3 = input_pts[2][0]
    x4 = input_pts[3][0]
    y1 = input_pts[0][1]
    y2 = input_pts[1][1]
    y3 = input_pts[2][1]
    y4 = input_pts[3][1]
    xp1 = output_pts[0][0]
    xp2 = output_pts[1][0]
    xp3 = output_pts[2][0]
    xp4 = output_pts[3][0]
    yp1 = output_pts[0][1]
    yp2 = output_pts[1][1]
    yp3 = output_pts[2][1]
    yp4 = output_pts[3][1]
    
    A = np.matrix([[-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1], 
                   [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1], 
                   [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2], 
                   [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
                   [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3], 
                   [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
                   [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4], 
                   [0, 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]])   
    U, Sigma, V = np.linalg.svd(A)   
    H = np.reshape(V[-1, :], (3, 3))
    Lambda = H[-1,-1]
    H = H/Lambda   
    return H

def warp(H,img,max_height,max_width):
    H_inv=np.linalg.inv(H)
    warp=np.zeros((max_height,max_width,3),np.uint8)
    for a in range(max_height):
        for b in range(max_width):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            warp[a][b] = img[int(y/z)][int(x/z)]
    warp = imutils.rotate(warp,90)
    return warp


def projectionMatrix(H, K):      
    
    h1 = H[:,0]          #taking column vectors h1,h2 and h3
    h2 = H[:,1]
    h3 = H[:,2]
    #calculating lamda
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    bt = lamda * np.matmul(np.linalg.inv(K),H)

    det = np.linalg.det(bt)

    if det > 0:
        b = bt
    else:                    
        b = -1 * bt  
        
    row1 = b[:, 0]
    row2 = b[:, 1]

    row3 = np.cross(row1.T, row2.T)
    row3 = row3.T
    
    t = b[:, 2]
    Rt = np.column_stack((row1, row2, row3, t))
    P = np.matmul(K,Rt)  
    return(P,Rt,t)



def wraps(image , H , dst):
    #assuming grayscale image
    im = cv2.transpose(image)
    y , x = image.shape[:2]
    xlim ,ylim = dst.shape[:2]

    
    # new_image = np.zeros((x,y),dtype = np.float32)
    for row in range(x):
        for col in range(y):
            p1 = ([x,y,1])
            pixel_value = image[x,y]
            p2 = H.dot(p1)
            x2 = p2[0]/p2[2]
            y2 = p2[1]/p2[2]
            if (x > 0 & x < xlim) and (y > 0 & y < ylim):
                dst[x2,y2] = pixel_value

    return dst

def flip(im):
    return cv2.flip(im,-1)

def euclediandist(a,b): 
    dist = np.sqrt(((a[0]-b[0]) ** 2) + ((a[1]-b[1]) ** 2)) 
    return (dist)

def arcorners(out):
    p1 = out[0]
    p2 = out[1]
    p3 = out[2]
    bl = out[3]
    width_1 = euclediandist(p3,bl)
    width_2 = euclediandist(p2,p1)
    max_w = max(int(width_1), int(width_2))

    height_1 = euclediandist(p3,p2)
    height_2 = euclediandist(bl,p1)
    max_h = max(int(height_1), int(height_2))

    dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]], dtype = "float32")

    return (p1,p2,p3,bl,max_w,max_h,dst)

def Corner_Detection(img):
        img = np.float32(img)

        corners = cv2.goodFeaturesTop2ack(img,9,0.01,100)
        corners = np.int0(corners)

        return corners

def sort_pts(points):
    rect = rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis = 1) 
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points,axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect

def final_corners(img):
    corners = Corner_Detection(img)
    points = []
    rect = []
    corners = corners.reshape((-1,2))
    y = np.sort(corners[:,1])
    y_max = y[-1]
    y_min = y[0]
    
    corners = np.delete(corners,np.where(corners[:,1] == y_max)[0],0)
    corners = np.delete(corners,np.where(corners[:,1] == y_min)[0],0)

    x = np.sort(corners[:,0])
    x_max = x[-1]
    x_min = x[0]

    corners = np.delete(corners,np.where(corners[:,0] == x_max)[0],0)
    corners = np.delete(corners,np.where(corners[:,0] == x_min)[0],0)
    
    y = np.sort(corners[:,1])
    y_max = y[-1]
    y_min = y[0]

    points.append(corners[np.where(corners[:,1] == y_max)[0]][0])    
    points.append(corners[np.where(corners[:,1] == y_min)[0]][0])
    
    x = np.sort(corners[:,0])
    x_max = x[-1]
    x_min = x[0]
    
    points.append(corners[np.where(corners[:,0] == x_max)[0]][0])    
    points.append(corners[np.where(corners[:,0] == x_min)[0]][0])
    points = np.array(points)
    points = points.reshape((-1,2))
    points = sort_pts(points)
    return np.array(points)

#return 2nd 4 highest corners for an image
def give_corners(coo):
    xmin = 3000
    xmax = 0
    ymin = 3000
    ymax = 0

    for i in coo:
        if i[0][0] < xmin:
            xmin = i[0][0]
        
    print("xmin = " , xmin)
        
    for i in coo:
        if i[0][1] < ymin:
            ymin = i[0][1]
    print("ymin = " , ymin)

    for i in coo:
        if i[0][0] > xmax:
            xmax = i[0][0]
            
    print("xmax = " , xmax)

    for i in coo:
        if i[0][1] > ymax:
            ymax = i[0][1]

    print("ymax = " , ymax)

    x = []
    y = []
    for i in coo:
        x.append(i[0][0])
        y.append(i[0][1])

    xmax_loc = np.where(x == xmax)
    ymax_loc = np.where(y == ymax)
    xmin_loc = np.where(x == xmin)
    ymin_loc = np.where(y == ymin)

    print(int(xmax_loc[0]))

    corner1 = [xmax , y[ int(xmax_loc[0]) ] ]
    corner2 = [xmin , y[int(xmin_loc[0]) ] ]
    corner3 = [x[ int(ymax_loc[0])] , ymax ] 
    corner4 = [x[int(ymin_loc[0])] , ymin ]


    coi = []
    coi.append(corner1)
    coi.append(corner2)
    coi.append(corner3)
    coi.append(corner4)
    # print(coi)
    for i in coi:
        x.remove(i[0])
        y.remove(i[1])

    x_max = 0
    x_min = 3000

    y_max  = 0
    y_min = 3000

    for i in x:
        if i < x_min:
            x_min = i
        if i > x_max:
            x_max = i
            
    for i in y:
        if i < y_min:
            y_min = i
        if i > y_max:
            y_max = i
    # 1065 , 841 , 526 , 307
    print(x_max , x_min , y_max , y_min)

    x_max_loc = np.where(x == x_max)
    y_max_loc = np.where(y == y_max)
    x_min_loc = np.where(x == x_min)
    y_min_loc = np.where(y == y_min)

    print(int(xmax_loc[0]))

    corner1 = [x_max , y[ int(x_max_loc[0]) ] ]
    corner2 = [x_min , y[ int(x_min_loc[0]) ] ]
    corner3 = [x[ int(y_max_loc[0]) ] , y_max ] 
    corner4 = [x[ int(y_min_loc[0]) ] , y_min ] 

    coii = []
    coii.append(corner1)
    coii.append(corner2)
    coii.append(corner3)
    coii.append(corner4)

    return coii

#returns 
def corner_points(image):
    operatedImage = image.copy()
    
    gray = cv2.cvtColor(operatedImage , cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
    ret,thresh = cv2.threshold(blur,220,255,cv2.THRESH_BINARY)

    coo = cv2.goodFeaturesToTrack(thresh , 10 , 0.01 ,70)
    coo = np.int0(coo)

    final_points = give_corners(coo)

    return thresh ,  final_points

def mean(image , start , length):
    sum = 0
    roi =image[start[0]: start[0] + length, start[1]: start[1] + length]
    for row in range(len(roi.shape[0])):
        for col in range(len(roi.shape[1])):
            sum += roi[row][col]
        mean =sum/(length**2)



def rotate(points):
    coords =points
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    t = (sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::1]))) % 360))
    return (t)


def finalpts(out):
    tl = out[0]
    tr = out[1]
    br = out[2]
    bl = out[3]
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

    return (tl,tr,br,bl,maxWidth,maxHeight,dst)

def Corner_Detection(img):
        img = np.float32(img)

        corners = cv2.goodFeaturesToTrack(img,9,0.01,100)
        corners = np.int0(corners)
        return corners

def order_points(points):
    rect = rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis = 1) 
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points,axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect

def tag_corners(img):
    corners = Corner_Detection(img)
    points = []
    rect = []
    corners = corners.reshape((-1,2))
    y = np.sort(corners[:,1])
    y_max = y[-1]
    y_min = y[0]
    
    corners = np.delete(corners,np.where(corners[:,1] == y_max)[0],0)
    corners = np.delete(corners,np.where(corners[:,1] == y_min)[0],0)

    x = np.sort(corners[:,0])
    x_max = x[-1]
    x_min = x[0]

    corners = np.delete(corners,np.where(corners[:,0] == x_max)[0],0)
    corners = np.delete(corners,np.where(corners[:,0] == x_min)[0],0)
    
    y = np.sort(corners[:,1])
    y_max = y[-1]
    y_min = y[0]

    points.append(corners[np.where(corners[:,1] == y_max)[0]][0])    
    points.append(corners[np.where(corners[:,1] == y_min)[0]][0])
    
    x = np.sort(corners[:,0])
    x_max = x[-1]
    x_min = x[0]
    
    points.append(corners[np.where(corners[:,0] == x_max)[0]][0])    
    points.append(corners[np.where(corners[:,0] == x_min)[0]][0])
    points = np.array(points)
    points = points.reshape((-1,2))
    points = order_points(points)
    return np.array(points)
   
     

    
K = [[1346.100595	,0,	        932.1633975],   
    [ 0,	       1355.933136, 654.8986796],
    [0,	             0	,       1]]

def projectionMatrix(H, K):  
    h1 = H[:,0]          #taking column vectors h1,h2 and h3
    h2 = H[:,1]
    h3 = H[:,2]
    #calculating lamda
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    bt = lamda * np.matmul(np.linalg.inv(K),H)

    #check if determinant is greater than 0 ie. has a positive determinant when object is in front of camera
    det = np.linalg.det(bt)

    if det > 0:
        b = bt
    else:                    #else make it positive
        b = -1 * bt  
        
    row1 = b[:, 0]
    row2 = b[:, 1]                      #extract rotation and translation vectors
    row3 = np.cross(row1, row2)
    
    t = b[:, 2]
    Rt = np.column_stack((row1, row2, row3, t))
#     r = np.column_stack((row1, row2, row3))
    P = np.matmul(K,Rt)  
    return(P,Rt,t)


#enter projective matrix and get return the coordinates of cube on video plane

def getcubecoor(P , size = 128):
    x1, y1,z1 = P.dot([[0],[0],[0],[1]]).astype(int)
    x2 , y2 , z2 = P.dot([[0],[size] , [0],[1]]).astype(int)
    x3 , y3 , z3 = P.dot([[size] , [0] , [0] ,[1]]).astype(int)
    x4 , y4 , z4 = P.dot([[size],[size] , [0] ,[1]]).astype(int)

    x5 , y5 , z5 = P.dot([[0] , [0] , [-size] , [1]]).astype(int)
    x6 , y6 , z6 = P.dot([[0],[size] , [-size] , [1]]).astype(int)
    x7 , y7 , z7 = P.dot([[size] , [0],[-size],[1]]).astype(int)
    x8 , y8 , z8 = P.dot([[size] , [size] , [-size],[1]]).astype(int)
   

    X = [x1/z1 ,x2/z2 ,x3/z3 ,x4/z4 ,x5/z5 ,x6/z6 ,x7/z7 ,x8/z8]
    Y = [y1/z1 ,y2/z2 ,y3/z3 ,y4/z4 ,y5/z5 ,y6/z6 ,y7/z7 ,y8/z8]

    coordinates = np.dstack((X,Y))
    coordinates = coordinates.reshape((-1,2))
    return coordinates

def drawcube(image , coord):
    img = image.copy()
    for i in coord:
        x , y  = i.ravel()
        x = int(x)
        y = int(y)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        img = cv2.circle(img , (int(x),int(y)) ,2,(0,0,255) , -1)
        
        cube = cv2.line(img , tuple(coord[0].astype(int)) , tuple(coord[1].astype(int)) , (r,g,b) , 2)
        cube = cv2.line(img , tuple(coord[4].astype(int)) , tuple(coord[5].astype(int)) , (r,g,b) , 2)
        cube = cv2.line(img , tuple(coord[0].astype(int)) , tuple(coord[2].astype(int)) , (r,g,b) , 2)
        cube = cv2.line(img , tuple(coord[0].astype(int)) , tuple(coord[4].astype(int)) , (r,g,b) , 2)
        cube = cv2.line(img , tuple(coord[1].astype(int)) , tuple(coord[3].astype(int)) , (r,g,b) , 2)
        cube = cv2.line(img , tuple(coord[1].astype(int)) , tuple(coord[5].astype(int)) , (r,g,b) , 2)
        cube = cv2.line(img , tuple(coord[2].astype(int)) , tuple(coord[6].astype(int)) , (r,g,b) , 2)
        cube = cv2.line(img , tuple(coord[2].astype(int)) , tuple(coord[3].astype(int)) , (r,g,b) , 2)
        cube = cv2.line(img , tuple(coord[3].astype(int)) , tuple(coord[7].astype(int)) , (r,g,b) , 2)
        cube = cv2.line(img , tuple(coord[4].astype(int)) , tuple(coord[6].astype(int)) , (r,g,b) , 2)
        cube = cv2.line(img , tuple(coord[5].astype(int)) , tuple(coord[7].astype(int)) , (r,g,b) , 2)
        cube = cv2.line(img , tuple(coord[6].astype(int)) , tuple(coord[7].astype(int)) , (r,g,b) , 2)
    
    return cube




