import cv2 
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('marker.png')
img = image.copy()
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)


img = cv2.resize(img , (640,640),interpolation = cv2.INTER_AREA)
start_x = 80
start_y = 0
end_x = 80
end_y = 640

startv_x = 0
startv_y = 80
endv_x = 640
endv_y = 80
for i in range(8):
    cv2.line(img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 3, 3)
    start_x += 80
    end_x += 80
    
    cv2.line(img , (startv_x , startv_y) , (endv_x , endv_y) , (255 , 0,0),3,3)
    startv_y += 80
    endv_y += 80
plt.imshow(img)
plt.show()


end1 = np.array(thresh[160:240,160:240])
end2 = np.array(thresh[400:480,160:240])
end3 = np.array(thresh[160:240,400:480])

end1_median = int(np.median(end1))
end2_median = int(np.median(end2))
end3_median = int(np.median(end3))
# print(end1_median , end2_median , end3_median)



while (end1_median != 0) or (end2_median != 0 ) or (end3_median != 0):
    thresh = np.rot90(thresh)
    end1 = np.array(thresh[160:240,160:240])
    end2 = np.array(thresh[400:480,160:240])
    end3 = np.array(thresh[160:240,400:480])

    end1_median = int(np.median(end1))
    end2_median = int(np.median(end2))
    end3_median = int(np.median(end3))
    print(end1_median , end2_median , end3_median)
    
box1 = np.array(image[240:320 , 240:320])
box2 = np.array(image[240:320 , 320:400])
box3 = np.array(image[320:400 , 320:400])
box4 = np.array(image[320:400 , 240:320])


box1_median = int(np.median(box1))
box2_median = int(np.median(box2))
box3_median = int(np.median(box3))
box4_median = int(np.median(box4))
                  
# print(box1_median , box2_median , box3_median , box4_median)
str = ""
if box1_median == 255:
    str = str + '1'
elif box1_median == 1:
    str = str + '1'
else:
    str = str + '0'

    
if box2_median == 255:
    str = str + '1'
elif box2_median == 1:
    str = str + '1'
else:
    str = str + '0'
    
    
if box3_median == 255:
    str = str + '1'
elif box3_median == 1:
    str = str + '1'
else:
    str = str + '0'
    
    
if box4_median == 255:
    str = str + '1'
elif box4_median == 1:
    str = str + '1'
else:
    str = str + '0'

code = int(str , 2)
print("The correct code is :",code)


