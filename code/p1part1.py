import cv2 
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

img = cv2.imread('frame.png')

frame = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

x , y  = frame.shape
crow, ccol = int(x / 2), int(y / 2)

mask = np.ones((x,y) , dtype = np.uint8)

r = 100
center = [crow, ccol]
x, y = np.ogrid[:x, :y]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 0

# plt.imshow(mask, cmap = 'gray')
# plt.show()
fft = np.fft.fftshift(np.fft.fft2(frame))

ffto = np.log(abs(fft))

# plt.imshow(ffto, cmap='gray')
# plt.show()

fft_img = fft*mask

freq_filt_img = np.fft.ifft2(np.fft.ifftshift(fft_img))

freq_filt_img = np.abs(freq_filt_img)
freq_filt_img = freq_filt_img.astype(np.uint8)

crop = freq_filt_img[300:550 , 770:1080]


fig = plt.figure(figsize=(10, 7))
fig.add_subplot(2, 2, 1)  
plt.imshow(img , cmap = 'gray')
plt.axis('off')
plt.title("Image")
fig.add_subplot(2, 2, 2)
# showing image
plt.imshow(mask , cmap = 'gray')
plt.axis('off')
plt.title("Mask")
# Adds a subplot at the 3rd position
fig.add_subplot(2, 2, 3) 
# showing image
plt.imshow(ffto , cmap = 'gray')
plt.axis('off')
plt.title("FFT Output") 
# Adds a subplot at the 4th position
fig.add_subplot(2, 2, 4)
plt.imshow(crop , cmap = 'gray')
plt.axis('off')
plt.title("Final Croped")
plt.show()
